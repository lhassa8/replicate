"""Loopback TLS source review with a separate verifier process and restart.

Eight public fixture sources are used. No model or native Linux device runs.
The harness provisions all lab identities; process separation is not OS or
physical-host isolation. Private keys are excluded from retained evidence.
"""

import argparse
import asyncio
import base64
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import tempfile
import time
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .capsule import canonical
from .discovery_book import DiscoveryBook, EXPERIENCE_SCHEMA, sign_discovery_book
from .enrollment import Enrollment, public_bytes, sign_enrollment
from .experience import VerifiedExperience, require
from .experience_endorsement import CHECKER_SOURCES, checker_identity, review_source
from .experience_exchange import ExperienceExchange
from .experience_proposal import MAX_APPROVAL_BYTES, accept, check_approval, propose, sha
from .experience_store import ExperienceStore, verify_copy_receipt
from .inference_server import RuntimeConfig
from .package_application import APPLICATION_MODULES
from .peer_identity import context, device_certificate, owner_certificate, write_identity
from .probe_experience import NOW, RELEASE, current_task, lab, source_episode
from .probe_separate_endorsement import endorse_request, fixture_reviewed_plans
from .review_inbox import ReviewInbox, packet_hash


PROFILES = tuple("abcdefgh")
SOURCES = tuple(sorted({name + ".py" for name in APPLICATION_MODULES} | set(CHECKER_SOURCES) | {
    "probe_source_review.py", "probe_experience.py", "probe_separate_endorsement.py",
    "equipment_development.py", "equipment_baseline.py", "package_application.py"}))


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fixture_config(work, identity, index):
    model, executable = work / "fixture-model", work / "fixture-executable"
    model.write_bytes(b"Explicit fixture: no model weights")
    executable.write_bytes(b"Explicit fixture: never executed")
    executable.chmod(0o700)
    return RuntimeConfig(model, sha(model.read_bytes()), model.stat().st_size,
                         executable, sha(executable.read_bytes()), uuid.uuid4().hex,
                         identity, f"{index + 1:032x}", RELEASE, False, False)


def verifier_worker(control, config, enrollment, owner_public, signed_book, secret, identity_path,
                    reviewed_plans, drop_first):
    async def serve():
        policy = Enrollment(enrollment, owner_public)
        book = DiscoveryBook(signed_book, policy, owner_public)
        key = Ed25519PrivateKey.from_private_bytes(secret)
        calls, dropped = 0, False

        def reviewer(proposal, observation, job):
            nonlocal calls
            calls += 1
            return endorse_request({"proposal": proposal, "raw": observation, "job": job},
                                   policy, key, reviewed_plans)

        inbox = ReviewInbox(Path(identity_path).parent / "source-review", config, policy,
                            checker_id=checker_identity()[0], reviewer=reviewer,
                            require_live_source=False, clock=lambda: NOW)
        store = ExperienceStore(Path(identity_path).parent / "experience", policy, key, release_id=RELEASE,
                                allowed_checkers=book.experience_checkers, require_live_source=False, clock=lambda: NOW)
        exchange = ExperienceExchange(config, store, book,
            context(policy, config.device_id, identity_path),
            context(policy, config.device_id, identity_path, server=True), reviews=inbox)
        from . import experience_exchange as wire
        original_write = wire.write_blob

        async def maybe_drop(writer, raw, maximum):
            nonlocal dropped
            if drop_first and not dropped and maximum == MAX_APPROVAL_BYTES:
                dropped = True
                writer.transport.abort()
                raise ConnectionResetError("Injected response loss after committed source review")
            await original_write(writer, raw, maximum)

        wire.write_blob = maybe_drop
        try:
            await exchange.start()
            control.send({"ready": True, "pid": os.getpid(), "inbox": inbox.snapshot()})
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and not control.poll():
                if exchange.failure:
                    raise RuntimeError("Verifier handler failed: " + exchange.failure)
                await asyncio.sleep(0.01)
            require(control.poll() and control.recv() == "stop", "Verifier did not receive its bounded stop")
        finally:
            await exchange.close()
            wire.write_blob = original_write
        return {"stopped": True, "pid": os.getpid(), "reviewer_calls": calls, "reply_dropped": dropped,
                "inbox": inbox.snapshot(), "handlers_remaining": len(exchange.handlers),
                "listener_closed": exchange.listener is None, "producer_private_key_provided": False}

    try:
        control.send(asyncio.run(serve()))
    except BaseException as exc:
        control.send({"failure": type(exc).__name__ + ": " + str(exc), "pid": os.getpid()})
        raise
    finally:
        control.close()


async def worker_message(pipe, seconds=10):
    require(await asyncio.to_thread(pipe.poll, seconds), "Verifier control response timed out")
    value = pipe.recv()
    require("failure" not in value, "Verifier failed: " + str(value.get("failure")))
    return value


async def stop_worker(process, pipe):
    pipe.send("stop")
    report = await worker_message(pipe)
    await asyncio.to_thread(process.join, 5)
    require(not process.is_alive() and process.exitcode == 0, "Verifier exit is not successful and terminal")
    report["exitcode"] = process.exitcode
    pipe.close()
    return report


def audit_saved(output):
    report = json.loads((output / "report.json").read_bytes())
    lab_raw, plans_raw = (output / "lab.json").read_bytes(), (output / "reviewed-plans.json").read_bytes()
    require(sha(lab_raw) == report["lab_sha256"], "Retained lab context differs")
    saved, plans = json.loads(lab_raw), json.loads(plans_raw)
    require(sha(plans_raw) == saved["reviewed_plans_sha256"] and plans == fixture_reviewed_plans(),
            "Independent fixture procedure inventory differs")
    policy = Enrollment(base64.b64decode(saved["enrollment"]), base64.b64decode(saved["owner_public"]))
    book = DiscoveryBook(base64.b64decode(saved["book"]), policy, base64.b64decode(saved["owner_public"]))
    checker = checker_identity()[0]
    require(book.experience_checkers == {checker}, "Source checker differs")
    require(report["failure"] is None and len(report["verifier_incarnations"]) == 2
            and report["verifier_incarnations"][0]["reply_dropped"]
            and all(item["exitcode"] == 0 and item["listener_closed"] and item["handlers_remaining"] == 0
                    for item in report["verifier_incarnations"]), "Verifier lifecycle or lost-response check failed")
    require([item["reviewer_calls"] for item in report["verifier_incarnations"]] == [1, 7],
            "Restart repeated a previously committed review")
    require(all(item["pid"] != report["producer_pid"] for item in report["verifier_incarnations"]),
            "Verifier was not a separate process")
    require(report["producer_review_attempts"] == {
        f"{index:032x}": 2 if index == 1 else 1 for index in range(1, 9)},
        "Producer review attempts did not survive restart")
    require(set(report["source_hashes"]) == set(SOURCES), "Source inventory differs")
    for name, expected in report["source_hashes"].items():
        require(sha((output / "sources" / name).read_bytes()) == expected
                == sha(Path(__file__).with_name(name).read_bytes()), "Probe source changed")
    config = RuntimeConfig(**{**saved["verifier_config"],
        "model": Path(saved["verifier_config"]["model"]), "executable": Path(saved["verifier_config"]["executable"])})
    inbox = ReviewInbox(output / "work-evidence/verifier", config, policy, checker_id=checker,
                        reviewer=lambda *args: None, require_live_source=False, clock=lambda: NOW)
    require(inbox.snapshot()["decisions"] == 8 and inbox.snapshot()["approved"] == 8,
            "Retained verifier decisions are incomplete")
    notes = {}
    for index, profile in enumerate(PROFILES, 1):
        item = json.loads((output / ("source-" + profile + ".json")).read_bytes())
        raw = canonical(item["case"]["document"])
        proposal, approval, record = (base64.b64decode(item[key]) for key in ("proposal", "approval", "record"))
        body, _ = check_approval(proposal, approval, policy=policy, release_id=RELEASE,
                                 allowed_checkers={checker}, now=NOW, require_live_source=False)
        expected, receipt = review_source(proposal, raw, item["job"], plans[body["capability_sha256"]],
            policy=policy, verifier_device_id=saved["device_ids"][1], release_id=RELEASE,
            created=NOW, expires=NOW + 3600, require_live_source=False)
        require(body == expected, "Independent source verification differs from approved record")
        decision = inbox.get(saved["device_ids"][0], f"{index:032x}")
        require(decision["approval"] == approval and decision["job"] == item["job"]
                and decision["observation"] == raw and decision["validation"] == canonical(receipt)
                and decision["packet_sha256"] == packet_hash(proposal, raw, canonical(item["job"])),
                "Committed source decision differs from independent readback")
        note = VerifiedExperience(record, policy, release_id=RELEASE, allowed_checkers={checker},
                                  now=NOW, require_live_source=False)
        require(note.body == body and note.body["source"]["kind"] == "development_fixture",
                "Final record differs or fixture provenance changed")
        notes[note.note_id] = record
        from .audit_equipment_diagnostics import inspect_job
        target = json.loads((output / ("target-" + profile + ".json")).read_bytes())
        require(inspect_job(target["fixture"], canonical(target["fixture"]["document"]),
                            target["job"])["semantic_passed"], "Fresh fixture diagnosis failed")
        require(target["preparation"]["note_id"] == note.note_id, "Target used another inherited record")
    receipts, received, confirmed = 0, {}, {}
    for label, index, expected_head in (("retained-parent", 0, (8, 0, 8)),
                                       ("child", 2, (8, 8, 8)), ("grandchild", 3, (8, 8, 0))):
        path = output / "work-evidence" / label / "experience.sqlite3"
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            require(dict(db.execute("SELECT note_id,raw FROM notes")) == notes, "Retained procedure copies differ")
            head = db.execute("SELECT notes,issued,confirmed FROM head WHERE singleton=1").fetchone()
            actual = tuple(db.execute("SELECT count(*) FROM " + table).fetchone()[0]
                           for table in ("notes", "issued", "confirmed"))
            require(head == actual == expected_head, "Retained experience head differs")
            received[label], confirmed[label] = {}, {}
            for sender, request, note_id, raw in db.execute("SELECT sender,request,note_id,raw FROM issued"):
                verify_copy_receipt(raw, policy, release_id=RELEASE, note_id=note_id, sender_id=sender,
                                    receiver_id=saved["device_ids"][index], request_id=request, now=NOW)
                received[label][note_id] = raw
                receipts += 1
            for note_id, receiver, request, raw in db.execute("SELECT note_id,receiver,request,raw FROM confirmed"):
                verify_copy_receipt(raw, policy, release_id=RELEASE, note_id=note_id,
                                    sender_id=saved["device_ids"][index], receiver_id=receiver,
                                    request_id=request, now=NOW)
                confirmed[label][note_id] = raw
    require(confirmed["retained-parent"] == received["child"]
            and confirmed["child"] == received["grandchild"], "Sender and receiver copy receipts differ")
    return {"passed": True, "approved_fixture_sources": 8, "reviewer_calls": 8,
            "verifier_incarnations": 2, "signed_copy_receipts": receipts, "retained_record_copies": 24,
            "fresh_fixture_diagnoses": 8, "saved_source_decisions_independently_reconstructed": True,
            "model_calls": 0, "native_execution_verified": False, "qualification_passed": False}


async def run(output):
    output.mkdir(parents=True, exist_ok=False)
    plans = fixture_reviewed_plans()
    write(output / "reviewed-plans.json", plans)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "producer_pid": os.getpid(),
              "scope": "Separate-process TLS review and restart using eight public fixture sources",
              "source_hashes": {}, "verifier_incarnations": [], "failure": None, "model_calls": 0,
              "native_execution_verified": False, "physical_host_independence_verified": False,
              "os_security_isolation_verified": False, "private_keys_retained_in_evidence": False,
              "qualification_passed": False}
    (output / "sources").mkdir()
    for name in SOURCES:
        raw = Path(__file__).with_name(name).read_bytes()
        (output / "sources" / name).write_bytes(raw)
        report["source_hashes"][name] = sha(raw)
    write(output / "request.json", report)
    started = time.monotonic()
    processes, pipes, exchanges = [], [], []
    with tempfile.TemporaryDirectory(prefix="review-process-", dir="/private/tmp") as temporary:
        root = Path(temporary)
        initial, _, keys, identities = lab()
        owner = Ed25519PrivateKey.generate()
        owner_public = public_bytes(owner.public_key())
        policy = Enrollment(sign_enrollment(initial.body, owner), owner_public)
        reserved = [socket.socket() for _ in range(8)]
        try:
            for item in reserved:
                item.bind(("127.0.0.1", 0))
            ports = [item.getsockname()[1] for item in reserved]
        finally:
            for item in reserved:
                item.close()
        body = {"schema": EXPERIENCE_SCHEMA, "policy_id": policy.policy_id, "epoch": 1,
                "lab_cidrs": ["127.0.0.0/24"], "experience_checkers": [checker_identity()[0]],
                "endpoints": {identity: {"address": "127.0.0.1", "port": ports[index],
                                        "experience_port": ports[index + 4]}
                              for index, identity in enumerate(identities)}}
        book = DiscoveryBook(sign_discovery_book(body, owner, policy), policy, owner_public)
        cert = owner_certificate(owner)
        work, configs, credentials, stores = [], [], [], []
        for index, identity in enumerate(identities):
            directory = root / str(index)
            directory.mkdir(mode=0o700)
            work.append(directory)
            configs.append(fixture_config(directory, identity, index))
            credentials.append(write_identity(directory / "identity", keys[index],
                device_certificate(owner, cert, keys[index].public_key()), cert))
            stores.append(ExperienceStore(directory / "experience", policy, keys[index], release_id=RELEASE,
                          allowed_checkers=book.experience_checkers, require_live_source=False, clock=lambda: NOW))

        def exchange(index):
            return ExperienceExchange(configs[index], stores[index], book,
                context(policy, identities[index], credentials[index]),
                context(policy, identities[index], credentials[index], server=True))

        async def start_verifier(drop):
            parent, child = multiprocessing.get_context("spawn").Pipe()
            process = multiprocessing.get_context("spawn").Process(target=verifier_worker, args=(
                child, configs[1], policy.signed, owner_public, book.signed, keys[1].private_bytes_raw(),
                credentials[1], plans, drop))
            processes.append(process)
            pipes.append(parent)
            process.start()
            child.close()
            ready = await worker_message(parent)
            require(ready["ready"] and ready["pid"] == process.pid, "Verifier did not reach service readiness")
            print(json.dumps({"verifier": "serving", "pid": process.pid, "drop_first_reply": drop}), flush=True)
            return process, parent

        cfg = asdict(configs[1])
        cfg.update(model=str(configs[1].model), executable=str(configs[1].executable))
        saved = {"enrollment": base64.b64encode(policy.signed).decode(),
                 "owner_public": base64.b64encode(owner_public).decode(),
                 "book": base64.b64encode(book.signed).decode(), "device_ids": identities,
                 "verifier_config": cfg, "reviewed_plans_sha256": sha((output / "reviewed-plans.json").read_bytes())}
        write(output / "lab.json", saved)
        report["lab_sha256"] = sha((output / "lab.json").read_bytes())
        observations = root / "observations"
        observations.mkdir()
        try:
            exchanges = [exchange(index) for index in range(4)]
            await exchanges[2].start()
            process, pipe = await start_verifier(True)
            for index, profile in enumerate(PROFILES, 1):
                fixture, raw, job, _ = source_episode(observations, profile)
                request_id = f"{index:032x}"
                proposal = propose(raw, job, policy=policy, producer_key=keys[0],
                    producer_individual_id=configs[0].individual_id, verifier_device_id=identities[1],
                    release_id=RELEASE, request_id=request_id, source_clock=NOW, live=False)
                if index == 1:
                    try:
                        await exchanges[0].request_review(identities[1], proposal, raw, job)
                    except (ValueError, OSError, asyncio.IncompleteReadError):
                        pass
                    else:
                        raise ValueError("Expected injected approval-response loss")
                    report["verifier_incarnations"].append(await stop_worker(process, pipe))
                    await exchanges[0].close()
                    exchanges[0] = exchange(0)
                    configs[1] = replace(configs[1], start_token=uuid.uuid4().hex)
                    process, pipe = await start_verifier(False)
                decision = await exchanges[0].request_review(identities[1], proposal, raw, job)
                require(decision["status"] == "approved" and decision["cached"] == (index == 1),
                        "Verifier did not preserve the expected decision across restart")
                record = accept(proposal, decision["approval"], policy=policy, producer_key=keys[0],
                                release_id=RELEASE, allowed_checkers=book.experience_checkers,
                                now=NOW, require_live_source=False)
                note_id = stores[0].add(record)
                await exchanges[0].deliver(identities[2], note_id, request_id=f"{100 + index:032x}")
                write(output / ("source-" + profile + ".json"), {
                    "case": fixture, "job": job, "proposal": base64.b64encode(proposal).decode(),
                    "approval": base64.b64encode(decision["approval"]).decode(),
                    "record": base64.b64encode(record).decode(), "cached": decision["cached"]})
            report["verifier_incarnations"].append(await stop_worker(process, pipe))
            report["producer_review_attempts"] = {
                request: count for (peer, request), (_, count) in exchanges[0].review_attempts.items()}
            await exchanges[0].close()
            shutil.move(stores[0].directory, root / "retained-parent")
            await exchanges[2].close()
            stores[2] = ExperienceStore(work[2] / "experience", policy, keys[2], release_id=RELEASE,
                allowed_checkers=book.experience_checkers, require_live_source=False, clock=lambda: NOW)
            exchanges[2] = exchange(2)
            await exchanges[2].start()
            await exchanges[3].start()
            for _ in range(4):
                await exchanges[2].round()
            require(len(stores[3].inventory()) == 8, "Reopened child did not forward all retained procedures")
            for profile in PROFILES:
                stored = json.loads((output / ("source-" + profile + ".json")).read_bytes())
                record = base64.b64decode(stored["record"])
                note = VerifiedExperience(record, policy, release_id=RELEASE,
                    allowed_checkers=book.experience_checkers, now=NOW, require_live_source=False)
                write(output / ("target-" + profile + ".json"),
                      current_task(observations, profile, stores[3].get(note.note_id)))
            report["parent_store_removed_from_active_path"] = True
            report["child_store_reopened_before_forwarding"] = True
        except Exception as exc:
            report["failure"] = type(exc).__name__ + ": " + str(exc)
        finally:
            for item in exchanges:
                await item.close()
            for process, pipe in zip(processes, pipes):
                if process.is_alive():
                    try:
                        await stop_worker(process, pipe)
                    except Exception:
                        process.terminate()
                        await asyncio.to_thread(process.join, 5)
                require(not process.is_alive(), "Owned verifier process did not stop")
            retained = output / "work-evidence"
            retained.mkdir()
            for label, source in (("verifier", work[1] / "source-review"),
                                  ("retained-parent", root / "retained-parent"),
                                  ("child", work[2] / "experience"), ("grandchild", work[3] / "experience")):
                if source.exists():
                    shutil.copytree(source, retained / label)
    report["seconds"] = round(time.monotonic() - started, 3)
    report["private_key_files_found_in_evidence"] = [
        str(path.relative_to(output)) for path in output.rglob("*.pem")]
    require(not report["private_key_files_found_in_evidence"], "Private-key files entered retained evidence")
    write(output / "report.json", report)
    try:
        audit = audit_saved(output)
    except Exception as exc:
        audit = {"passed": False, "failure": type(exc).__name__ + ": " + str(exc), "qualification_passed": False}
    write(output / "audit.json", audit)
    print(json.dumps({"report": str(output / "report.json"), "audit_passed": audit["passed"],
                      "failure": report["failure"]}), flush=True)
    return report, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, audit = asyncio.run(run(args.output.resolve()))
    if report["failure"] or not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
