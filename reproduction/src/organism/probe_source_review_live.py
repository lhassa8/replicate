"""One actual Mac stock diagnostic followed by automatic separate-process review.

The harness provisions a public profile-b procedure to the reviewer before the
model runs. It supplies no model answer or review request. This is a component
check on one Mac, not native, trained-model, physical-host or Q7 qualification.
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

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .application import AgentApplication, local_request
from .audit_equipment_diagnostics import inspect_job
from .audit_inherited_diagnostics import inspect_inherited_job
from .capsule import canonical
from .discovery_book import DiscoveryBook, EXPERIENCE_SCHEMA, sign_discovery_book
from .diagnostic_oracle import reconstruct
from .enrollment import Enrollment, public_bytes, sign_enrollment
from .equipment_development import case
from .experience import VerifiedExperience, require
from .experience_endorsement import CHECKER_SOURCES, checker_identity, endorse, review_source
from .experience_exchange import ExperienceExchange
from .experience_proposal import check_approval, check_proposal, sha, source_facts
from .experience_store import ExperienceStore
from .inference_server import RuntimeConfig
from .package_application import APPLICATION_MODULES
from .peer_identity import context, device_certificate, owner_certificate, write_identity
from .probe_experience import lab
from .probe_separate_endorsement import fixture_reviewed_plans
from .probe_source_review import stop_worker, worker_message
from .review_inbox import ReviewInbox
from .source_journal import SourceJournal
from .source_inventory import check_source_inventory


RELEASE = sha(b"Mac automatic source-review development component, not a complete release")
SOURCES = tuple(sorted({name + ".py" for name in APPLICATION_MODULES} | set(CHECKER_SOURCES) | {
    "probe_source_review_live.py", "probe_source_review.py", "probe_experience.py",
    "probe_separate_endorsement.py", "equipment_development.py", "equipment_baseline.py",
    "package_application.py"}))


def write(path, value):
    with path.open("x") as output:
        output.write(json.dumps(value, indent=2) + "\n")


def verifier_worker(control, config, enrollment, owner_public, signed_book, secret, identity_path, plans):
    async def serve():
        policy = Enrollment(enrollment, owner_public)
        book = DiscoveryBook(signed_book, policy, owner_public)
        key = Ed25519PrivateKey.from_private_bytes(secret)
        calls = 0

        def reviewer(proposal, observation, job, **kwargs):
            nonlocal calls
            calls += 1
            source = check_proposal(proposal, policy, release_id=RELEASE)
            require(source["capability_sha256"] in plans, "No independently provisioned procedure")
            created = int(time.time())
            return endorse(proposal, observation, job, plans[source["capability_sha256"]],
                           policy=policy, verifier_key=key, release_id=RELEASE, created=created,
                           expires=created + 3600, require_live_source=True, **kwargs)

        work = Path(identity_path).parent
        inbox = ReviewInbox(work / "source-review", config, policy, checker_id=checker_identity()[0],
                            reviewer=reviewer, inventory_key=key, inventory_checkers=book.experience_checkers)
        store = ExperienceStore(work / "experience", policy, key, release_id=RELEASE,
                                allowed_checkers=book.experience_checkers)
        exchange = ExperienceExchange(config, store, book, context(policy, config.device_id, identity_path),
                                      context(policy, config.device_id, identity_path, server=True), reviews=inbox)
        try:
            await exchange.start()
            control.send({"ready": True, "pid": os.getpid()})
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline and not control.poll():
                require(exchange.failure is None, "Verifier handler failed")
                await asyncio.sleep(0.02)
            require(control.poll() and control.recv() == "stop", "Verifier bounded lifetime expired")
        finally:
            await exchange.close()
        return {"pid": os.getpid(), "stopped": True, "reviewer_calls": calls, "inbox": inbox.snapshot(),
                "handlers_remaining": len(exchange.handlers), "listener_closed": exchange.listener is None,
                "producer_private_key_provided": False, "model_executed_by_verifier": False}

    try:
        control.send(asyncio.run(serve()))
    except BaseException as exc:
        control.send({"failure": type(exc).__name__ + ": " + str(exc), "pid": os.getpid()})
        raise
    finally:
        control.close()


def inspect_committed_source(source, raw, case, policy, book, producer_id, verifier_id):
    inventory = source["source_inventory"]
    require(inventory is not None and inventory["context"] == source["job"]["record"]["trace"].get("source_inventory"),
            "Actual diagnostic has no committed inventory")
    records = check_source_inventory(
        inventory["snapshot"], inventory["receipt"], inventory["context"], policy,
        release_id=RELEASE, allowed_checkers=book.experience_checkers, require_live_source=True,
        producer_id=producer_id, request_id=source["request_id"], verifier_id=verifier_id)
    # This probe starts both stores empty before launching the diagnostic.
    # The expected starting inventory comes from that setup, not the trace.
    require(records == {} and json.loads(inventory["snapshot"])["body"]["head"] == [0, 0, 0],
            "Probe's independently fixed starting inventory changed")
    return inspect_inherited_job(
        case, raw, source["job"], policy=policy, release_id=RELEASE,
        allowed_checkers=book.experience_checkers, records=records, require_live_source=True,
        live=True, clock_window=[source["started"], source["ended"]])


def audit_saved(output):
    report = json.loads((output / "report.json").read_bytes())
    require(report["failure"] is None and report["server_returncode"] == 0
            and report["verifier"]["exitcode"] == 0 and report["verifier"]["reviewer_calls"] == 1,
            "Owned process completion or independent review is unconfirmed")
    for name, identity in report["source_hashes"].items():
        require(sha((output / "sources" / name).read_bytes()) == identity
                == sha(Path(__file__).with_name(name).read_bytes()), "Probe source changed")
    saved = json.loads((output / "lab.json").read_bytes())
    policy = Enrollment(base64.b64decode(saved["enrollment"]), base64.b64decode(saved["owner_public"]))
    book = DiscoveryBook(base64.b64decode(saved["book"]), policy, base64.b64decode(saved["owner_public"]))
    configs = [RuntimeConfig(**{**value, "model": Path(value["model"]), "executable": Path(value["executable"])})
               for value in saved["configs"]]
    producer, verifier = output / "work-evidence/producer", output / "work-evidence/verifier"
    source = SourceJournal(producer / "source-evidence", replace(configs[0], start_token=uuid.uuid4().hex))
    captured = source.get(report["request_id"])
    raw = (output / "observation.json").read_bytes()
    diagnostic = json.loads((output / "diagnostic.json").read_bytes())
    require(captured["observations"][sha(raw)] == raw and captured["job"] == diagnostic["job"],
            "Captured source differs from the actual diagnostic")
    plans = fixture_reviewed_plans()
    facts = source_facts(raw, captured["job"])
    reads = [event["result"] for event in captured["job"]["record"]["trace"]["tool_events"]
             if event["tool"] == "read_equipment" and event["result"].get("status") == "observed"]
    expected = reconstruct(raw, plans[facts["capability_sha256"]], now=reads[-1]["read_unix_seconds"])
    document = json.loads(raw)
    independent_case = {"case_id": document["device_id"], "document": document,
                        **{name: expected[name] for name in (
                            "expected_reader", "expected_status", "expected_value", "expected_unit")}}
    checked = inspect_committed_source(captured, raw, independent_case, policy, book,
                                       configs[0].device_id, configs[1].device_id)
    require(checked == diagnostic["evaluation"], "Independent source-diagnostic readback differs")
    inbox = ReviewInbox(verifier / "source-review", configs[1], policy, checker_id=checker_identity()[0],
                        reviewer=lambda *args: None)
    decision = inbox.get(configs[0].device_id, report["request_id"])
    require(decision is not None and decision["observation"] == raw and decision["job"] == captured["job"],
            "Verifier did not retain the exact producer source")
    independent_inventory = inbox.source_inventory_for(configs[0].device_id, report["request_id"], captured["job"])
    require(independent_inventory == captured["source_inventory"],
            "Producer inventory differs from the verifier's separate durable commitment")
    expected_state = "installed" if checked["semantic_passed"] else "rejected"
    with sqlite3.connect((producer / "experience/source-review-queue/queue.sqlite3").as_uri() + "?mode=ro",
                         uri=True) as db:
        rows = db.execute("SELECT * FROM requests").fetchall()
        head = db.execute("SELECT requests,bytes FROM head WHERE singleton=1").fetchone()
        require(len(rows) == 1 and head == (1, len(rows[0][2]) + len(rows[0][4])), "Queue history is incomplete")
    request, peer, proposal, state, approval, note_id = rows[0]
    require(request == report["request_id"] and peer == configs[1].device_id and state == expected_state
            and proposal == decision["proposal"] and approval == decision["approval"],
            "Automatic queue result differs from the independent source decision")
    proposed = check_proposal(proposal, policy, release_id=RELEASE)
    notes = {}
    if state == "installed":
        created = json.loads(approval)["body"]["created_unix_seconds"]
        body, _ = check_approval(proposal, approval, policy=policy, release_id=RELEASE,
                                allowed_checkers=book.experience_checkers, now=created)
        expected, validation = review_source(proposal, raw, captured["job"], plans[proposed["capability_sha256"]],
            policy=policy, verifier_device_id=peer, release_id=RELEASE,
            created=created, expires=body["expires_unix_seconds"], source_inventory=independent_inventory,
            inventory_checkers=book.experience_checkers)
        require(body == expected and canonical(validation) == decision["validation"],
                "Approved source does not independently reconstruct")
        with sqlite3.connect((producer / "experience/experience.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
            notes = dict(db.execute("SELECT note_id,raw FROM notes"))
        require(list(notes) == [note_id], "Producer did not retain exactly the approved procedure")
        VerifiedExperience(notes[note_id], policy, release_id=RELEASE,
                           allowed_checkers=book.experience_checkers, now=created)
    else:
        require(decision["status"] == "rejected" and approval == b"" and note_id is None,
                "Rejected source received an endorsement")
        with sqlite3.connect((producer / "experience/experience.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
            require(db.execute("SELECT count(*) FROM notes").fetchone()[0] == 0,
                    "Rejected source created a procedure")
    private_keys = [str(p.relative_to(output)) for p in output.rglob("*")
                    if p.is_file() and p.suffix == ".pem" and b"PRIVATE KEY" in p.read_bytes()]
    require(not private_keys, "Private key material was retained")
    return {"passed": True, "source_diagnostic_passed": checked["semantic_passed"],
            "automatic_queue_state": state, "independent_decisions": 1,
            "independent_pre_inference_inventory_commit_verified": True,
            "frozen_inventory_entries": 0,
            "producer_procedures": len(notes), "model_calls": checked["model_calls"],
            "actual_stock_model_used": True, "native_execution_verified": False,
            "physical_host_independence_verified": False, "qualification_passed": False}


async def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "failure": None, "source_hashes": {},
              "scope": "Actual Mac stock source with prior inventory commitment and automatic separate-process review",
              "trained_for_organism": False, "native_execution_verified": False, "qualification_passed": False}
    (output / "sources").mkdir()
    for name in SOURCES:
        raw = (root / "organism" / name).read_bytes()
        (output / "sources" / name).write_bytes(raw)
        report["source_hashes"][name] = sha(raw)
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="auto-review-live-", dir="/private/tmp") as temporary:
        work = Path(temporary)
        initial, _, keys, ids = lab()
        owner = Ed25519PrivateKey.generate()
        owner_public = public_bytes(owner.public_key())
        enrollment = sign_enrollment(initial.body, owner)
        policy = Enrollment(enrollment, owner_public)
        directories = [work / name for name in ("producer", "verifier", "other-a", "other-b")]
        for directory in directories:
            directory.mkdir(mode=0o700)
        contract = json.loads((root / "organism/experiment-contract.draft.json").read_bytes())
        base = RuntimeConfig(
            root / "models/granite-3b/granite-base-Q4_K_M.gguf", contract["model"]["local_export_sha256"],
            contract["model"]["local_export_bytes"], root / "tools/llama.cpp/b9850/llama-b9850/llama-server",
            contract["runtime"]["development_binary_sha256"], uuid.uuid4().hex, ids[0], uuid.uuid4().hex,
            RELEASE, False, False)
        configs = [replace(base, device_id=identity, individual_id=uuid.uuid4().hex, start_token=uuid.uuid4().hex)
                   for identity in ids[:2]]
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
                                        "experience_port": ports[index + 4]} for index, identity in enumerate(ids)}}
        book = DiscoveryBook(sign_discovery_book(body, owner, policy), policy, owner_public)
        certificate = owner_certificate(owner)
        identities = [write_identity(directory / "identity", key,
            device_certificate(owner, certificate, key.public_key()), certificate)
            for directory, key in zip(directories, keys)]
        write(output / "lab.json", {"enrollment": base64.b64encode(enrollment).decode(),
            "owner_public": base64.b64encode(owner_public).decode(), "book": base64.b64encode(book.signed).decode(),
            "configs": [{**asdict(config), "model": str(config.model), "executable": str(config.executable)}
                        for config in configs]})
        process_context = multiprocessing.get_context("spawn")
        pipe, child_pipe = process_context.Pipe()
        process = process_context.Process(target=verifier_worker, args=(
            child_pipe, configs[1], enrollment, owner_public, book.signed,
            keys[1].private_bytes_raw(), identities[1], fixture_reviewed_plans()))
        store = ExperienceStore(directories[0] / "experience", policy, keys[0], release_id=RELEASE,
                                allowed_checkers=book.experience_checkers)
        exchange = ExperienceExchange(configs[0], store, book, context(policy, ids[0], identities[0]),
                                      context(policy, ids[0], identities[0], server=True), interval=1)
        app = AgentApplication(configs[0], directories[0], experience=exchange)
        try:
            process.start()
            child_pipe.close()
            report["verifier_ready"] = await worker_message(pipe)
            await app.start()
            report["server_pid"] = app.inference.process.pid
            print(json.dumps({"model_pid": report["server_pid"], "verifier_pid": process.pid}), flush=True)
            item = case("b", "fault", time.time())
            raw = (json.dumps(item["document"], indent=3) + "\n").encode()
            (output / "observation.json").write_bytes(raw)
            (app.equipment_interfaces.directory / (item["case_id"] + ".json")).write_bytes(raw)
            request = report["request_id"] = uuid.uuid4().hex
            await local_request(app.socket, {"operation": "diagnose", "request_id": request,
                                            "device_id": item["case_id"]})
            async with asyncio.timeout(180):
                while app.busy or app.store.get(request)["record"]["status"] == "running":
                    await asyncio.sleep(0.05)
            captured = app.source_journal.get(request)
            require(captured["state"] == "awaiting_review", "Actual model did not produce a reviewable source")
            checked = inspect_committed_source(captured, raw, item, policy, book, ids[0], ids[1])
            write(output / "diagnostic.json", {"case": item, "job": captured["job"], "evaluation": checked})
            async with asyncio.timeout(20):
                while True:
                    rows = exchange.source_queue.records()
                    if rows and rows[0]["state"] in ("installed", "rejected", "exhausted"):
                        break
                    require(app.failure is None and exchange.failure is None, "Application or exchange failed")
                    await asyncio.sleep(0.05)
            report["queue"] = exchange.source_queue.snapshot()
            report["diagnostic_semantic_passed"] = checked["semantic_passed"]
            report["model_calls"] = checked["model_calls"]
            report["review_attempts"] = exchange.review_attempts[(ids[1], request)][1]
        except Exception as exc:
            report["failure"] = type(exc).__name__ + ": " + str(exc)
        finally:
            try:
                await app.close()
            except Exception as exc:
                report["failure"] = report["failure"] or "Application shutdown: " + str(exc)
            report["server_returncode"] = app.inference.process.returncode if app.inference.process else None
            if process.is_alive():
                try:
                    report["verifier"] = await stop_worker(process, pipe)
                except Exception as exc:
                    report["failure"] = report["failure"] or "Verifier shutdown: " + str(exc)
                    process.terminate()
                    await asyncio.to_thread(process.join, 5)
                    if process.is_alive():
                        process.kill()
                        await asyncio.to_thread(process.join, 5)
            report["verifier_exitcode"] = process.exitcode
            report["verifier_still_alive"] = process.is_alive()
            pipe.close()
            child_pipe.close()
            shutil.copytree(work, output / "work-evidence", ignore=shutil.ignore_patterns("key.pem", "*.sock"))
    report["seconds"] = round(time.monotonic() - began, 3)
    write(output / "report.json", report)
    try:
        audit = audit_saved(output)
    except Exception as exc:
        audit = {"passed": False, "failure": type(exc).__name__ + ": " + str(exc), "qualification_passed": False}
    write(output / "audit.json", audit)
    print(json.dumps({"report": str(output / "report.json"), **audit}), flush=True)
    return report, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, audit = asyncio.run(run(Path(__file__).resolve().parents[1], args.output.resolve()))
    if report["failure"] or not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
