"""Separate-process verifier and public-fixture record handoff on this Mac.

No model, native Linux device, peer network or active endorsement service runs.
The fixture harness creates the lab keys; only the verifier's private key is
passed to the spawned verifier. This is process separation, not host or OS
security isolation, and is not Q7 qualification.
"""

import argparse
import base64
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .capsule import canonical
from .enrollment import Enrollment, key_id, public_bytes
from .experience import VerifiedExperience, capability_hash, exact, require
from .experience_endorsement import CHECKER_SOURCES, checker_identity, endorse, review_source
from .experience_proposal import accept, check_approval, check_proposal, propose, sha
from .experience_store import ExperienceStore, verify_copy_receipt
from .probe_experience import NOW, RELEASE, current_task, lab, source_episode
from .equipment_development import case


PROFILES = tuple("abcdefgh")
SOURCES = tuple(sorted(set(CHECKER_SOURCES) | {
    "probe_separate_endorsement.py", "probe_experience.py", "experience_store.py",
    "equipment_development.py", "equipment_baseline.py", "equipment_interfaces.py", "equipment.py"}))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fixture_reviewed_plans():
    """Independent fixture provisioning, before any producer source is executed."""
    plans = {}
    for profile in PROFILES:
        item = case(profile, "fault", NOW)
        document = item["document"]
        published = {key: document[key] for key in ("device_id", "family", "task")}
        published["sources"] = [{key: value for key, value in source.items() if key != "payload"}
                                for source in document["sources"]]
        plans[capability_hash(published)] = {
            "schema": "organism-diagnostic-oracle-v1",
            **{key: value for key, value in item["expected_reader"].items() if key != "device_id"},
            "operation": {"kind": "scalar"}}
    return plans


def endorse_request(request, policy, key, reviewed_plans):
    exact(request, ("proposal", "raw", "job"))
    source = check_proposal(request["proposal"], policy, release_id=RELEASE)
    require(source["capability_sha256"] in reviewed_plans, "No independently provisioned source procedure")
    return endorse(request["proposal"], request["raw"], request["job"],
        reviewed_plans[source["capability_sha256"]], policy=policy, verifier_key=key, release_id=RELEASE,
        created=NOW, expires=NOW + 3600, require_live_source=False)


def verifier_worker(receive, send, enrollment, owner_public, verifier_secret, reviewed_plans):
    """Local owned test worker; one private-key argument and at most eight requests."""
    policy = Enrollment(enrollment, owner_public)
    key = Ed25519PrivateKey.from_private_bytes(verifier_secret)
    identity = key_id(public_bytes(key.public_key()))
    del verifier_secret
    try:
        for _ in PROFILES:
            request = receive.recv()
            try:
                approval, receipt = endorse_request(request, policy, key, reviewed_plans)
                response = {"passed": True, "approval": approval, "receipt": receipt,
                            "worker_pid": os.getpid(), "verifier_device_id": identity}
            except Exception as exc:
                response = {"passed": False, "failure": type(exc).__name__ + ": " + str(exc),
                            "worker_pid": os.getpid(), "verifier_device_id": identity}
            send.send(response)
    finally:
        receive.close()
        send.close()


def audit_saved(output):
    output = Path(output)
    report = json.loads((output / "report.json").read_bytes())
    context_raw = (output / "lab.json").read_bytes()
    require(sha(context_raw) == report["lab_sha256"], "Lab context changed")
    context = json.loads(context_raw)
    plans_raw = (output / "reviewed-plans.json").read_bytes()
    require(sha(plans_raw) == context["reviewed_plans_sha256"]
            and json.loads(plans_raw) == fixture_reviewed_plans(), "Provisioned fixture procedures differ")
    plans = json.loads(plans_raw)
    policy = Enrollment(base64.b64decode(context["enrollment"]), base64.b64decode(context["owner_public"]))
    checker, sources = checker_identity()
    require(context["checker_sha256"] == checker and context["checker_sources"] == sources,
            "Independent checker changed")
    require(set(report["source_hashes"]) == set(SOURCES), "Probe source inventory differs")
    for name, expected in report["source_hashes"].items():
        require(sha((output / "sources" / name).read_bytes()) == expected
                == sha(Path(__file__).with_name(name).read_bytes()), "Probe source changed")
    require(report["failure"] is None and report["profiles"] == list(PROFILES)
            and report["verifier_exitcode"] == 0 and report["model_calls"] == 0,
            "Separate-process check is incomplete")
    require(report["verifier_pid"] != report["producer_pid"], "Verifier did not use a separate process")
    count, records, receipts = 0, {}, {"child": {}, "grandchild": {}}
    for profile in PROFILES:
        stored = json.loads((output / ("source-" + profile + ".json")).read_bytes())
        raw = canonical(stored["case"]["document"])
        proposal, approval, signed = (base64.b64decode(stored[name]) for name in ("proposal", "approval", "record"))
        body, signature = check_approval(
            proposal, approval, policy=policy, release_id=RELEASE,
            allowed_checkers={checker}, now=NOW, require_live_source=False)
        reconstructed, receipt = review_source(
            proposal, raw, stored["job"], stored["reviewed_source_plan"], policy=policy,
            verifier_device_id=context["device_ids"][1], release_id=RELEASE,
            created=NOW, expires=NOW + 3600, require_live_source=False)
        require(body == reconstructed and receipt == stored["validation"], "Saved source verification differs")
        record = VerifiedExperience(signed, policy, release_id=RELEASE, allowed_checkers={checker},
                                    now=NOW, require_live_source=False)
        require(record.body == body and json.loads(signed)["verifier_signature"] == signature,
                "Final record differs from its verified approval")
        require(stored["worker_pid"] == report["verifier_pid"], "Approval has no matching verifier execution")
        require(stored["reviewed_source_plan"] == plans[record.body["capability_sha256"]],
                "Source verification used a request-supplied procedure")
        records[record.note_id] = record.raw
        for offset, label in ((2, "child"), (3, "grandchild")):
            copied = json.loads((output / (label + "-" + profile + ".json")).read_bytes())
            verified = verify_copy_receipt(base64.b64decode(copied["receipt"]), policy,
                receiver_id=context["device_ids"][offset], sender_id=context["device_ids"][0 if offset == 2 else 2],
                release_id=RELEASE, note_id=record.note_id, request_id=copied["request_id"], now=NOW)
            require(verified["note_id"] == record.note_id, "Copied record identity changed")
            receipts[label][record.note_id] = base64.b64decode(copied["receipt"])
        target = json.loads((output / ("target-" + profile + ".json")).read_bytes())
        from .audit_equipment_diagnostics import inspect_job
        require(inspect_job(target["fixture"], canonical(target["fixture"]["document"]),
                            target["job"])["semantic_passed"], "Fresh target diagnosis failed readback")
        require(target["preparation"]["note_id"] == record.note_id
                and not target["preparation"]["diagnosis_supplied"], "Target did not use the retained procedure")
        count += 1
    # Check the actual retained stores, not only their signed copy receipts.
    for label, expected_head in (("retained-parent-evidence", (8, 0, 8)),
                                 ("child", (8, 8, 8)), ("grandchild", (8, 8, 0))):
        directory = output / "work-evidence" / label
        database = sqlite3.connect((directory / "experience.sqlite3").as_uri() + "?mode=ro", uri=True)
        try:
            database.execute("PRAGMA query_only=ON")
            stored = {note: bytes(raw) for note, raw in database.execute("SELECT note_id,raw FROM notes")}
            head = database.execute("SELECT notes,issued,confirmed FROM head WHERE singleton=1").fetchone()
            actual = tuple(database.execute("SELECT count(*) FROM " + table).fetchone()[0]
                           for table in ("notes", "issued", "confirmed"))
            require(stored == records and head == actual == expected_head, "Retained store differs from its history")
            for table, kind in (("issued", label), ("confirmed", "child" if label == "retained-parent-evidence"
                                                   else "grandchild" if label == "child" else None)):
                rows = {note: bytes(raw) for note, raw in database.execute("SELECT note_id,raw FROM " + table)}
                require(rows == receipts.get(kind, {}), "Retained copy acknowledgments differ")
        finally:
            database.close()
    require(not (output / "work-evidence/parent").exists(), "Original parent store remains on its active path")
    return {"passed": True, "verified_sources": count, "signed_copy_receipts": count * 2,
            "fresh_fixture_diagnoses": count, "public_key_only_readback": True,
            "retained_stores_checked": 3, "stored_note_copies_checked": count * 3,
            "verifier_procedures_provisioned_before_source_requests": True,
            "source_and_target_model_calls": 0, "peer_network_exercised": False,
            "independent_physical_hosts_verified": False, "qualification_passed": False}


def run(output):
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    policy, owner, keys, identities = lab()
    checker, checker_sources = checker_identity()
    plans = fixture_reviewed_plans()
    write(output / "reviewed-plans.json", plans)
    context = {"enrollment": base64.b64encode(policy.signed).decode(),
               "owner_public": base64.b64encode(owner).decode(), "device_ids": identities,
               "release_id": RELEASE, "checker_sha256": checker, "checker_sources": checker_sources,
               "reviewed_plans_sha256": sha((output / "reviewed-plans.json").read_bytes()),
               "procedures_are_public_fixtures_not_human_admitted": True}
    write(output / "lab.json", context)
    report = {"schema": "organism-separate-endorsement-probe-v1",
              "started_at": datetime.now(timezone.utc).isoformat(), "profiles": list(PROFILES),
              "scope": "Eight public fixture sources, separate verifier process and retained-procedure handoff",
              "producer_pid": os.getpid(), "verifier_pid": None, "verifier_exitcode": None,
              "lab_sha256": sha((output / "lab.json").read_bytes()), "source_hashes": {},
              "cases": [], "model_calls": 0, "private_keys_retained": False,
              "peer_network_exercised": False, "active_application_service_exercised": False,
              "independent_physical_hosts_verified": False, "os_security_isolation_verified": False,
              "native_execution_verified": False, "qualification_passed": False, "failure": None}
    report["verifier_procedures_provisioned_before_source_requests"] = True
    report["source_request_fields"] = ["proposal", "raw", "job"]
    (output / "sources").mkdir()
    for name in SOURCES:
        raw = Path(__file__).with_name(name).read_bytes()
        (output / "sources" / name).write_bytes(raw)
        report["source_hashes"][name] = sha(raw)
    write(output / "request.json", report)
    ctx = multiprocessing.get_context("spawn")
    receive_requests, send_requests = ctx.Pipe(duplex=False)
    receive_replies, send_replies = ctx.Pipe(duplex=False)
    worker = ctx.Process(target=verifier_worker, args=(
        receive_requests, send_replies, policy.signed, owner, keys[1].private_bytes_raw(), plans))
    worker.start()
    receive_requests.close()
    send_replies.close()
    report["verifier_pid"] = worker.pid
    # The fixture harness generated the lab keys. It drops its verifier key
    # reference; neither producer API accepts that key or the private plan.
    keys[1] = None
    try:
        with tempfile.TemporaryDirectory(prefix="organism-endorsement-", dir="/private/tmp") as temporary:
            work = Path(temporary)
            observations = work / "observations"
            observations.mkdir()
            options = {"release_id": RELEASE, "allowed_checkers": {checker},
                       "require_live_source": False, "clock": lambda: NOW}
            parent = ExperienceStore(work / "parent", policy, keys[0], **options)
            child = ExperienceStore(work / "child", policy, keys[2], **options)
            grandchild = ExperienceStore(work / "grandchild", policy, keys[3], **options)
            records = {}
            for index, profile in enumerate(PROFILES):
                item, raw, job, plan = source_episode(observations, profile)
                proposal = propose(raw, job, policy=policy, producer_key=keys[0],
                    producer_individual_id="a" * 32, verifier_device_id=identities[1], release_id=RELEASE,
                    request_id=f"{index + 1:032x}", source_clock=NOW, live=False)
                send_requests.send({"proposal": proposal, "raw": raw, "job": job})
                require(receive_replies.poll(15), "Owned verifier did not return within its allowance")
                reply = receive_replies.recv()
                require(reply["passed"], reply.get("failure", "Verifier rejected the source"))
                require(reply["worker_pid"] == worker.pid and reply["verifier_device_id"] == identities[1],
                        "Verifier response came from a different worker")
                record = accept(proposal, reply["approval"], policy=policy, producer_key=keys[0],
                    release_id=RELEASE, allowed_checkers={checker}, now=NOW, require_live_source=False)
                note_id = parent.add(record)
                records[profile] = note_id
                write(output / ("source-" + profile + ".json"), {
                    "case": item, "job": job, "reviewed_source_plan": plan, "validation": reply["receipt"],
                    "proposal": base64.b64encode(proposal).decode(), "approval": base64.b64encode(reply["approval"]).decode(),
                    "record": base64.b64encode(record).decode(), "worker_pid": reply["worker_pid"]})
                request = f"{index + 101:032x}"
                receipt = child.receive(record, sender_id=identities[0], request_id=request)
                parent.confirm_copy(note_id, receiver_id=identities[2], request_id=request, receipt=receipt)
                write(output / ("child-" + profile + ".json"),
                      {"request_id": request, "receipt": base64.b64encode(receipt).decode()})
                report["cases"].append({"profile": profile, "note_id": note_id, "source_verified": True,
                                        "worker_pid": reply["worker_pid"], "parent_loss_reuse_passed": False})
                write(output / "progress.json", report)
            worker.join(10)
            require(not worker.is_alive() and worker.exitcode == 0, "Verifier did not exit successfully")
            # Move only this newly created fixture store, retaining its files
            # as evidence. Later reads come from the child's own store.
            parent_path = parent.directory
            parent_path.rename(work / "retained-parent-evidence")
            require(not parent_path.exists(), "Original parent store is still present")
            del parent
            child = ExperienceStore(work / "child", policy, keys[2], **options)
            for index, profile in enumerate(PROFILES):
                note_id = records[profile]
                record = child.get(note_id)
                request = f"{index + 201:032x}"
                receipt = grandchild.receive(record.raw, sender_id=identities[2], request_id=request)
                child.confirm_copy(note_id, receiver_id=identities[3], request_id=request, receipt=receipt)
                write(output / ("grandchild-" + profile + ".json"),
                      {"request_id": request, "receipt": base64.b64encode(receipt).decode()})
                target = current_task(observations, profile, grandchild.get(note_id))
                write(output / ("target-" + profile + ".json"), target)
                report["cases"][index]["parent_loss_reuse_passed"] = True
            shutil.copytree(work, output / "work-evidence")
            report["parent_store_removed_from_active_path"] = True
            report["child_store_reopened_before_forwarding"] = True
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        send_requests.close()
        receive_replies.close()
        worker.join(1)
        if worker.is_alive():
            worker.terminate()
            worker.join(5)
        if worker.is_alive():
            worker.kill()
            worker.join(5)
        report["verifier_exitcode"] = worker.exitcode
        report["owned_verifier_terminal"] = not worker.is_alive() and worker.exitcode is not None
        if not report["owned_verifier_terminal"]:
            report["failure"] = report["failure"] or "Owned verifier exit remains uncertain"
        report["elapsed_seconds"] = round(time.monotonic() - began, 3)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["sources_unchanged"] = all(sha(Path(__file__).with_name(name).read_bytes()) == expected
                                          for name, expected in report["source_hashes"].items())
        if not report["sources_unchanged"]:
            report["failure"] = report["failure"] or "Probe source changed during the run"
        write(output / "report.json", report)
    try:
        audit = audit_saved(output)
    except Exception as exc:
        audit = {"passed": False, "failure": type(exc).__name__ + ": " + str(exc), "qualification_passed": False}
    write(output / "audit.json", audit)
    print(json.dumps({"failure": report["failure"], "verifier_exitcode": worker.exitcode, "audit": audit}))
    return report, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, audit = run(args.output)
    return 0 if report["failure"] is None and audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
