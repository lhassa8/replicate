"""Local procedure handoff using public fixtures and ephemeral lab identities.

This exercises three persistent stores in one process on the Mac, not peer
network transport, independent native hosts, model reasoning, or Q7.
"""

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .audit_equipment_diagnostics import inspect_job
from .capsule import canonical
from .enrollment import Enrollment, key_id, public_bytes, sign_enrollment
from .equipment_baseline import solve
from .equipment_development import case
from .equipment_interfaces import InterfaceDirectory, InterfaceSession
from .experience import require, VerifiedExperience
from .experience_issue import checker_identity, CHECKER_SOURCES, issue
from .experience_store import ExperienceStore, verify_copy_receipt


NOW = 1700000000
RELEASE = hashlib.sha256(b"Public experience component fixture, not a released organism").hexdigest()
SOURCES = tuple(sorted(set(CHECKER_SOURCES) | {
    "probe_experience.py", "experience_store.py", "equipment_interfaces.py",
    "equipment_development.py", "equipment_baseline.py", "equipment.py"}))


def lab():
    owner = Ed25519PrivateKey.generate()
    keys = [Ed25519PrivateKey.generate() for _ in range(4)]
    ids = [key_id(public_bytes(key.public_key())) for key in keys]
    release_key = Ed25519PrivateKey.generate()
    release_public = public_bytes(release_key.public_key())
    body = {"schema": "organism-enrollment-v1", "experiment_id": "public-experience-component-check",
            "epoch": 1, "devices": {identity: {
                "public_key": base64.b64encode(public_bytes(key.public_key())).decode(),
                "platform": "linux-arm64" if index % 2 else "linux-x86_64"}
                for index, (identity, key) in enumerate(zip(ids, keys))},
            "release_keys": {key_id(release_public): base64.b64encode(release_public).decode()},
            "require_trained": True, "max_payload_bytes": 1024**2, "birth_seconds": 60,
            "max_births_per_device": 8, "max_process_starts": 3}
    owner_public = public_bytes(owner.public_key())
    signed = sign_enrollment(body, owner)
    return Enrollment(signed, owner_public), owner_public, keys, ids


def job(session):
    body = {"device_id": session.device_id, "task_kind": "equipment_interface_v1",
            "status": "finished", "submission": deepcopy(session.submission),
            "trace": {"tool_events": deepcopy(session.events)}}
    return {"record": body, "digest": hashlib.sha256(canonical(body)).hexdigest()}


def source_episode(root, profile):
    item = case(profile, "fault", NOW)
    raw = canonical(item["document"])
    (root / (item["case_id"] + ".json")).write_bytes(raw)
    session = InterfaceSession(InterfaceDirectory(root, clock=lambda: NOW), item["case_id"])
    solve(session, item["case_id"])
    plan = {"schema": "organism-diagnostic-oracle-v1",
            **{key: value for key, value in item["expected_reader"].items() if key != "device_id"},
            "operation": {"kind": "scalar"}}
    return item, raw, job(session), plan


def endorsed(root, profile, policy, keys):
    item, raw, source_job, plan = source_episode(root, profile)
    signed, validation = issue(raw, source_job, plan, policy=policy, producer_key=keys[0], verifier_key=keys[1],
        producer_individual_id="a" * 32, release_id=RELEASE, created=NOW, expires=NOW + 86400,
        source_clock=NOW, live=False)
    return signed, {"fixture": item, "job": source_job, "reviewed_source_plan": plan, "validation": validation}


def current_task(root, profile, record):
    item = case(profile, "nominal", NOW)
    # Prove that instance identity and the observation need not be inherited.
    item["case_id"] = item["document"]["device_id"] = "new-device-" + profile
    item["expected_reader"]["device_id"] = item["case_id"]
    raw = canonical(item["document"])
    path = root / (item["case_id"] + ".json")
    path.write_bytes(raw)
    session = InterfaceSession(InterfaceDirectory(root, clock=lambda: NOW), item["case_id"])
    prepared = record.prepare_reader(session)
    require(session.measurement is None and session.submission is None,
            "Inherited procedure improperly supplied a measurement or diagnosis")
    reading = session.read_equipment(item["case_id"])
    # The fixture driver supplies this answer, not a model or the experience.
    session.submit_diagnosis(item["case_id"], item["expected_status"], item["expected_value"],
                             item["expected_unit"], reading["evidence_id"])
    saved_job = job(session)
    checked = inspect_job(item, raw, saved_job)
    require(checked["semantic_passed"] is True and path.read_bytes() == raw, "Current diagnostic fixture failed")
    return {"fixture": item, "job": saved_job, "preparation": prepared, "trace_check": checked,
            "harness_supplied_current_diagnosis": True}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def receipt(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def audit_saved(output):
    output = Path(output).absolute()
    report = json.loads((output / "report.json").read_bytes())
    context = json.loads((output / "lab.json").read_bytes())
    policy = Enrollment(base64.b64decode(context["signed_enrollment"]), base64.b64decode(context["owner_public"]))
    require(report["schema"] == "organism-experience-development-check-v1" and report["failure"] is None
            and report["model_calls"] == 0 and report["network_transport_exercised"] is False
            and report["qualification_passed"] is False, "Experience probe scope differs")
    require(receipt(output / "lab.json") == report["lab_receipt"], "Experience lab context changed")
    require(set(report["source_receipts"]) == set(SOURCES), "Experience probe source inventory differs")
    for name, expected in report["source_receipts"].items():
        require(receipt(output / "sources" / name) == expected == receipt(Path(__file__).with_name(name)),
                "Experience probe source changed")
    checker, checker_sources = checker_identity()
    require(checker == context["checker_sha256"] and checker_sources == context["checker_sources"],
            "Endorsing checker identity differs")
    require(len(report["cases"]) == 8 and {row["profile"] for row in report["cases"]} == set("abcdefgh"),
            "Experience probe profile inventory differs")
    identities = context["device_ids"]
    for row in report["cases"]:
        path = output / "cases" / (row["profile"] + ".json")
        require(receipt(path) == row["receipt"], "Experience handoff record changed")
        saved = json.loads(path.read_bytes())
        raw = base64.b64decode(saved["signed_experience"])
        record = VerifiedExperience(raw, policy, release_id=RELEASE, allowed_checkers={checker},
                                    now=NOW, require_live_source=False)
        source = saved["source"]
        fixture = source["fixture"]
        checked = inspect_job(fixture, canonical(fixture["document"]), source["job"])
        require(checked["semantic_passed"] is True and record.body["source"]["job_sha256"] == source["job"]["digest"]
                and record.body["source"]["observation_sha256"] == hashlib.sha256(canonical(fixture["document"])).hexdigest()
                and record.body["source"]["validation_sha256"] == hashlib.sha256(canonical(source["validation"])).hexdigest(),
                "Experience does not bind its checked source task")
        from .diagnostic_oracle import reconstruct
        reconstruction = reconstruct(canonical(fixture["document"]), source["reviewed_source_plan"], now=NOW)
        require(reconstruction == source["validation"]["reconstruction"]
                and checked == source["validation"]["trace_check"], "Source verification changed")
        for label, sender, receiver, request in (
                ("child_receipt", identities[0], identities[2], row["profile"].encode().hex().ljust(32, "0")),
                ("grandchild_receipt", identities[2], identities[3], row["profile"].encode().hex().ljust(32, "1"))):
            verify_copy_receipt(base64.b64decode(saved[label]), policy, release_id=RELEASE, note_id=record.note_id,
                                sender_id=sender, receiver_id=receiver, request_id=request, now=NOW)
        target = saved["current_task"]
        target_fixture = target["fixture"]
        final_check = inspect_job(target_fixture, canonical(target_fixture["document"]), target["job"])
        require(final_check == target["trace_check"] and final_check["semantic_passed"] is True,
                "Destination task readback failed")
        cap = target["job"]["record"]["trace"]["tool_events"][0]["result"]
        require(record.applicable(cap["descriptor"], now=NOW)
                and target["job"]["record"]["trace"]["tool_events"][1]["arguments"] == {
                    "device_id": target_fixture["case_id"], **record.body["reader"]},
                "Inherited reader differs from its endorsed procedure")
        require(saved["parent_store_unavailable"] is True and saved["child_reopened"] is True
                and saved["child_acknowledgment_retry_identical"] is True
                and saved["grandchild_retained_identical_record"] is True,
                "Experience handoff observations are incomplete")
        for label, identity, expected_counts in (
                ("parent", identities[0], (1, 0, 1)), ("child", identities[2], (1, 1, 1)),
                ("grandchild", identities[3], (1, 1, 0))):
            store = output / "stores" / (row["profile"] + "-" + label)
            require(set(saved["store_snapshots"][label]) == {"experience.sqlite3", "experience.enrolled"},
                    "Store snapshot inventory differs")
            for filename, expected in saved["store_snapshots"][label].items():
                require(receipt(store / filename) == expected, "Retained experience store changed")
            marker = json.loads((store / "experience.enrolled").read_bytes())
            require(marker["device_id"] == identity and marker["policy_id"] == policy.policy_id
                    and marker["release_id"] == RELEASE, "Retained store belongs to another context")
            database = sqlite3.connect((store / "experience.sqlite3").as_uri() + "?mode=ro", uri=True)
            try:
                counts = tuple(database.execute("SELECT count(*) FROM " + table).fetchone()[0]
                               for table in ("notes", "issued", "confirmed"))
                require(counts == expected_counts
                        and database.execute("SELECT notes,issued,confirmed FROM head").fetchall() == [expected_counts],
                        "Retained store does not contain the committed copy history")
                require(database.execute("SELECT note_id,raw FROM notes").fetchall() == [(record.note_id, raw)],
                        "Retained store lacks the exact endorsed record")
                if label != "parent":
                    ack_name = "child_receipt" if label == "child" else "grandchild_receipt"
                    ack_raw = base64.b64decode(saved[ack_name])
                    require(database.execute("SELECT raw FROM issued").fetchall() == [(ack_raw,)]
                            and json.loads(ack_raw)["body"]["store_id"] == marker["store_id"],
                            "Copy acknowledgment does not bind the retained destination commit")
                if label != "grandchild":
                    ack_name = "child_receipt" if label == "parent" else "grandchild_receipt"
                    require(database.execute("SELECT raw FROM confirmed").fetchall()
                            == [(base64.b64decode(saved[ack_name]),)],
                            "Source confirmation differs from the destination acknowledgment")
            finally:
                database.close()
    return {"audit_passed": True, "public_development_profiles": 8, "model_calls": 0,
            "scope": "Saved signatures, source/current task evidence and local handoff receipts",
            "network_transport_exercised": False, "independent_physical_hosts_verified": False,
            "q7_effort_comparison_executed": False, "qualification_passed": False}


def run(output):
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("cases", "sources", "stores"):
        (output / name).mkdir()
    for name in SOURCES:
        (output / "sources" / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    policy, owner, keys, identities = lab()
    checker, checker_sources = checker_identity()
    write(output / "lab.json", {"signed_enrollment": base64.b64encode(policy.signed).decode(),
          "owner_public": base64.b64encode(owner).decode(), "device_ids": identities,
          "checker_sha256": checker, "checker_sources": checker_sources, "release_id": RELEASE,
          "private_keys_saved": False, "native_platform_labels_are_fixture_metadata": True})
    report = {"schema": "organism-experience-development-check-v1",
              "started_at": datetime.now(timezone.utc).isoformat(), "cases": [], "failure": None,
              "source_receipts": {name: receipt(output / "sources" / name) for name in SOURCES},
              "lab_receipt": receipt(output / "lab.json"), "model_calls": 0,
              "network_transport_exercised": False, "independent_physical_hosts_verified": False,
              "q7_effort_comparison_executed": False, "qualification_passed": False}
    write(output / "request.json", report)
    began = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="experience-development-") as temporary:
            root = Path(temporary)
            observations = root / "observations"
            observations.mkdir()
            for profile in "abcdefgh":
                raw, source = endorsed(observations, profile, policy, keys)
                kwargs = {"release_id": RELEASE, "allowed_checkers": {checker},
                          "require_live_source": False, "clock": lambda: NOW}
                parent = ExperienceStore(root / (profile + "-parent"), policy, keys[0], **kwargs)
                child = ExperienceStore(root / (profile + "-child"), policy, keys[2], **kwargs)
                grandchild = ExperienceStore(root / (profile + "-grandchild"), policy, keys[3], **kwargs)
                note_id = parent.add(raw)
                first_request = profile.encode().hex().ljust(32, "0")
                ack = child.receive(parent.get(note_id).raw, sender_id=identities[0], request_id=first_request)
                parent_copies = parent.confirm_copy(note_id, receiver_id=identities[2], request_id=first_request, receipt=ack)
                # Lose the first acknowledgment at the caller and reopen the
                # persistent destination before retrying the same transfer.
                child = ExperienceStore(root / (profile + "-child"), policy, keys[2], **kwargs)
                retry = child.receive(raw, sender_id=identities[0], request_id=first_request)
                require(ack == retry, "Lost acknowledgment retry changed its committed receipt")
                old_parent = parent.directory
                unavailable_parent = root / (profile + "-parent-unavailable")
                old_parent.rename(unavailable_parent)
                require(not old_parent.exists(), "Original parent store is still available at its path")
                next_request = profile.encode().hex().ljust(32, "1")
                inherited = child.get(note_id)
                next_ack = grandchild.receive(inherited.raw, sender_id=identities[2], request_id=next_request)
                child_copies = child.confirm_copy(note_id, receiver_id=identities[3], request_id=next_request, receipt=next_ack)
                descendant = grandchild.get(note_id)
                require(descendant.raw == inherited.raw == raw, "Forwarded experience changed")
                current = current_task(observations, profile, descendant)
                snapshots = {}
                for label, source_path in (("parent", unavailable_parent), ("child", child.directory),
                                           ("grandchild", grandchild.directory)):
                    destination = output / "stores" / (profile + "-" + label)
                    destination.mkdir(mode=0o700)
                    snapshots[label] = {}
                    for filename in ("experience.sqlite3", "experience.enrolled"):
                        target = destination / filename
                        target.write_bytes((source_path / filename).read_bytes())
                        target.chmod(0o600)
                        snapshots[label][filename] = receipt(target)
                stored = {"profile": profile, "source": source, "signed_experience": base64.b64encode(raw).decode(),
                          "child_receipt": base64.b64encode(ack).decode(),
                          "grandchild_receipt": base64.b64encode(next_ack).decode(),
                          "parent_copy_state": parent_copies, "child_copy_state": child_copies,
                          "parent_store_unavailable": True, "child_reopened": True,
                          "child_acknowledgment_retry_identical": True, "grandchild_retained_identical_record": True,
                          "current_task": current, "store_snapshots": snapshots}
                path = output / "cases" / (profile + ".json")
                write(path, stored)
                report["cases"].append({"profile": profile, "receipt": receipt(path), "experience_bytes": len(raw)})
                write(output / "progress.json", report)
        require(all(receipt(Path(__file__).with_name(name)) == expected
                    for name, expected in report["source_receipts"].items()), "Experience probe source changed")
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    report["elapsed_seconds"] = round(time.monotonic() - began, 3)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    write(output / "report.json", report)
    audit = audit_saved(output) if report["failure"] is None else {
        "audit_passed": False, "failure": report["failure"], "qualification_passed": False}
    write(output / "audit.json", audit)
    return report, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, audit = run(args.output)
    print(json.dumps({"profiles": len(report["cases"]), "failure": report["failure"],
                      "elapsed_seconds": report["elapsed_seconds"], "audit_passed": audit["audit_passed"],
                      "qualification_passed": False}))
    if report["failure"] or not audit["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
