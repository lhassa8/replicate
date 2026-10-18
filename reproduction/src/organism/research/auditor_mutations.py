"""Evaluate unchanged lineage auditors against six retained native births.

Mutations are applied to copies. Hash-repaired decision mutations distinguish
semantic checks from merely detecting a stale job digest. This is a finite
fault-detection experiment, not an adversarial security evaluation.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from organism.audit_application import canonical, sha
from organism.native_preflight import audit_lineage_transport as auditor
from organism.native_preflight.audit_startup import Evidence


ROOT = Path(__file__).resolve().parents[2]
TRIALS = ("08", "09", "10")
MUTATIONS = (
    ("decision", "wrong_model_incarnation"),
    ("decision", "changed_generation_budget"),
    ("decision", "missing_model_request"),
    ("decision", "parsed_choice_disagrees_with_stream"),
    ("decision", "unbacked_transport_receipt"),
    ("decision", "wrong_saved_submission"),
    ("decision", "no_model_calls"),
    ("transfer", "missing_payload"),
    ("transfer", "foreign_contributor"),
    ("transfer", "wrong_payload_hash"),
    ("transfer", "wrong_payload_length"),
    ("transfer", "missing_reservation"),
    ("transfer", "missing_commit"),
    ("link", "availability_after_proposal"),
    ("link", "wrong_child_individual"),
    ("chain", "deleted_peer_record"),
    ("chain", "modified_peer_record"),
    ("chain", "modified_history_head"),
)


def identity(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inputs(trial, slot):
    directory = ROOT / f"organism/evidence/first-lineage-trial-20261017-{trial}"
    parent_slot = "seed" if slot == "child" else "child"
    original = json.loads((directory / "independent-lineage-audit.json").read_text())
    parent_root, child_root = directory / "returned" / parent_slot, directory / "returned" / slot
    parent_full = json.loads((parent_root / "observe/observation.json").read_text())["ready"]
    child_full = json.loads((child_root / "observe/observation.json").read_text())["ready"]
    parent = {k: parent_full[k] for k in ("binding", "policy_id", "readiness_records")}
    child = {k: child_full[k] for k in ("binding", "policy_id", "birth")}
    catalog_full = json.loads((child_root / "input-catalog.json").read_text())
    catalog = {k: catalog_full[k] for k in ("files", "release_id", "platforms")}
    enabled = original["devices"][slot]["controls"][0]["observed_at"]
    parent_evidence, child_evidence = Evidence(parent_root), Evidence(child_root)
    history = auditor.chain(child_evidence, "seed-evidence/control/state_directory/")
    work = "seed-evidence/data/work_root/" + parent["binding"]["individual_id"] + "/"
    rows = parent_evidence.rows(work + "reproduction/application.sqlite3",
                                "SELECT digest,body FROM jobs")
    if len(rows) != 1:
        raise ValueError("Expected exactly one retained reproduction decision")
    job = {"digest": rows[0][0], "record": json.loads(rows[0][1])}
    return dict(parent=parent, child=child, catalog=catalog, enabled_at=enabled, history=history, job=job), parent_evidence, child_evidence


def mutate(material, category, name):
    changed = deepcopy(material)
    if category == "decision":
        record = changed["job"]["record"]
        trace = record["trace"]
        if name == "wrong_model_incarnation":
            trace["http_requests"][0]["body"]["model"] = "organism-" + "0" * 32
        elif name == "changed_generation_budget":
            trace["http_requests"][0]["body"]["max_tokens"] = 193
        elif name == "missing_model_request":
            trace["http_requests"].pop()
        elif name == "parsed_choice_disagrees_with_stream":
            for response in trace["model_responses"]:
                for block in response.get("message", {}).get("content", []):
                    if block.get("toolUse", {}).get("name") == "choose_child_action":
                        block["toolUse"]["input"]["action"] = "defer"
        elif name == "unbacked_transport_receipt":
            trace["tool_events"][0]["result"]["transport_receipt"]["digest"] = "0" * 64
        elif name == "wrong_saved_submission":
            record["submission"]["platform"] = "linux-arm64"
        elif name == "no_model_calls":
            for key in ("model_calls", "model_streams", "model_responses", "http_requests"):
                trace[key] = []
        changed["job"]["digest"] = sha(canonical(record))
    elif category == "transfer":
        history = changed["history"]
        operation = material["child"]["birth"]["operation_id"]
        selected = next(i for i, row in enumerate(history)
                        if row["record"]["operation_id"] == operation
                        and row["record"]["phase"] == "completed"
                        and row["record"]["operation"] == "put")
        target = history[selected]["record"]
        if name == "missing_payload":
            payload = target["detail"]["payload"]
            changed["history"] = [r for r in history if not (
                r["record"]["operation_id"] == operation and r["record"]["operation"] == "put"
                and r["record"]["detail"].get("payload") == payload)]
        elif name == "foreign_contributor":
            target["peer_id"] = "0" * 64
        elif name == "wrong_payload_hash":
            target["detail"]["sha256"] = "0" * 64
        elif name == "wrong_payload_length":
            target["detail"]["bytes"] += 1
        else:
            missing = "reserve" if name == "missing_reservation" else "commit"
            changed["history"] = [r for r in history if not (
                r["record"]["operation_id"] == operation and r["record"]["operation"] == missing)]
    elif category == "link":
        if name == "wrong_child_individual":
            changed["child"]["birth"]["individual_id"] = "0" * 32
        else:
            changed["enabled_at"] = (
                datetime.fromisoformat(material["enabled_at"]) + timedelta(days=1)).isoformat()
    return changed


def check(material, category, parent_root, child_root):
    if category == "decision":
        return auditor.decision(material["job"], parent=material["parent"], child=material["child"],
                                catalog=material["catalog"], child_history=material["history"])
    if category == "transfer":
        return auditor.transfer(material["history"], material["child"], material["catalog"],
                                material["parent"]["binding"]["device_id"])
    if category == "chain":
        return auditor.chain(Evidence(child_root), "seed-evidence/control/state_directory/")
    return auditor.link(Evidence(parent_root), material["parent"], Evidence(child_root),
                        material["child"], material["catalog"], child_enabled_at=material["enabled_at"])


def observed(function):
    try:
        value = function()
        return {"accepted": True, "rejection": None}
    except ValueError as exc:
        return {"accepted": False, "rejection": str(exc)}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    source_names = (
        "organism/native_preflight/audit_lineage_transport.py",
        "organism/native_preflight/audit_startup.py",
        "organism/audit_application.py",
        "organism/research/auditor_mutations.py",
    )
    sources = {}
    for name in source_names:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        sources[name] = identity(target)
    write(output / "protocol.json", {
        "schema": "replicate-native-record-mutations-v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "births": [f"{trial}-{slot}" for trial in TRIALS for slot in ("child", "grandchild")],
        "positive_checks_per_birth": ["decision", "transfer", "chain", "link"],
        "mutations_per_birth": [{"category": c, "name": n} for c, n in MUTATIONS],
        "source_files": sources, "decision_job_hashes_recomputed": True,
        "transfer_scope": "Semantic transfer function on previously chain-verified records; chain integrity evaluated separately",
        "expected": "Accept every unmodified control; reject every listed invalid variant",
        "scope": "Finite consistency violations on retained native evidence, not live fault injection or cryptographic forgery resistance",
    })
    results, provenance = [], []
    for trial in TRIALS:
        for slot in ("child", "grandchild"):
            name = f"{trial}-{slot}"
            directory = output / "inputs" / name
            directory.mkdir(parents=True)
            material, parent, child = inputs(trial, slot)
            # Run the complete positive link once to enumerate exactly the public
            # evidence files needed by its checks; copy only those accessed files.
            auditor.link(parent, material["parent"], child, material["child"], material["catalog"],
                         child_enabled_at=material["enabled_at"])
            identities = {}
            for role, evidence in (("parent", parent), ("child", child)):
                for relative, expected_sha in evidence.read_hashes.items():
                    target = directory / role / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(evidence.path(relative), target)
                    identities[role + "/" + relative] = identity(target)
                    if identities[role + "/" + relative]["sha256"] != expected_sha:
                        raise ValueError("Evidence changed during subset preservation")
            write(directory / "material.json", material)
            write(directory / "identities.json", identities)
            provenance.append({"birth": name, "material": identity(directory / "material.json"),
                               "identities": identity(directory / "identities.json")})
            for category in ("decision", "transfer", "chain", "link"):
                result = observed(lambda c=category: check(material, c, directory / "parent", directory / "child"))
                results.append({"birth": name, "category": category, "variant": "valid_control",
                                "expected_acceptance": True, **result})
            db_path = directory / "child/seed-evidence/control/state_directory/peer-audit.sqlite3"
            for category, variant in MUTATIONS:
                changed = mutate(material, category, variant)
                if category == "chain":
                    original = db_path.read_bytes()
                    try:
                        with sqlite3.connect(db_path) as db:
                            sql = {
                                "deleted_peer_record": "DELETE FROM records WHERE sequence=2",
                                "modified_peer_record": "UPDATE records SET body='{}' WHERE sequence=2",
                                "modified_history_head": "UPDATE head SET digest='" + "0" * 64 + "'",
                            }[variant]
                            db.execute(sql)
                        result = observed(lambda: check(changed, category, directory / "parent", directory / "child"))
                    finally:
                        db_path.write_bytes(original)
                else:
                    result = observed(lambda: check(changed, category, directory / "parent", directory / "child"))
                results.append({"birth": name, "category": category, "variant": variant,
                                "expected_acceptance": False, "outer_job_hash_recomputed": category == "decision",
                                **result})
    for entry in provenance:
        directory = output / "inputs" / entry["birth"]
        for relative, expected in json.loads((directory / "identities.json").read_text()).items():
            if identity(directory / relative) != expected:
                raise ValueError("Original evidence subset was not restored")
    controls = [r for r in results if r["expected_acceptance"]]
    invalid = [r for r in results if not r["expected_acceptance"]]
    report = {
        "schema": "replicate-native-record-mutation-results-v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol": identity(output / "protocol.json"), "input_sets": provenance,
        "native_births": 6, "valid_controls": len(controls),
        "valid_controls_accepted": sum(r["accepted"] for r in controls),
        "invalid_variants": len(invalid), "invalid_variants_rejected": sum(not r["accepted"] for r in invalid),
        "false_rejections": sum(not r["accepted"] for r in controls),
        "false_acceptances": sum(r["accepted"] for r in invalid),
        "results": results, "native_fault_injection": False, "qualification_passed": False,
    }
    write(output / "results.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("results", "input_sets")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())
