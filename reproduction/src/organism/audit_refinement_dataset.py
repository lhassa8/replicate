"""Independent checks for one versioned supplement to the unchanged v4 data.

No fixture generator or reference policy is imported during record inspection.
The runtime audit separately replays all old and new references through tools.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path

from .audit_equipment_diagnostics import inspect_job, resolve
from .audit_runtime_dataset import canonical, digest, require
from .runtime_rows import inspect_record as inspect_original, targets as original_targets


SCHEMA = "organism-refinement-runtime-trajectory-v1"
CORPUS_SCHEMA = "organism-refinement-runtime-corpus-v1"
PARENT_SHA = "0717976d7399d64c40e83e426b3b1afea774bea61c40b982b0ba4b9447750690"
REFERENCE_SHA = "4e29e882ee2f55bdf6f20d3d101acc710b8be2257bfea1c050c751e76e524b69"
FAMILIES = dict(zip("abcdef", ("gateway-json-v1", "gateway-register-v2", "condition-json-v1",
                             "condition-register-v3", "inspection-status-v1", "event-counter-v2")))
VARIANTS = ("low-value", "lower-boundary", "middle", "upper-boundary", "high-value",
            "scaled-middle", "tighter-high-range", "tighter-low-range", "invalid", "stale",
            "future", "missing", "numeric-string", "boolean")
PROVENANCE = {
    "kind": "locally_authored_runtime_reference_from_pinned_refinement_probe",
    "teacher_model_used": False, "human_review_claimed": False, "real_peer_transport_used": False,
    "native_process_used": False, "training_admitted": False, "qualification_data_present": False,
}


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def inspect_probe(reference, contract):
    case, execution = reference["case"], reference["execution"]
    fixture, identity = case["fixture"], case["case_id"]
    profile = case["profile_id"].removeprefix("environment-")
    require(profile in FAMILIES and case["family_id"] == FAMILIES[profile],
            "Supplement crossed a training family")
    require(case["variation"] in VARIANTS and type(case["seed"]) is int and case["seed"] == 0
            and case["clock"] == 1700100000 and case["role"] == "equipment"
            and case["split"] == "train_preparation" and case["inject_configuration_error"] is False,
            "Supplement condition changed")
    expected_id = "refine-" + digest(canonical(
        ["training-refinement-probe-v1", profile, case["variation"]]))[:32]
    require(identity == expected_id and fixture["split"] == "development_train"
            and fixture["profile"] == case["profile_id"] and fixture["variation"] == case["variation"],
            "Supplement identity or development split differs")
    document, reader = fixture["document"], fixture["expected_reader"]
    require(document["device_id"] == fixture["case_id"] == reader["device_id"] == identity
            and document["family"] == FAMILIES[profile] and len(document["sources"]) == 1
            and len(reader["value_selectors"]) == 1, "Supplement equipment identity differs")
    source = document["sources"][0]
    field = next(field for field in source["fields"] if field["selector"] == reader["value_selectors"][0])
    require(field["unit"] in ("C", "F", "mm/s", "in/s", "ratio", "count"),
            "A nontraining measurement unit entered the supplement")
    raw, timestamp = resolve(source, field["selector"]), resolve(source, reader["timestamp_selector"])
    valid = resolve(source, reader["validity_selector"])
    value, status = None, "unknown"
    if (numeric(raw) and numeric(timestamp) and valid is True and 0 <= timestamp <= case["clock"]
            and case["clock"] - timestamp <= document["task"]["maximum_age_seconds"]):
        value = Fraction(str(raw)) * Fraction(str(field["scale"]))
        if field["unit"] == "F":
            value = (value - 32) * Fraction(5, 9)
        elif field["unit"] == "in/s":
            value *= Fraction(127, 5)
        elif field["unit"] == "ratio":
            value *= 100
        low, high = (Fraction(str(item)) for item in document["task"]["normal_range"])
        require(low <= high, "Declared normal range is reversed")
        status = "low" if value < low else "high" if value > high else "normal"
        value = float(value)
    truth = {"status": status, "value": value, "unit": document["task"]["canonical_unit"]}
    require(truth == reference["decimal_oracle"]
            and (fixture["expected_status"], fixture["expected_value"], fixture["expected_unit"])
            == (status, value, truth["unit"]), "Independent rational interpretation differs")
    events = execution["events"]
    require([event["tool"] for event in events] ==
            ["inspect_equipment", "configure_reader", "read_equipment", "submit_diagnosis"],
            "Incomplete or altered reference sequence")
    require(execution["actions"] == [{"tool": e["tool"], "arguments": e["arguments"]} for e in events],
            "Executed actions differ")
    user = "Diagnose the equipment " + identity + "."
    strands = [{"role": "user", "content": [{"text": user}]}]
    wire = [{"role": "system", "content": contract["system_prompt"]},
            {"role": "user", "content": [{"text": user, "type": "text"}]}]
    native = [{"role": "system", "content": contract["system_prompt"]}, {"role": "user", "content": user}]
    for index, event in enumerate(events):
        call_id = "data-" + identity + "-" + str(index)
        result = canonical(event["result"]).decode()
        strands.extend([
            {"role": "assistant", "content": [{"toolUse": {
                "name": event["tool"], "input": event["arguments"], "toolUseId": call_id}}]},
            {"role": "user", "content": [{"toolResult": {
                "toolUseId": call_id, "status": "success", "content": [{"text": result}]}}]},
        ])
        call = {"function": {"arguments": event["arguments"], "name": event["tool"]},
                "id": call_id, "type": "function"}
        native.extend([{"role": "assistant", "content": "", "tool_calls": [call]},
                       {"role": "tool", "tool_call_id": call_id, "content": result}])
        wire_call = deepcopy(call)
        wire_call["function"]["arguments"] = json.dumps(event["arguments"])
        wire.extend([{"role": "assistant", "content": [], "tool_calls": [wire_call]},
                     {"role": "tool", "tool_call_id": call_id, "content": [{"text": result, "type": "text"}]}])
    require(contract["final_acknowledgement"] == "Diagnosis recorded.", "Final acknowledgement differs")
    final = contract["final_acknowledgement"]
    strands.append({"role": "assistant", "content": [{"text": final}]})
    wire.append({"role": "assistant", "content": [{"text": final, "type": "text"}]})
    native.append({"role": "assistant", "content": final})
    require(execution["strands_messages"] == strands and execution["wire_messages"] == wire
            and execution["messages"] == native, "Complete reference messages differ")
    job = {"device_id": identity, "task_kind": "equipment_interface_v1",
           "trace": {"tool_events": events, "model_calls": []},
           "submission": execution["submission"], "status": "finished"}
    checked = inspect_job(fixture, canonical(document), {"record": job, "digest": digest(canonical(job))})
    require(checked == reference["audit"] and checked["semantic_passed"] is True,
            "Independent tool-result and answer interpretation differs")
    require(reference["replay_identical"] is True and reference["training_admitted"] is False
            and reference["qualification_passed"] is False, "Reference overclaims admission")
    require(not any(key in canonical(native).decode() for key in
                    ("expected_status", "expected_reader", "decimal_oracle", "admission_ticket")),
            "Private reference metadata entered model messages")
    return truth


def inspect_record(record, contract, release):
    if record["schema"] != SCHEMA:
        return inspect_original(record, contract, release)
    require(record["provenance"] == PROVENANCE, "Refinement provenance changed")
    require(digest(canonical({k: v for k, v in contract.items() if k != "sha256"}))
            == contract["sha256"] == record["tool_contract_sha256"], "Refinement contract changed")
    reference = record["reference"]
    truth = inspect_probe(reference, contract)
    expected_case = deepcopy(reference["case"])
    expected_case["split"] = "train"
    require(record["case"] == expected_case
            and all(record[k] == expected_case[k] for k in ("case_id", "role", "split", "profile_id")),
            "Normalized refinement case differs")
    require(all(record[k] == v for k, v in reference["execution"].items()),
            "Normalization changed executed messages or targets")
    require(record["supervised_assistants"] == [True] * 5
            and all(type(v) is bool for v in record["supervised_assistants"]),
            "Refinement supervision mask differs")
    return {"passed": True, "semantic_answer_verified": True, "rational_fixture_truth_verified": True,
            "complete_assistant_targets": 5, "status": truth["status"],
            "training_admitted": False, "qualification_passed": False}


def targets(record):
    if record["schema"] != SCHEMA:
        yield from original_targets(record)
        return
    assistants = [(i, m) for i, m in enumerate(record["messages"]) if m["role"] == "assistant"]
    require(len(assistants) == 5 and record["supervised_assistants"] == [True] * 5,
            "Refinement target inventory differs")
    for ordinal, (index, message) in enumerate(assistants):
        yield ordinal, index, message


def verify_dataset(source):
    """Reconstruct parent byte prefixes and bind the exact reviewed supplement."""
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_bytes())
    require(manifest["schema"] == CORPUS_SCHEMA, "Unexpected refinement corpus")
    require(all(manifest[k] is False for k in
                ("training_admitted", "optimization_started", "qualification_data_present", "human_review_claimed")),
            "Refinement corpus overclaims admission")
    require(set(manifest["files"]) == {
        "train.jsonl", "validation.jsonl", "contracts.json", "release.json", "references.jsonl", "parent-manifest.json",
    }, "Refinement file inventory differs")
    files = {}
    for name, expected in manifest["files"].items():
        path = source / name
        require(not path.is_symlink(), "Refinement input is a link")
        raw = path.read_bytes()
        require(expected == {"bytes": len(raw), "sha256": digest(raw)}, "Refinement input identity changed: " + name)
        files[name] = raw
    require(digest(files["parent-manifest.json"]) == PARENT_SHA
            and digest(files["references.jsonl"]) == REFERENCE_SHA, "Pinned refinement source changed")
    parent = json.loads(files["parent-manifest.json"])
    for name in ("contracts.json", "release.json", "validation.jsonl"):
        require(manifest["files"][name] == parent["files"][name], "A preserved parent input changed")
    train_lines = files["train.jsonl"].splitlines(keepends=True)
    prefix = b"".join(train_lines[:372])
    require({"bytes": len(prefix), "sha256": digest(prefix)} == parent["files"]["train.jsonl"],
            "Original training trajectory prefix changed")
    groups = {split: [json.loads(line) for line in files[split + ".jsonl"].splitlines()]
              for split in ("train", "validation")}
    require(len(groups["train"]) == 456 and len(groups["validation"]) == 128, "Refinement split counts differ")
    references = [json.loads(line) for line in files["references.jsonl"].splitlines()]
    require(len(references) == 84 and [r["reference"] for r in groups["train"][372:]] == references,
            "Normalized supplement changed or reordered references")
    expected_pairs = {(f"environment-{p}", v) for p in FAMILIES for v in VARIANTS}
    require({(r["case"]["profile_id"], r["case"]["variation"]) for r in references} == expected_pairs,
            "Incomplete training-only refinement coverage")
    identities, profiles = set(), set()
    for split, rows in groups.items():
        ids = {r["case_id"] for r in rows}
        current_profiles = {r["profile_id"] for r in rows}
        require(len(ids) == len(rows) and not ids & identities and not current_profiles & profiles
                and all(r["split"] == split for r in rows), "Refinement splits overlap or move cases")
        identities.update(ids)
        profiles.update(current_profiles)
    return manifest, groups


async def audit(source, parent):
    from importlib.metadata import version
    from strands.models import LlamaCppModel
    from .audit_startup_dataset import audit as audit_parent
    from .equipment_dataset import execute_case

    require(version("strands-agents") == "1.57.1", "Runtime version differs")
    require(digest((Path(parent) / "manifest.json").read_bytes()) == PARENT_SHA, "Wrong replay parent")
    parent_result = await audit_parent(parent)
    require(parent_result["passed"] is True, "Original runtime replay failed")
    manifest, groups = verify_dataset(source)
    contracts = json.loads((Path(source) / "contracts.json").read_bytes())
    release = json.loads((Path(source) / "release.json").read_bytes())
    model = LlamaCppModel(base_url="http://127.0.0.1:9", model_id="reference-formatting-only",
                         params={"temperature": 0, "max_tokens": 192})
    statuses = Counter()
    try:
        for split, records in groups.items():
            for record in records:
                require(inspect_record(record, contracts[record["role"]], release) == record["evaluation"],
                        "Independent record semantics differ")
                if record["schema"] == SCHEMA:
                    actual = await execute_case(record["case"], model, contracts["equipment"], release,
                                                supplied_actions=record["actions"],
                                                final=record["messages"][-1]["content"])
                    require(all(record[name] == value for name, value in actual.items()),
                            "New reference runtime replay differs")
                    statuses[record["evaluation"]["status"]] += 1
    finally:
        await model.client.aclose()
    require(statuses == {"low": 12, "normal": 24, "high": 12, "unknown": 36}, "Refinement status balance differs")
    return {"passed": True, "source_manifest_sha256": digest((Path(source) / "manifest.json").read_bytes()),
            "parent_trajectories_replayed": 500, "supplement_trajectories_replayed": 84,
            "counts": {s: {"trajectories": len(rs), "assistant_targets": sum(len(list(targets(r))) for r in rs)}
                       for s, rs in groups.items()}, "supplement_status_counts": dict(statuses),
            "current_runtime_replay_verified": True, "independent_semantics_verified": True,
            "provider_formatting_verified": True, "training_admitted": False, "qualification_passed": False}
