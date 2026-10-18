"""Independent semantics, message reconstruction and replay for equipment data."""

import argparse
import asyncio
from decimal import Decimal
import json
from pathlib import Path

from .audit_equipment_diagnostics import inspect_job, resolve
from .audit_runtime_dataset import canonical, digest, inspect_record as inspect_legacy, require


FAMILIES = {
    "gateway-json-v1": ("environment-a", "train"),
    "gateway-register-v2": ("environment-b", "train"),
    "condition-json-v1": ("environment-c", "train"),
    "condition-register-v3": ("environment-d", "train"),
    "inspection-status-v1": ("environment-e", "train"),
    "event-counter-v2": ("environment-f", "train"),
    "thermal-status-v4": ("environment-g", "validation"),
    "inspection-register-v5": ("environment-h", "validation"),
}


def verify_fixture(case):
    fixture = case["fixture"]
    document = fixture["document"]
    require(document["family"] in FAMILIES
            and FAMILIES[document["family"]] == (case["profile_id"], case["split"])
            and case["family_id"] == document["family"], "Equipment family crosses or escapes development splits")
    require(case["variation"] in ("nominal", "fault", "scaled_fault", "stale", "invalid", "missing")
            and type(case["seed"]) is int and 0 <= case["seed"] < 5
            and case["clock"] == 1700000000 + case["seed"] * 1000, "Unexpected development condition")
    require(fixture["case_id"] == document["device_id"] == case["case_id"]
            and fixture["profile"] == case["profile_id"] and fixture["variation"] == case["variation"],
            "Equipment fixture identity differs")
    require(type(case["inject_configuration_error"]) is bool
            and case["inject_configuration_error"] == (case["variation"] == "nominal" and case["seed"] == 4),
            "Injected-error condition changed")
    reader = fixture["expected_reader"]
    source = next(item for item in document["sources"] if item["source_id"] == reader["source_id"])
    require(len(reader["value_selectors"]) == 1, "Development fixture has an unexpected measurement count")
    selected = reader["value_selectors"][0]
    field = next(item for item in source["fields"] if item["selector"] == selected)
    value = resolve(source, selected)
    timestamp = resolve(source, reader["timestamp_selector"])
    valid = True if reader["validity_selector"] is None else resolve(source, reader["validity_selector"])
    unknown = (type(value) not in (int, float) or type(timestamp) not in (int, float) or valid is not True
               or timestamp > case["clock"] or case["clock"] - timestamp > document["task"]["maximum_age_seconds"])
    expected_value, status = None, "unknown"
    if not unknown:
        converted = Decimal(str(value)) * Decimal(str(field["scale"]))
        if field["unit"] == "F":
            converted = (converted - 32) * 5 / 9
        elif field["unit"] == "K":
            converted -= Decimal("273.15")
        elif field["unit"] == "in/s":
            converted *= Decimal("25.4")
        elif field["unit"] == "ratio":
            converted *= 100
        else:
            require(field["unit"] in ("C", "mm/s", "percent", "count"), "Unknown reference measurement unit")
        low, high = map(lambda item: Decimal(str(item)), document["task"]["normal_range"])
        status = "low" if converted < low else "high" if converted > high else "normal"
        expected_value = float(round(converted, 6))
    require(fixture["expected_status"] == status and fixture["expected_value"] == expected_value
            and fixture["expected_unit"] == document["task"]["canonical_unit"],
            "Private fixture truth disagrees with independent decimal interpretation")


def inspect_record(record, contract, release):
    if record["schema"] == "organism-runtime-trajectory-v1":
        require(record["role"] == "reproduction", "Legacy temperature rows cannot replace equipment coverage")
        return inspect_legacy(record, contract, release)
    require(record["schema"] == "organism-equipment-runtime-trajectory-v1" and record["role"] == "equipment",
            "Unknown equipment trajectory schema")
    body = {key: value for key, value in contract.items() if key != "sha256"}
    require(digest(canonical(body)) == contract["sha256"] == record["tool_contract_sha256"],
            "Equipment tool contract changed")
    require(record["provenance"] == {
        "kind": "locally_authored_runtime_tool_reference_with_declared_error_context",
        "teacher_model_used": False, "human_review_claimed": False, "real_peer_transport_used": False,
        "native_process_used": False, "training_admitted": False, "qualification_data_present": False,
    }, "Reference provenance changed or overclaims verification")
    case = record["case"]
    require(all(record[key] == case[key] for key in ("case_id", "role", "split", "profile_id")),
            "Equipment case binding differs")
    verify_fixture(case)
    injected = case["inject_configuration_error"]
    names = ["inspect_equipment", "configure_reader", "read_equipment", "submit_diagnosis"]
    if injected:
        names.insert(1, "configure_reader")
    events = record["events"]
    require([event["tool"] for event in events] == names, "Reference sequence changed or lacks completion")
    require(record["actions"] == [{key: event[key] for key in ("tool", "arguments")} for event in events],
            "Actions differ from the executed reference")
    if injected:
        expected_bad = dict(events[2]["arguments"])
        expected_bad["value_selectors"] = expected_bad["value_selectors"][0]
        require(events[1]["arguments"] == expected_bad and events[1]["result"] == {
            "status": "invalid_observation_or_arguments", "error_type": "ValidationError"},
            "Injected error no longer matches the declared invalid argument")
    mask = [True] * (len(events) + 1)
    if injected:
        mask[1] = False
    require(record["supervised_assistants"] == mask
            and all(type(value) is bool for value in record["supervised_assistants"]),
            "Injected bad action entered targets or a required positive target was removed")
    job_record = {"device_id": case["case_id"], "task_kind": "equipment_interface_v1", "status": "finished",
                  "submission": record["submission"], "trace": {"tool_events": events}}
    job = {"record": job_record, "digest": digest(canonical(job_record))}
    checked = inspect_job(case["fixture"], canonical(case["fixture"]["document"]), job)
    require(checked["semantic_passed"] is True, "Equipment reference failed independent observation/answer checks")
    final = "Diagnosis recorded."
    require(contract["final_acknowledgement"] == final, "Equipment acknowledgement changed")
    user = "Diagnose the equipment " + case["case_id"] + "."
    strands = [{"role": "user", "content": [{"text": user}]}]
    wire = [{"role": "system", "content": contract["system_prompt"]},
            {"role": "user", "content": [{"type": "text", "text": user}]}]
    native = [{"role": "system", "content": contract["system_prompt"]}, {"role": "user", "content": user}]
    for index, event in enumerate(events):
        identity = "data-" + case["case_id"] + "-" + str(index)
        text = canonical(event["result"]).decode()
        call = {"name": event["tool"], "input": event["arguments"], "toolUseId": identity}
        result = {"toolUseId": identity, "status": "success", "content": [{"text": text}]}
        strands.extend([{"role": "assistant", "content": [{"toolUse": call}]},
                        {"role": "user", "content": [{"toolResult": result}]}])
        native.extend([{"role": "assistant", "content": "", "tool_calls": [{
            "type": "function", "id": identity,
            "function": {"name": event["tool"], "arguments": event["arguments"]}}]},
            {"role": "tool", "tool_call_id": identity, "content": text}])
        wire.extend([{"role": "assistant", "content": [], "tool_calls": [{
            "type": "function", "id": identity,
            "function": {"name": event["tool"], "arguments": json.dumps(event["arguments"])}}]},
            {"role": "tool", "tool_call_id": identity, "content": [{"type": "text", "text": text}]}])
    strands.append({"role": "assistant", "content": [{"text": final}]})
    wire.append({"role": "assistant", "content": [{"type": "text", "text": final}]})
    native.append({"role": "assistant", "content": final})
    for field, expected in (("strands_messages", strands), ("wire_messages", wire), ("messages", native)):
        require(record[field] == expected, "Reference message or complete target differs: " + field)
    require(all(key not in canonical(native).decode() for key in ("expected_status", "expected_reader", "admission_ticket")),
            "Private oracle or admission authority entered model messages")
    return {"passed": True, "semantic_answer_verified": True, "decimal_fixture_truth_verified": True,
            "complete_assistant_targets": sum(mask), "context_only_injected_actions": int(injected),
            "final_acknowledgement_verified": True, "runtime_call_budget_compatible": len(events) + 1 <= 6,
            "training_admitted": False, "native_execution_verified": False, "qualification_passed": False}


async def audit(source):
    from importlib.metadata import version
    from strands.models import LlamaCppModel
    from . import equipment_dataset as dataset
    from .runtime_rows import targets
    from .runtime_dataset import provider_sources, stock_release

    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_bytes())
    require(manifest["schema"] == "organism-equipment-runtime-corpus-v1"
            and manifest["scope"] == "development_references_through_current_runtime_tools"
            and manifest["strands_version"] == version("strands-agents") == "1.57.1",
            "Equipment corpus or runtime version changed")
    require(all(manifest[key] is False for key in
                ("training_admitted", "optimization_started", "qualification_data_present", "human_review_claimed")),
            "Corpus scope or provenance changed")
    require(set(manifest["files"]) == {"train.jsonl", "validation.jsonl", "contracts.json", "release.json"},
            "Unexpected corpus file inventory")
    for name, expected in manifest["files"].items():
        raw = (source / name).read_bytes()
        require(expected == {"bytes": len(raw), "sha256": digest(raw)}, "Corpus file hash changed")
    paths = {name: Path(dataset.__file__).parent / name for name in dataset.SOURCES}
    paths.update(provider_sources())
    require(set(paths) == set(manifest["sources"]), "Corpus source inventory changed")
    for name, path in paths.items():
        require(digest(path.read_bytes()) == digest((source / "sources" / name).read_bytes()) == manifest["sources"][name],
                "Current or retained corpus source changed: " + name)
    cases = {case["case_id"]: case for case in dataset.development_cases()}
    require(len(cases) == 320, "Incomplete declared development inventory")
    definitions = json.loads((source / "contracts.json").read_bytes())
    release = json.loads((source / "release.json").read_bytes())
    require(release == stock_release(Path(dataset.__file__).parent), "Retained reproduction package metadata changed")
    model = LlamaCppModel(base_url="http://127.0.0.1:9", model_id="reference-formatting-only",
                         params={"temperature": 0, "max_tokens": 192})
    counts, seen, profiles = {}, set(), set()
    try:
        require(dataset.contracts(model) == definitions, "Current registered tool contract differs")
        for split in ("train", "validation"):
            records = [json.loads(line) for line in (source / (split + ".jsonl")).read_text().splitlines()]
            split_profiles = {record["profile_id"] for record in records}
            require(not profiles & split_profiles, "Development profiles cross splits")
            profiles.update(split_profiles)
            for record in records:
                identity = record["case_id"]
                require(identity not in seen and record["case"] == cases.get(identity) and record["split"] == split,
                        "Duplicate, moved or changed development fixture")
                seen.add(identity)
                require(inspect_record(record, definitions[record["role"]], release) == record["evaluation"],
                        "Saved semantic audit differs")
                await dataset.replay(record, model, definitions[record["role"]], release)
            counts[split] = {
                "trajectories": len(records), "profiles": sorted(split_profiles),
                "roles": {role: sum(record["role"] == role for record in records) for role in definitions},
                "assistant_targets": sum(len(list(targets(record))) for record in records),
                "context_only_injected_actions": sum(record["role"] == "equipment"
                                                   and record["case"]["inject_configuration_error"] for record in records),
            }
        require(seen == set(cases) and counts == manifest["counts"], "Corpus coverage or target count differs")
        require(counts["train"]["roles"] == {"equipment": 180, "reproduction": 60}
                and counts["validation"]["roles"] == {"equipment": 60, "reproduction": 20}
                and counts["train"]["assistant_targets"] == 1080
                and counts["validation"]["assistant_targets"] == 360
                and counts["train"]["context_only_injected_actions"] == 6
                and counts["validation"]["context_only_injected_actions"] == 2,
                "Equipment/admission balance or injected-action count changed")
    finally:
        await model.client.aclose()
    return {"passed": True, "source_manifest_sha256": digest((source / "manifest.json").read_bytes()),
            "counts": counts, "current_runtime_replay_verified": True, "independent_semantics_verified": True,
            "provider_formatting_verified": True, "context_only_bad_actions_verified": 8,
            "serving_template_parity_verified": False, "training_admitted": False, "qualification_passed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as destination:
        try:
            result = asyncio.run(audit(args.source))
        except Exception as exc:
            result = {"passed": False, "failure": type(exc).__name__ + ": " + str(exc), "training_admitted": False}
        destination.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
