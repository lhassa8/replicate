"""Independent observation, answer, transcript and subset checks for v6 data."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from .audit_equipment_diagnostics import inspect_job
from .audit_runtime_dataset import (
    canonical, digest, require, diagnostic_expectation, reproduction_expectation,
)


SCHEMA = "organism-first-lineage-runtime-trajectory-v1"
CORPUS_SCHEMA = "organism-first-lineage-runtime-corpus-v1"
PARENT_SPLITS = {
    "train": ("organism/data/equipment-runtime-development-v5-refinement-01/train.jsonl",
              "db3183fd07ab30d5d6d58e3e7cbc2b1069444f4104868b3d61708aff05e30a82"),
    "validation": ("organism/data/equipment-runtime-development-v5-refinement-01/validation.jsonl",
                   "624f75edd24b9855a09527324c3190545f0b6d70f0a094344fb7d29f5ed687ac"),
}
EXPECTED_COUNTS = {
    "train": {"trajectories": 172, "assistant_targets": 716,
              "roles": {"equipment": 88, "reproduction": 60, "diagnostic": 24}},
    "validation": {"trajectories": 62, "assistant_targets": 258,
                   "roles": {"equipment": 30, "reproduction": 20, "diagnostic": 12}},
}


def expected_messages(record, contract):
    role, case = record["role"], record["case"]
    user = (("Diagnose the equipment " + case["case_id"]) if role == "equipment"
            else ("Assess the device " + case["case_id"]) if role == "diagnostic"
            else ("Consider discovered device " + case["peer_id"])) + "."
    strands = [{"role": "user", "content": [{"text": user}]}]
    wire = [{"role": "system", "content": contract["system_prompt"]},
            {"role": "user", "content": [{"type": "text", "text": user}]}]
    native = [{"role": "system", "content": contract["system_prompt"]}, {"role": "user", "content": user}]
    for index, event in enumerate(record["events"]):
        identifier = "data-" + case["case_id"] + "-" + str(index)
        result = canonical(event["result"]).decode()
        call = {"name": event["tool"], "input": event["arguments"], "toolUseId": identifier}
        strands.extend([
            {"role": "assistant", "content": [{"toolUse": call}]},
            {"role": "user", "content": [{"toolResult": {
                "toolUseId": identifier, "status": "success", "content": [{"text": result}]}}]},
        ])
        function = {"name": event["tool"], "arguments": event["arguments"]}
        native.extend([
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "id": identifier, "function": function}]},
            {"role": "tool", "tool_call_id": identifier, "content": result},
        ])
        wire.extend([
            {"role": "assistant", "content": [], "tool_calls": [{
                "type": "function", "id": identifier, "function": {
                    "name": event["tool"], "arguments": json.dumps(event["arguments"])}}]},
            {"role": "tool", "tool_call_id": identifier, "content": [{"type": "text", "text": result}]},
        ])
    final = {"equipment": "Diagnosis recorded.", "diagnostic": "Assessment recorded.",
             "reproduction": "Choice recorded."}[role]
    require(contract["final_acknowledgement"] == final, "Missing complete final acknowledgement")
    strands.append({"role": "assistant", "content": [{"text": final}]})
    wire.append({"role": "assistant", "content": [{"type": "text", "text": final}]})
    native.append({"role": "assistant", "content": final})
    return {"strands_messages": strands, "wire_messages": wire, "messages": native}


def inspect_record(record, contract, release):
    require(record["schema"] == SCHEMA, "Unknown first-lineage trajectory schema")
    case, role = record["case"], record["role"]
    require(role in ("equipment", "diagnostic", "reproduction")
            and record["split"] in ("train", "validation")
            and all(record[key] == case[key] for key in ("case_id", "role", "split", "profile_id")),
            "Role or development case binding changed")
    require(record["provenance"] == {
        "kind": "previously_admitted_development_observations_replayed_through_versioned_tools",
        "teacher_model_used": False, "model_inference_used": False, "human_review_claimed": False,
        "real_peer_transport_used": False, "native_process_used": False,
        "training_admitted": False, "qualification_data_present": False,
    }, "Reference provenance overclaims verification")
    require(digest(canonical({k: v for k, v in contract.items() if k != "sha256"}))
            == contract["sha256"] == record["tool_contract_sha256"], "Tool contract changed")
    events = record["events"]
    require(record["actions"] == [{"tool": event["tool"], "arguments": event["arguments"]} for event in events],
            "Actions differ from actual tool events")
    mask = [True] * (len(events) + 1)
    if role == "equipment":
        require(contract["tool_profile"] == "organism-canonical-temperature-tools-v1"
                and case["fixture"]["document"]["task"]["quantity"] == "temperature",
                "Reference left the canonical temperature scope")
        names = ["inspect_equipment", "configure_reader", "read_equipment", "submit_diagnosis"]
        if case["inject_configuration_error"]:
            names.insert(1, "configure_reader")
            require(len(events) == 5, "Incorrect error-context sequence")
            bad = deepcopy(events[2]["arguments"])
            bad["value_selectors"] = bad["value_selectors"][0]
            require(events[1]["arguments"] == bad and events[1]["result"] == {
                "status": "invalid_observation_or_arguments", "error_type": "ValidationError"},
                "Injected error no longer matches its original typed error")
            mask[1] = False
        require([event["tool"] for event in events] == names, "Temperature procedure is incomplete")
        job = {"device_id": case["case_id"], "task_kind": "equipment_interface_v1", "status": "finished",
               "submission": record["submission"], "trace": {
                   "tool_profile": contract["tool_profile"], "tool_events": events}}
        result = inspect_job(case["fixture"], canonical(case["fixture"]["document"]),
                             {"record": job, "digest": digest(canonical(job))})
        require(result["semantic_passed"] is True and result["correct_reader_selected"] is True,
                "Temperature reference fails independent observation and answer checks")
    else:
        if role == "diagnostic":
            expected, submission = diagnostic_expectation(case)
        else:
            require(contract["tool_profile"] == "organism-reproduction-platform-choice-v2",
                    "Reproduction reference uses an ambiguous platform protocol")
            expected, submission = reproduction_expectation(case, release)
            # Independent wire decoding: the model explicitly selects the enum
            # "none" when deferring; the retained semantic choice has no platform.
            if expected[-1]["arguments"]["platform"] is None:
                expected[-1]["arguments"]["platform"] = "none"
        require(events == expected and record["submission"] == submission,
                "Reference actions, observations or answers fail independent semantics")
    require(record["supervised_assistants"] == mask
            and all(type(value) is bool for value in record["supervised_assistants"]),
            "Missing required target or injected bad action entered supervision")
    for name, expected in expected_messages(record, contract).items():
        require(record[name] == expected, "Complete target or provider transcript differs: " + name)
    text = canonical(record["messages"]).decode()
    require(not any(marker in text for marker in (
        "expected_status", "expected_reader", "expected_value", "decimal_oracle", "admission_ticket")),
        "A private oracle field or admission authority entered model context")
    return {
        "passed": True, "semantic_answer_verified": True,
        "complete_assistant_targets": sum(mask), "context_only_injected_actions": mask.count(False),
        "final_acknowledgement_verified": True, "training_admitted": False,
        "native_execution_verified": False, "qualification_passed": False,
    }


def targets(record):
    assistants = [(i, m) for i, m in enumerate(record["messages"]) if m["role"] == "assistant"]
    mask = record["supervised_assistants"]
    require(len(mask) == len(assistants) and all(type(value) is bool for value in mask),
            "Invalid supervision mask")
    for ordinal, ((index, message), selected) in enumerate(zip(assistants, mask)):
        if selected:
            yield ordinal, index, message


async def audit(source):
    from strands.models import LlamaCppModel
    from . import first_lineage_dataset as dataset

    root, source = Path(__file__).resolve().parents[1], Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_bytes())
    require(manifest["schema"] == CORPUS_SCHEMA and manifest["counts"] == EXPECTED_COUNTS
            and manifest["scope"] == "development_references_through_current_runtime_tools"
            and manifest["target_codec"] == "granite-json-scalar-targets-v1",
            "First-lineage corpus composition or target encoding changed")
    require(all(manifest[key] is False for key in
                ("training_admitted", "optimization_started", "qualification_data_present",
                 "human_review_claimed", "teacher_model_used"))
            and manifest["model_generations"] == 0, "Reference corpus overclaims readiness")
    require(set(manifest["files"]) == {"train.jsonl", "validation.jsonl", "contracts.json", "release.json"},
            "Unexpected source-file inventory")
    for name, expected in manifest["files"].items():
        raw = (source / name).read_bytes()
        require(expected == {"bytes": len(raw), "sha256": digest(raw)}, "Dataset file changed")
    require(set(manifest["sources"]) == set(dataset.SOURCES), "Dataset source inventory changed")
    for name, expected in manifest["sources"].items():
        require(Path(name).name == name
                and digest((source / "sources" / name).read_bytes()) == expected
                and digest((root / "organism" / name).read_bytes()) == expected,
                "Current or retained dataset implementation changed")
    frozen_path = root / "organism/evidence/first-lineage-milestone-20261016-01/temperature-cases.frozen.json"
    require(digest(frozen_path.read_bytes()) ==
            "aff7cf82494c5f1f194b0272245835b030990a978786c1ec51aac2d0889e9685",
            "Frozen temperature evaluation changed")
    frozen = json.loads(frozen_path.read_bytes())["cases"]
    reserved = {case["case_id"] for case in frozen}
    definitions = json.loads((source / "contracts.json").read_bytes())
    release = json.loads((source / "release.json").read_bytes())
    recipe_item = manifest["parent_recipe"]
    recipe_raw = (root / recipe_item["path"]).read_bytes()
    require({"bytes": len(recipe_raw), "sha256": digest(recipe_raw)}
            == {key: recipe_item[key] for key in ("bytes", "sha256")}, "Parent admission recipe changed")
    expected_release = json.loads(recipe_raw)["inputs"]["release"]
    require(manifest["files"]["release.json"] ==
            {key: expected_release[key] for key in ("bytes", "sha256")}, "Parent public release metadata changed")
    model = LlamaCppModel(base_url="http://127.0.0.1:9", model_id="reference-formatting-only",
                         params={"temperature": 0, "max_tokens": 192})
    counts = {}
    try:
        require(definitions == dataset.contracts(model), "Registered current runtime contract changed")
        for split, (parent_path, parent_sha) in PARENT_SPLITS.items():
            parent_raw = (root / parent_path).read_bytes()
            require(digest(parent_raw) == parent_sha, "Previously admitted development observations changed")
            parents = {}
            for line in parent_raw.splitlines():
                parent = json.loads(line)
                if parent["role"] in ("diagnostic", "reproduction") or (
                    parent["role"] == "equipment"
                    and parent["case"]["fixture"]["document"]["task"]["quantity"] == "temperature"
                ):
                    parents[parent["case_id"]] = parent
            seen, roles, total = set(), Counter(), 0
            for line in (source / (split + ".jsonl")).read_text().splitlines():
                record = json.loads(line)
                identifier = record["case_id"]
                require(identifier in parents and identifier not in seen and identifier not in reserved,
                        "Missing parent, repeated reference or frozen evaluation case in corpus")
                parent = parents[identifier]
                require(record["case"] == parent["case"] and record["split"] == split
                        and record["parent_record_sha256"] == digest(canonical(parent)),
                        "Reference changed an admitted observation or moved a development split")
                contract = definitions[record["role"]]
                require(inspect_record(record, contract, release) == record["evaluation"],
                        "Saved semantic evaluation differs")
                replay = await dataset.execute_case(record["case"], model, contract, release,
                                                     supplied_actions=record["actions"])
                require(all(record[key] == value for key, value in replay.items()),
                        "Current runtime replay differs from recorded reference")
                seen.add(identifier)
                roles[record["role"]] += 1
                total += len(list(targets(record)))
            require(seen == set(parents), "Selected parent subset is incomplete")
            counts[split] = {"trajectories": len(seen), "assistant_targets": total, "roles": dict(roles)}
        require(counts == EXPECTED_COUNTS, "Reconstructed role or target counts differ")
    finally:
        await model.client.aclose()
    return {
        "passed": True, "source_manifest_sha256": digest((source / "manifest.json").read_bytes()),
        "counts": counts, "current_runtime_replay_verified": True,
        "independent_semantics_verified": True, "provider_formatting_verified": True,
        "prior_observations_and_splits_preserved": True,
        "frozen_temperature_cases_excluded": True, "sealed_qualification_data_accessed": False,
        "training_admitted": False, "optimization_started": False, "qualification_passed": False,
    }
