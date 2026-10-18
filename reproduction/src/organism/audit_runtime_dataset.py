"""Independent semantic checks for development references, never admission.

The record inspector uses only the standard library and saved observations.
Full audit additionally replays actual tools and compares the current provider.
"""

import argparse
import asyncio
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def diagnostic_expectation(case):
    raw = case["observation"]
    device, sensor = raw["device_id"], raw["sensor_id"]
    require(device == case["case_id"], "Observation device differs from case")
    capability = {key: raw[key] for key in
                  ("device_id", "sensor_id", "unit", "normal_range_celsius", "maximum_observation_age_seconds")}
    first = {"status": "ok", "device_id": device, "evidence_id": digest(canonical(capability)),
             "sensors": [{"quantity": "temperature", **{k: v for k, v in capability.items() if k != "device_id"}}]}
    age = case["clock"] - raw["observed_unix_seconds"]
    require(age >= 0, "Observation is from the future")
    reading = {"status": "ok", "device_id": device, "sensor_id": sensor, "value": raw["value"], "unit": raw["unit"],
               "age_seconds": age, "sequence": raw["sequence"], "observed_unix_seconds": raw["observed_unix_seconds"],
               "observation_sha256": digest(canonical(raw))}
    reading["evidence_id"] = digest(canonical(reading))
    if age > raw["maximum_observation_age_seconds"]:
        status, temperature = "unknown", None
    else:
        value = Decimal(str(raw["value"]))
        require(raw["unit"] in ("C", "F", "K"), "Unknown temperature unit")
        # Decimal arithmetic is independent of the reference's floating conversion.
        value = value if raw["unit"] == "C" else ((value - 32) * 5 / 9 if raw["unit"] == "F" else value - Decimal("273.15"))
        temperature = float(round(value, 6))
        lower, upper = map(lambda item: Decimal(str(item)), raw["normal_range_celsius"])
        status = "low" if value < lower else ("high" if value > upper else "normal")
    answer = {"device_id": device, "status": status, "temperature_celsius": temperature,
              "evidence_ids": [first["evidence_id"], reading["evidence_id"]]}
    events = [
        {"tool": "inspect_device", "arguments": {"device_id": device}, "result": first},
        {"tool": "read_measurement", "arguments": {"device_id": device, "sensor_id": sensor}, "result": reading},
        {"tool": "submit_assessment", "arguments": answer, "result": {"status": "recorded", "independently_verified": False}},
    ]
    return events, answer


def reproduction_expectation(case, release):
    device = case["peer_id"]
    policy_id = digest((case["profile_id"] + "/synthetic-policy").encode())
    public = {"device_id": device, "policy_id": policy_id, "observation_valid_seconds": 30,
              **{key: case[key] for key in ("platform", "owner_mode", "available", "revision")}}
    observed = {"device_id": device, **{key: public[key] for key in ("platform", "owner_mode", "available", "revision")},
                "supported_platforms": sorted(release["body"]["platforms"]), "release_id": release["release_id"],
                "package_bytes": sum(item["bytes"] for item in release["body"]["files"].values()),
                "transport_receipt": {"sequence": 1, "digest": digest(canonical(public))}}
    observed["evidence_id"] = digest(canonical(observed))
    eligible = case["owner_mode"] == "enabled" and case["available"] and case["platform"] in observed["supported_platforms"]
    answer = {"action": "establish" if eligible else "defer", "platform": case["platform"] if eligible else None,
              "evidence_id": observed["evidence_id"]}
    submission = {**answer, "device_id": device, "observed_revision": case["revision"], "release_id": release["release_id"]}
    events = [
        {"tool": "inspect_candidate", "arguments": {"device_id": device}, "result": observed},
        {"tool": "choose_child_action", "arguments": answer, "result": {"status": "choice_recorded", "child_established": False}},
    ]
    return events, submission


def inspect_record(record, contract, release):
    require(record["schema"] == "organism-runtime-trajectory-v1", "Unknown runtime trajectory schema")
    body = {key: value for key, value in contract.items() if key != "sha256"}
    require(digest(canonical(body)) == contract["sha256"] == record["tool_contract_sha256"], "Tool contract hash differs")
    require(record["provenance"] == {
        "kind": "locally_authored_deterministic_runtime_tool_reference", "teacher_model_used": False,
        "human_review_claimed": False, "real_peer_transport_used": False, "native_process_used": False,
        "training_admitted": False, "qualification_data_present": False,
    }, "Reference provenance changed or overclaims verification")
    case = record["case"]
    require(all(record[key] == case[key] for key in ("case_id", "role", "split", "profile_id")), "Case binding differs")
    require(record["split"] in ("train", "validation"), "Qualification data is not accepted")
    role = record["role"]
    require(role in ("diagnostic", "reproduction"), "Unsupported runtime-data role")
    events, submission = diagnostic_expectation(case) if role == "diagnostic" else reproduction_expectation(case, release)
    require(record["events"] == events and record["submission"] == submission, "Recorded answer or observations fail independent semantics")
    if role == "diagnostic" and record["submission"]["temperature_celsius"] is not None:
        require(type(record["submission"]["temperature_celsius"]) in (int, float), "Temperature must be numeric")
    actions = [{key: event[key] for key in ("tool", "arguments")} for event in events]
    require(record["actions"] == actions, "Action transcript differs from verified events")
    expected_final = "Assessment recorded." if role == "diagnostic" else "Choice recorded."
    require(contract["final_acknowledgement"] == expected_final, "Unexpected final acknowledgement")
    user = ("Assess the device " + case["case_id"] if role == "diagnostic" else
            "Consider discovered device " + case["peer_id"]) + "."
    strands = [{"role": "user", "content": [{"text": user}]}]
    wire = [{"role": "system", "content": contract["system_prompt"]},
            {"role": "user", "content": [{"type": "text", "text": user}]}]
    native = [{"role": "system", "content": contract["system_prompt"]}, {"role": "user", "content": user}]
    # Preserve equivalent integer/float spellings after checking their meaning.
    for index, event in enumerate(record["events"]):
        identity = "data-" + record["case_id"] + "-" + str(index)
        text = canonical(event["result"]).decode()
        call = {"name": event["tool"], "input": event["arguments"], "toolUseId": identity}
        result = {"toolUseId": identity, "status": "success", "content": [{"text": text}]}
        strands.extend([{"role": "assistant", "content": [{"toolUse": call}]},
                        {"role": "user", "content": [{"toolResult": result}]}])
        function = {"name": event["tool"], "arguments": event["arguments"]}
        native.extend([{"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "id": identity, "function": function}]},
            {"role": "tool", "tool_call_id": identity, "content": text}])
        wire.extend([{"role": "assistant", "content": [], "tool_calls": [
            {"type": "function", "id": identity, "function": {
                "name": event["tool"], "arguments": json.dumps(event["arguments"])}}]},
            {"role": "tool", "tool_call_id": identity, "content": [{"type": "text", "text": text}]}])
    strands.append({"role": "assistant", "content": [{"text": expected_final}]})
    wire.append({"role": "assistant", "content": [{"type": "text", "text": expected_final}]})
    native.append({"role": "assistant", "content": expected_final})
    for field, expected in (("strands_messages", strands), ("wire_messages", wire), ("messages", native)):
        require(record[field] == expected, "Message ordering, evidence, target or formatting differs: " + field)
    require("admission_ticket" not in canonical(native).decode(), "Admission authority leaked to model")
    return {"passed": True, "semantic_answer_verified": True, "complete_assistant_targets": len(events) + 1,
            "final_acknowledgement_verified": True, "training_admitted": False,
            "native_execution_verified": False, "qualification_passed": False}


async def audit(source):
    # Imported only for full replay; inspect_record stays separate from the generator.
    from importlib.metadata import version
    from strands.models import LlamaCppModel
    from . import runtime_dataset as runtime

    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_bytes())
    require(manifest["schema"] == runtime.SCHEMA and manifest["strands_version"] == version("strands-agents") == "1.57.1",
            "Dataset schema or provider version changed")
    require(manifest["scope"] == "development_references_through_current_runtime_tools"
            and all(manifest[key] is False for key in ("training_admitted", "optimization_started", "qualification_data_present")),
            "Dataset scope or provenance changed")
    require(set(manifest["files"]) == {"train.jsonl", "validation.jsonl", "contracts.json", "release.json"},
            "Dataset file inventory changed")
    for name, expected in manifest["files"].items():
        raw = (source / name).read_bytes()
        require(expected == {"bytes": len(raw), "sha256": digest(raw)}, "Dataset file hash differs: " + name)
    paths = {name: Path(runtime.__file__).parent / name for name in runtime.SOURCES}
    paths.update(runtime.provider_sources())
    require(set(manifest["sources"]) == set(paths), "Source inventory changed")
    for name, path in paths.items():
        require(digest(path.read_bytes()) == digest((source / "sources" / name).read_bytes()) == manifest["sources"][name],
                "Current or retained source differs: " + name)
    definitions = json.loads((source / "contracts.json").read_bytes())
    release = json.loads((source / "release.json").read_bytes())
    require(release == runtime.stock_release(Path(runtime.__file__).parent), "Carried release metadata differs")
    cases = {case["case_id"]: case for case in runtime.development_cases()}
    require(len(cases) == 320, "Expected development case inventory changed")
    model = LlamaCppModel(base_url="http://127.0.0.1:9", model_id="reference-formatting-only",
                         params={"temperature": 0, "max_tokens": 192})
    counts, seen, profiles = {}, set(), set()
    try:
        require(definitions == runtime.contracts(model), "Current runtime tool contract differs")
        for split in ("train", "validation"):
            records = [json.loads(line) for line in (source / (split + ".jsonl")).read_text().splitlines()]
            split_profiles = {record["profile_id"] for record in records}
            require(not profiles & split_profiles, "Parameter profile overlaps development splits")
            profiles.update(split_profiles)
            targets = 0
            for record in records:
                identity = record["case_id"]
                require(identity not in seen and cases.get(identity) == record["case"] and record["split"] == split,
                        "Duplicate, moved or changed development case")
                seen.add(identity)
                checked = inspect_record(record, definitions[record["role"]], release)
                require(checked == record["evaluation"], "Saved evaluation differs")
                await runtime.replay(record, model, definitions[record["role"]], release)
                targets += checked["complete_assistant_targets"]
            counts[split] = {"trajectories": len(records), "profiles": sorted(split_profiles),
                             "roles": {role: sum(record["role"] == role for record in records) for role in runtime.ROLES},
                             "assistant_targets": targets}
        require(seen == set(cases) and counts == manifest["counts"], "Incomplete development inventory or counts")
    finally:
        await model.client.aclose()
    return {"passed": True, "source_manifest_sha256": digest((source / "manifest.json").read_bytes()),
            "counts": counts, "current_runtime_replay_verified": True, "independent_semantics_verified": True,
            "provider_formatting_verified": True, "serving_template_parity_verified": False,
            "training_admitted": False, "qualification_passed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.output.open("x") as destination:
        try:
            report = asyncio.run(audit(args.source))
        except Exception as exc:
            report = {"passed": False, "failure": type(exc).__name__ + ": " + str(exc), "training_admitted": False}
        destination.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
