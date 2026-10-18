"""Independent readback of application traces, observations and task outcomes."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def streamed_arguments(streams):
    result = []
    for call, events in enumerate(streams):
        blocks, active, next_serial = {}, None, 0
        for event in events:
            start = event.get("contentBlockStart")
            if start is not None:
                # Strands' pinned llama.cpp provider emits serial blocks
                # without an index. Retain indexed-provider compatibility.
                active = start.get("contentBlockIndex", ("serial", next_serial))
                next_serial += 1
                if active in blocks:
                    raise ValueError("Streamed content block was started twice")
                tool = start.get("start", {}).get("toolUse")
                blocks[active] = ({"name": tool["name"], "arguments": "", "tool_use_id": tool["toolUseId"]}
                                  if tool is not None else None)
            delta = event.get("contentBlockDelta")
            if delta and "toolUse" in delta.get("delta", {}):
                index = delta.get("contentBlockIndex", active)
                if index not in blocks or blocks[index] is None:
                    raise ValueError("Streamed tool arguments have no matching start")
                blocks[index]["arguments"] += delta["delta"]["toolUse"]["input"]
            if "contentBlockStop" in event:
                active = None
        for item in blocks.values():
            if item is None:
                continue
            try:
                parsed = json.loads(item["arguments"])
                valid = isinstance(parsed, dict)
            except ValueError:
                valid = False
            result.append({"model_call": call + 1, **item, "valid_json_object": valid})
    return result


def match_streamed_tools(arguments, trace):
    """Check literal arguments against the SDK messages and executed tools."""
    returned = []
    for call, response in enumerate(trace["model_responses"], start=1):
        message = response.get("message") or {}
        returned.extend({"model_call": call, **block["toolUse"]} for block in message.get("content", [])
                        if "toolUse" in block)
    if len(returned) != len(arguments):
        raise ValueError("Literal tool calls differ from the parsed model responses")
    for raw, parsed in zip(arguments, returned):
        if (raw["model_call"] != parsed["model_call"] or raw["name"] != parsed["name"]
                or raw["tool_use_id"] != parsed["toolUseId"]):
            raise ValueError("Literal tool call identity differs from the parsed response")
        if raw["valid_json_object"] and json.loads(raw["arguments"]) != parsed["input"]:
            raise ValueError("Literal tool arguments differ from the parsed response")
    # Each model tool call must be represented by an executed or rejected tool
    # event. Malformed JSON may become an empty dictionary in the SDK; keeping
    # both representations makes that loss visible instead of repairing it.
    if len(returned) != len(trace["tool_events"]):
        raise ValueError("Parsed model tool calls differ from recorded tool executions")
    for parsed, event in zip(returned, trace["tool_events"]):
        if parsed["name"] != event["tool"] or parsed["input"] != event["arguments"]:
            raise ValueError("Executed tool arguments differ from the parsed model output")


def inspect_case(directory, item, run):
    case_id = item["case_id"]
    case = json.loads((directory / (case_id + ".json")).read_bytes())
    raw = (directory / "work-evidence/equipment" / (case_id + ".json")).read_bytes()
    return inspect_diagnostic(case, raw, item, run)


def inspect_diagnostic(case, raw, item, run):
    """Replay a saved diagnostic without depending on its capture layout."""
    case_id = item["case_id"]
    job, record = case["job"], case["job"]["record"]
    if sha(canonical(record)) != job["digest"] or record["request_id"] != case["request_id"]:
        raise ValueError("Saved job digest or request identity differs")
    source = json.loads(raw)
    if source != case["observation"] or source["device_id"] != case_id:
        raise ValueError("Saved equipment input differs from its recorded observation")
    trace = record["trace"]
    if len(trace["model_calls"]) > 4 or len(trace["tool_events"]) > 8:
        raise ValueError("Observed application exceeded its call allowances")
    if len(trace["http_requests"]) != len(trace["model_calls"]):
        raise ValueError("Model calls and actual HTTP requests differ")
    for request in trace["http_requests"]:
        if request["method"] != "POST" or request["path"] != "/v1/chat/completions":
            raise ValueError("Unexpected model endpoint or method")
        if request["body"].get("max_tokens") != 192 or request["body"].get("temperature") != 0:
            raise ValueError("Observed model generation limits differ")
    cap_id, reading, submission = None, None, None
    created = datetime.fromisoformat(record["created_at"]).timestamp()
    completed = datetime.fromisoformat(run["completed_at"]).timestamp()
    for event in trace["tool_events"]:
        tool, result, args = event["tool"], event["result"], event["arguments"]
        if result.get("status") == "ok" and tool == "inspect_device":
            expected = {key: source[key] for key in (
                "device_id", "sensor_id", "unit", "normal_range_celsius", "maximum_observation_age_seconds")}
            cap_id = sha(canonical(expected))
            if args != {"device_id": case_id} or result["evidence_id"] != cap_id:
                raise ValueError("Capabilities do not match the source device")
            expected_sensor = {"quantity": "temperature", **{key: value for key, value in expected.items() if key != "device_id"}}
            if result["sensors"] != [expected_sensor]:
                raise ValueError("Capability values changed in the tool trace")
        elif result.get("status") == "ok" and tool == "read_measurement":
            if cap_id is None or args != {"device_id": case_id, "sensor_id": source["sensor_id"]}:
                raise ValueError("Measurement lacks matching preceding capabilities")
            expected = {key: source[key] for key in ("device_id", "sensor_id", "value", "unit", "sequence", "observed_unix_seconds")}
            if any(result[key] != value for key, value in expected.items()) or result["observation_sha256"] != sha(raw):
                raise ValueError("Measurement does not match the saved adapter input")
            moment = result["observed_unix_seconds"] + result["age_seconds"]
            if not math.isfinite(moment) or not created - 1 <= moment <= completed + 1 or result["age_seconds"] < 0:
                raise ValueError("Measurement timestamp is outside the observed run")
            without_id = {key: value for key, value in result.items() if key != "evidence_id"}
            if result["evidence_id"] != sha(canonical(without_id)):
                raise ValueError("Measurement evidence ID does not bind its observed data")
            reading = result
        elif tool == "submit_assessment" and result.get("status") == "recorded":
            if submission is not None or reading is None or set(args["evidence_ids"]) != {cap_id, reading["evidence_id"]}:
                raise ValueError("Assessment does not identify both issued observations")
            if result["independently_verified"] is not False:
                raise ValueError("Runtime improperly claimed independent correctness")
            submission = args
    if record["submission"] != submission:
        raise ValueError("Saved submission differs from its recorded tool call")
    passed = False
    expected_status, expected_temperature = None, None
    if submission is not None:
        if reading["age_seconds"] > source["maximum_observation_age_seconds"]:
            expected_status, expected_temperature = "unknown", None
        else:
            value = source["value"]
            expected_temperature = value if source["unit"] == "C" else (
                (value - 32) * 5 / 9 if source["unit"] == "F" else value - 273.15)
            lower, upper = source["normal_range_celsius"]
            expected_status = "low" if expected_temperature < lower else ("high" if expected_temperature > upper else "normal")
        actual = submission["temperature_celsius"]
        temperature_ok = (actual is None if expected_temperature is None else
                          type(actual) in (int, float) and abs(actual - expected_temperature) <= 0.05)
        passed = record["status"] == "finished" and temperature_ok and submission["status"] == expected_status
    if bool(passed) != case["evaluation"]["passed"] or bool(passed) != item["passed"]:
        raise ValueError("Independent task evaluation disagrees with the recorded result")
    streams = trace.get("model_streams")
    if streams is not None and len(streams) != len(trace["model_calls"]):
        raise ValueError("Literal stream count differs from actual model calls")
    arguments = streamed_arguments(streams or [])
    if streams is not None:
        match_streamed_tools(arguments, trace)
    return {"case_id": case_id, "trace_consistent": True, "task_passed": bool(passed),
            "expected_status": expected_status, "expected_celsius": expected_temperature,
            "literal_streams_present": streams is not None,
            "streamed_tool_arguments": arguments, "malformed_streamed_tool_calls": sum(not item["valid_json_object"] for item in arguments)}


def audit(directory):
    directory = Path(directory).resolve()
    run = json.loads((directory / "report.json").read_bytes())
    if run["failure"] or run.get("shutdown_failure"):
        raise ValueError("Application run has an unresolved execution failure")
    if (run["anonymous_inference_request_status"] != 401 or run["final_health"]["status"] != "serving"
            or not run["shutdown"]["owned_server_stopped"] or not run["socket_removed"]):
        raise ValueError("Application service or shutdown evidence is incomplete")
    cases = [inspect_case(directory, item, run) for item in run["cases"]]
    if not cases or len(cases) != run["cases_total"] or sum(case["task_passed"] for case in cases) != run["cases_passed"]:
        raise ValueError("Application case count or success total differs")
    return {"scope": "Independent local artifact readback, not native or population qualification",
            "trace_consistency_passed": True, "task_successes": sum(case["task_passed"] for case in cases),
            "task_trials": len(cases), "all_literal_streams_present": all(case["literal_streams_present"] for case in cases),
            "cases": cases, "qualification_passed": False,
            "auditor_sha256": sha(Path(__file__).read_bytes())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    with (args.directory / "audit.json").open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({"trace_consistency_passed": result["trace_consistency_passed"],
                      "task_successes": result["task_successes"], "task_trials": result["task_trials"],
                      "literal_streams_present": result["all_literal_streams_present"]}))


if __name__ == "__main__":
    main()
