"""Independent development readback; no runtime reader or baseline is imported."""

from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math

from .audit_application import match_streamed_tools, streamed_arguments


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(source, selector):
    raw = source["payload"]
    if source["format"] == "register_map":
        bank, address = selector.split(":")
        path = [bank, str(int(address))]
    else:
        path = [] if selector == "" else [part.replace("~1", "/").replace("~0", "~")
                                          for part in selector[1:].split("/")]
    try:
        for item in path:
            raw = raw[int(item)] if isinstance(raw, list) else raw[item]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return raw


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def checked_temperature(task, raw_reading):
    """Recompute normalization from saved evidence, without the runtime helper."""
    reasons = []
    if raw_reading["validity"] is not True:
        reasons.append("invalid")
    if not raw_reading["timestamp_usable"]:
        reasons.append("unusable_timestamp")
    elif raw_reading["age_seconds"] > task["maximum_age_seconds"]:
        reasons.append("stale")
    fields = raw_reading["values"]
    if (len(fields) != 1 or fields[0]["kind"] != "number"
            or fields[0]["unit"] not in ("C", "F", "K")):
        reasons.append("unsupported_measurement")
    elif not fields[0]["available"]:
        reasons.append("missing")
    value = None
    if not reasons:
        field = fields[0]
        scaled = Fraction(str(field["value"])) * Fraction(str(field["scale"]))
        if field["unit"] == "F":
            scaled = (scaled - 32) * Fraction(5, 9)
        elif field["unit"] == "K":
            scaled -= Fraction(27315, 100)
        value = float(scaled)
    return {
        "schema": "organism-temperature-measurement-v1", "usable": not reasons,
        "reason_codes": reasons, "value": value, "unit": "C",
        "normal_range": list(task["normal_range"]), "bounds_inclusive": True,
    }


def check_value_feedback(result, args, reading, submission, identity):
    """Reconstruct value feedback from the independently verified reading."""
    require(submission is None and reading is not None
            and "temperature_measurement" in reading,
            "Value feedback lacks an unsubmitted canonical reading")
    require(set(args) == {"device_id", "status", "value", "unit", "evidence_id"}
            and args["device_id"] == identity and args["evidence_id"] == reading["evidence_id"]
            and args["status"] in ("normal", "low", "high", "unknown")
            and args["unit"] in ("C", "mm/s", "percent", "count")
            and (args["value"] is None or numeric(args["value"])),
            "Value feedback follows invalid arguments or invented evidence")
    measured = reading["temperature_measurement"]
    require(args["value"] != measured["value"], "Consistent value was reported as rejected")
    expected = {
        "status": "invalid_observation_or_arguments", "error_type": "TemperatureValueError",
        "measurement_value_feedback": {
            "schema": "organism-measurement-value-feedback-v1",
            "reading_evidence_id": reading["evidence_id"],
            "canonical_value": measured["value"], "canonical_unit": measured["unit"],
            "usable": measured["usable"],
            "instruction": "Copy canonical_value exactly without rounding and submit again. Choose the diagnosis status from the reading. No diagnosis was recorded.",
        },
    }
    require(result == expected, "Value feedback differs from the issued canonical reading")


def inspect_job(case, raw, job, *, live=False, clock_window=None):
    record = job["record"]
    require(digest(encoded(record)) == job["digest"], "Saved diagnostic job digest differs")
    document = json.loads(raw)
    identity = case["case_id"]
    require(document == case["document"] and document["device_id"] == record["device_id"] == identity,
            "Observed device input differs")
    require(record.get("task_kind") == "equipment_interface_v1", "Wrong diagnostic interface")
    trace = record["trace"]
    profile = trace.get("tool_profile", "organism-raw-equipment-tools-v1")
    require(profile in ("organism-raw-equipment-tools-v1", "organism-canonical-temperature-tools-v1"),
            "Unknown diagnostic tool profile")
    events = trace.get("tool_events", [])
    require(len(events) <= 8 and len(trace.get("model_calls", [])) <= 6, "Diagnostic call limits exceeded")
    expected_cap = {key: document[key] for key in ("device_id", "family", "task")}
    expected_cap["sources"] = [{key: value for key, value in source.items() if key != "payload"}
                               for source in document["sources"]]
    cap_id, configuration, config_id, reading, submission = None, None, None, None, None
    for event in events:
        tool, args, result = event["tool"], event["arguments"], event["result"]
        status = result.get("status")
        if tool == "inspect_equipment" and status == "ok":
            cap_id = digest(encoded(expected_cap))
            require(args == {"device_id": identity}
                    and result == {"status": "ok", "descriptor": expected_cap, "evidence_id": cap_id},
                    "Capabilities differ from the saved device input")
            configuration, config_id, reading = None, None, None
        elif tool == "configure_reader" and status == "configured":
            require(cap_id is not None and args["device_id"] == identity, "Configuration precedes capabilities")
            configuration = {**args, "capability_evidence_id": cap_id}
            config_id = digest(encoded(configuration))
            require(result == {"status": "configured", "configuration": configuration,
                               "evidence_id": config_id, "equipment_modified": False},
                    "Reader configuration or evidence binding differs")
            reading = None
        elif tool == "read_equipment" and status == "observed":
            require(configuration is not None and args == {"device_id": identity}, "Read precedes configuration")
            source = next(item for item in document["sources"] if item["source_id"] == configuration["source_id"])
            fields = {item["selector"]: item for item in source["fields"]}
            values = []
            for selector in configuration["value_selectors"]:
                field = fields[selector]
                value = resolve(source, selector)
                available = (numeric(value) if field["kind"] == "number" else isinstance(value, list)
                             and len(value) <= 16 and all(isinstance(item, dict) for item in value))
                values.append({**field, "value": value if available else None, "available": available})
            moment = result["read_unix_seconds"]
            require(numeric(moment) and moment >= 0, "Invalid recorded read time")
            if live:
                require(clock_window is not None and clock_window[0] <= moment <= clock_window[1],
                        "Read time is outside the observed run")
                require(moment >= datetime.fromisoformat(record["created_at"]).timestamp() - 1,
                        "Read precedes request admission")
            observed = resolve(source, configuration["timestamp_selector"])
            observed = observed if numeric(observed) else None
            usable = observed is not None and 0 <= observed <= moment
            validity = True
            if configuration["validity_selector"] is not None:
                validity = resolve(source, configuration["validity_selector"])
                validity = validity if type(validity) is bool else None
            expected = {
                "status": "observed", "device_id": identity, "source_id": source["source_id"],
                "values": values, "observed_unix_seconds": observed, "read_unix_seconds": moment,
                "age_seconds": moment - observed if usable else None, "timestamp_usable": usable,
                "validity": validity, "sequence": document["sequence"], "observation_sha256": digest(raw),
                "configuration_evidence_id": config_id,
            }
            if (profile == "organism-canonical-temperature-tools-v1"
                    and document["task"]["quantity"] == "temperature"):
                expected["temperature_measurement"] = checked_temperature(document["task"], expected)
            expected["evidence_id"] = digest(encoded(expected))
            require(result == expected, "Raw readback or chained observation evidence differs")
            reading = result
        elif tool == "submit_diagnosis" and status == "recorded":
            require(submission is None and reading is not None and args["device_id"] == identity
                    and args["evidence_id"] == reading["evidence_id"],
                    "Submission lacks its issued reading chain")
            require(result == {"status": "recorded", "independently_verified": False},
                    "Runtime improperly claimed independent correctness")
            submission = args
        else:
            require(status in ("invalid_observation_or_arguments", "budget_exhausted"),
                    "Unknown tool outcome")
            if ("measurement_value_feedback" in result
                    or result.get("error_type") == "TemperatureValueError"):
                require(tool == "submit_diagnosis"
                        and profile == "organism-canonical-temperature-tools-v1"
                        and document["task"]["quantity"] == "temperature",
                        "Value feedback is attached to the wrong tool or profile")
                check_value_feedback(result, args, reading, submission, identity)
    require(record["submission"] == submission, "Saved submission differs from the executed tool")
    expected_value, expected_status = case["expected_value"], case["expected_status"]
    correct_reader = configuration is not None and {
        key: value for key, value in configuration.items() if key != "capability_evidence_id"
    } == case["expected_reader"]
    if reading is not None and (not reading["timestamp_usable"]
                                or reading["age_seconds"] > document["task"]["maximum_age_seconds"]):
        expected_status, expected_value = "unknown", None
    value_ok = submission is not None and (
        submission["value"] is None if expected_value is None else
        numeric(submission["value"]) and abs(submission["value"] - expected_value) <= 0.0001)
    passed = bool(submission is not None and correct_reader and value_ok and record["status"] == "finished"
                  and submission["status"] == expected_status and submission["unit"] == case["expected_unit"])
    if live:
        calls, requests = trace["model_calls"], trace["http_requests"]
        require(len(calls) == len(requests) == len(trace["model_streams"]) == len(trace["model_responses"]),
                "Model call, request, response or stream counts differ")
        controls, control_responses = trace.get("model_controls", []), trace.get("control_responses", [])
        if controls:
            reason = "Equipment diagnostic model-call allowance exhausted"
            require(controls == [{"reason": "model_call_limit", "allowed_calls": 6}] and len(calls) == 6
                    and len(control_responses) == 1 and record["status"] == "failed"
                    and record["failure"] == reason, "Unexplained model-control event or successful budget overrun")
            require(control_responses[0] == {
                "source": "framework_cancellation", "stop_reason": "end_turn", "exception": None,
                "message": {"role": "assistant", "content": [{"text": reason}]}},
                "Framework cancellation differs from its recorded control event")
        else:
            require(not control_responses, "Unexplained framework response")
        for request in requests:
            require(request["method"] == "POST" and request["path"] == "/v1/chat/completions"
                    and request["body"]["max_tokens"] == 192 and request["body"]["temperature"] == 0,
                    "Observed inference route or generation limits differ")
        match_streamed_tools(streamed_arguments(trace["model_streams"]), trace)
        messages = trace["final_messages"]
        last = messages[-1] if messages else {}
        acknowledged = (last.get("role") == "assistant"
                        and not any("toolUse" in block for block in last.get("content", []))
                        and any(block.get("text", "").strip() for block in last.get("content", [])))
        passed = bool(passed and acknowledged)
    return {
        "semantic_passed": passed, "correct_reader_selected": correct_reader,
        "expected_status": expected_status, "expected_value": expected_value,
        "expected_unit": case["expected_unit"], "observation_and_evidence_verified": True,
        "live_model_trace_verified": live, "model_calls": len(trace.get("model_calls", [])),
        "tool_calls": len(events), "training_admitted": False, "qualification_passed": False,
    }
