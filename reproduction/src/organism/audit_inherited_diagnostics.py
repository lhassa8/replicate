"""Read back inherited diagnostics against separately supplied signed records.

The caller supplies the precommitted record inventory and trusted verification
context. Neither a model trace nor a note supplies its own trust configuration.
This evaluator is excluded from the deployed application.
"""

from copy import deepcopy
from datetime import datetime
import json

from .audit_application import match_streamed_tools, streamed_arguments
from .audit_equipment_diagnostics import digest, encoded, inspect_job, numeric, require
from .experience import VerifiedExperience


TOOL_NAME = "prepare_reader_from_experience"


def inspect_inherited_job(case, raw, job, *, policy, release_id, allowed_checkers, records,
                          require_live_source=True, live=False, clock_window=None):
    record, trace = job["record"], job["record"]["trace"]
    require(digest(encoded(record)) == job["digest"], "Saved inherited job digest differs")
    inheritance = trace.get("inheritance")
    require(isinstance(inheritance, dict) and set(inheritance) == {"schema", "attempts"}
            and inheritance["schema"] == "organism-diagnostic-inheritance-v1", "Missing inheritance trace")
    attempts, events = inheritance["attempts"], trace["tool_events"]
    require(isinstance(attempts, list) and len(attempts) <= 8 and len(events) <= 8
            and isinstance(records, dict) and len(records) <= 128, "Inheritance trace or record inventory exceeds bounds")
    document = json.loads(raw)
    capabilities = {key: document[key] for key in ("device_id", "family", "task")}
    capabilities["sources"] = [{key: value for key, value in source.items() if key != "payload"}
                              for source in document["sources"]]
    capability_hash = digest(encoded({key: value for key, value in capabilities.items() if key != "device_id"}))
    validated_records = {}
    for identity, signed in records.items():
        created = json.loads(signed)["body"]["created_unix_seconds"]
        verified = VerifiedExperience(signed, policy, release_id=release_id, allowed_checkers=allowed_checkers,
                                      now=created, require_live_source=require_live_source)
        require(verified.note_id == identity, "Independent record inventory identity differs")
        validated_records[identity] = verified
    by_index = {}
    for attempt in attempts:
        index = attempt["tool_event_index"]
        require(type(index) is int and 0 <= index < len(events) and index not in by_index,
                "Unfinished or duplicate inheritance attempt requires separate failure investigation")
        by_index[index] = attempt
    projected, uses, elapsed, operation_count = [], [], 0.0, 0
    for index, event in enumerate(events):
        if event["tool"] != TOOL_NAME:
            require(index not in by_index, "Inheritance attempt is attached to another tool")
            projected.append(deepcopy(event))
            operation_count += 1
            continue
        status, attempt = event["result"]["status"], by_index.get(index)
        if status in ("invalid_observation_or_arguments", "budget_exhausted"):
            require(attempt is None or attempt["status"] == "failed", "Rejected lookup claims successful preparation")
            if attempt is not None:
                steps = attempt["operations"]
                require([step["tool"] for step in steps] in (
                    [], ["inspect_equipment"], ["inspect_equipment", "configure_reader"]),
                    "Rejected lookup hid unexpected reader operations")
                partial = deepcopy(record)
                partial["status"], partial["submission"] = "failed", None
                partial["trace"]["tool_events"] = steps
                inspect_job(case, raw, {"record": partial, "digest": digest(encoded(partial))})
                require(numeric(attempt["elapsed_seconds"]) and attempt["elapsed_seconds"] >= 0,
                        "Rejected inheritance lookup omitted its effort")
                elapsed += attempt["elapsed_seconds"]
                operation_count += max(1, len(steps))
            else:
                operation_count += 1
            projected.append(deepcopy(event))
            continue
        require(attempt is not None and attempt["status"] == status and event["arguments"] == {"device_id": case["case_id"]}
                and status in ("prepared_from_experience", "no_applicable_experience", "conflicting_experience"),
                "Unexpected inherited-reader outcome")
        require(numeric(attempt["elapsed_seconds"]) and attempt["elapsed_seconds"] >= 0
                and numeric(attempt["started_unix_seconds"]), "Missing charged inheritance timing")
        elapsed += attempt["elapsed_seconds"]
        moment = attempt["validation_unix_seconds"]
        require(type(moment) is int and 0 <= moment <= 4000000000, "Invalid inheritance verification clock")
        available = sorted(identity for identity, note in validated_records.items()
                           if note.body["created_unix_seconds"] <= moment < note.body["expires_unix_seconds"])
        matching = sorted(identity for identity in available
                          if validated_records[identity].body["capability_sha256"] == capability_hash)
        require(attempt["available_note_ids"] == available and attempt["matching_note_ids"] == matching,
                "Recorded lookup differs from the independent source inventory or current capabilities")
        readers = {encoded(validated_records[identity].body["reader"]) for identity in matching}
        expected_status = ("no_applicable_experience" if not matching else
                           "conflicting_experience" if len(readers) != 1 else "prepared_from_experience")
        selected = min(matching) if expected_status == "prepared_from_experience" else None
        require(status == expected_status and attempt["selected_note_id"] == selected, "Inherited procedure selection differs")
        steps = attempt["operations"]
        require([step["tool"] for step in steps] == (
            ["inspect_equipment", "configure_reader"] if selected else ["inspect_equipment"]),
            "Inherited tool includes missing or unexpected internal operations")
        require(event["result"] == {"status": status, "note_id": selected, "capabilities": steps[0]["result"],
                "reader": steps[1]["result"] if selected else None,
                "equipment_modified": False, "diagnosis_supplied": False},
                "Model-facing inherited result differs from its recorded reader operations")
        if selected:
            note = validated_records[selected]
            when = attempt["application_unix_seconds"]
            require(type(when) is int and note.body["created_unix_seconds"] <= when < note.body["expires_unix_seconds"],
                    "Inherited reader was applied outside its lifetime")
            require(steps[1]["arguments"] == {"device_id": case["case_id"], **note.body["reader"]},
                    "Configured reader differs from the endorsed procedure")
            uses.append(selected)
        else:
            require(attempt["application_unix_seconds"] is None, "Unselected procedure was applied")
        projected.extend(deepcopy(steps))
        operation_count += len(steps)
    require(operation_count <= 8, "Combined reader tool hid excess operations")
    projection = deepcopy(record)
    projection["trace"]["tool_events"] = projected
    semantic = inspect_job(case, raw, {"record": projection, "digest": digest(encoded(projection))})
    if live:
        require(clock_window is not None and len(clock_window) == 2, "Live inherited trace requires an observed clock window")
        calls, requests = trace["model_calls"], trace["http_requests"]
        require(0 < len(calls) == len(requests) == len(trace["model_responses"]) == len(trace["model_streams"]) <= 6,
                "Live inherited model evidence is incomplete")
        require(numeric(trace.get("seconds")) and trace["seconds"] + 0.002 >= elapsed,
                "Inheritance validation/application time was omitted from case effort")
        for attempt in attempts:
            if attempt["status"] == "failed":
                continue
            for key in ("started_unix_seconds", "validation_unix_seconds", "application_unix_seconds"):
                moment = attempt[key]
                require(moment is None or clock_window[0] - 1 <= moment <= clock_window[1],
                        "Inheritance operation clock is outside the observed case")
        for event in projected:
            if event["tool"] == "read_equipment" and event["result"]["status"] == "observed":
                moment = event["result"]["read_unix_seconds"]
                require(clock_window[0] <= moment <= clock_window[1]
                        and moment >= datetime.fromisoformat(record["created_at"]).timestamp() - 1,
                        "Current measurement clock is outside the observed case")
        for request in requests:
            require(request["method"] == "POST" and request["path"] == "/v1/chat/completions"
                    and request["body"]["max_tokens"] == 192 and request["body"]["temperature"] == 0,
                    "Inherited inference route or generation limits differ")
        match_streamed_tools(streamed_arguments(trace["model_streams"]), trace)
        controls, responses = trace.get("model_controls", []), trace.get("control_responses", [])
        if controls:
            reason = "Equipment diagnostic model-call allowance exhausted"
            require(controls == [{"reason": "model_call_limit", "allowed_calls": 6}] and len(calls) == 6
                    and record["status"] == "failed" and record["failure"] == reason
                    and responses == [{"source": "framework_cancellation", "stop_reason": "end_turn", "exception": None,
                                       "message": {"role": "assistant", "content": [{"text": reason}]}}],
                    "Unexplained inherited model-control event")
        else:
            require(not responses, "Unexplained inherited framework response")
        messages = trace["final_messages"]
        last = messages[-1] if messages else {}
        acknowledged = (last.get("role") == "assistant" and not any("toolUse" in block for block in last.get("content", []))
                        and any(block.get("text", "").strip() for block in last.get("content", [])))
        semantic["semantic_passed"] = bool(semantic["semantic_passed"] and acknowledged)
    return {**semantic, "live_model_trace_verified": live, "tool_calls": len(events),
            "reader_operations": operation_count, "inherited_reader_uses": uses,
            "inheritance_lookup_seconds": round(elapsed, 6), "independent_record_inventory_verified": True,
            "qualification_passed": False}
