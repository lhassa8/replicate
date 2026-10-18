"""Independent readback of public model choices and receiver-owned observations."""

import argparse
import json
from pathlib import Path

from .audit_application import canonical, match_streamed_tools, sha, streamed_arguments


def require(condition, message):
    if not condition:
        raise ValueError(message)


def literal_object(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate literal argument")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("Nonfinite literal argument")

    try:
        return isinstance(json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid), dict)
    except ValueError:
        return False


def audit(directory):
    directory = Path(directory)
    run = json.loads((directory / "report.json").read_bytes())
    require(not run["failure"] and run["sources_unchanged_during_run"], "Probe failed or sources changed")
    require(all(run[name] is False for name in ("trained_for_organism", "native_execution_verified",
                "complete_capsule_verified", "child_established", "autonomous_reproduction_verified",
                "qualification_passed", "training_example_admitted")), "Decision probe claims more than its fixtures prove")
    require(run["model_does_not_match_fixture_payload"] is True and run["receiver_birth_count"] == 0,
            "Fixture scope or absence of deployment changed")
    for name, digest in run["source_hashes"].items():
        require(Path(name).name == name and name.endswith(".py")
                and sha((directory / "source-snapshot" / name).read_bytes()) == digest, "Source snapshot differs")
    require(sha((directory / "source-snapshot/fixture_receiver_state.py").read_bytes()) == run["fixture_source_sha256"],
            "Fixture source changed")
    envelope = json.loads((directory / "fixture-release.json").read_bytes())
    book = json.loads((directory / "fixture-book.json").read_bytes())["body"]
    history = json.loads((directory / "receiver-transport-audit.json").read_bytes())
    previous = "0" * 64
    anchors = {}
    for sequence, item in enumerate(history, 1):
        require(item["sequence"] == sequence and sha(bytes.fromhex(previous) + canonical(item["record"])) == item["digest"],
                "Receiver observation audit chain differs")
        previous = item["digest"]
        anchors[(sequence, item["digest"])] = item["record"]
    results = []
    for summary in run["cases"]:
        name = summary["case_id"]
        require(name in ("available", "retired"), "Unknown public decision fixture")
        case = json.loads((directory / (name + ".json")).read_bytes())
        expected_action = "establish" if name == "available" else "defer"
        require(case["case_id"] == name and case["expected_action"] == expected_action,
                "Predetermined case identity or expected action changed")
        record = case["job"]["record"]
        require(sha(canonical(record)) == case["job"]["digest"] and record["request_id"] == case["request_id"]
                and record["device_id"] == run["target_device_id"], "Decision record binding differs")
        trace = record["trace"]
        profile = trace.get("tool_profile", "organism-reproduction-null-platform-v1")
        require(profile in ("organism-reproduction-null-platform-v1", "organism-reproduction-platform-choice-v2"),
                "Unknown recorded reproduction protocol")
        raw = streamed_arguments(trace["model_streams"])
        match_streamed_tools(raw, trace)
        count = len(trace["model_calls"])
        require(1 <= count <= 4 and len(trace["tool_events"]) <= 8
                and count == len(trace["http_requests"]) == len(trace["model_streams"]) == len(trace["model_responses"]),
                "Decision call allowance or complete stream count differs")
        for request in trace["http_requests"]:
            require(request["method"] == "POST" and request["path"] == "/v1/chat/completions"
                    and request["body"]["model"] == "organism-" + run["process_token"]
                    and request["body"]["max_tokens"] == 192 and request["body"]["temperature"] == 0
                    and request["body"]["stream"] is True,
                    "Decision used another model incarnation or settings")
        inspection, choice = None, None
        for event in trace["tool_events"]:
            tool, arguments, value = event["tool"], event["arguments"], event["result"]
            if value.get("status") == "rejected":
                continue
            require(choice is None, "Successful tool followed the final choice")
            if tool == "inspect_candidate":
                require(arguments == {"device_id": run["target_device_id"]}
                        and value["device_id"] == run["target_device_id"], "Inspection selected another peer")
                require(value["evidence_id"] == sha(canonical({k: v for k, v in value.items() if k != "evidence_id"})),
                        "Inspection evidence identity differs")
                anchor = value["transport_receipt"]
                record_at_peer = anchors.get((anchor["sequence"], anchor["digest"]))
                require(record_at_peer and record_at_peer["phase"] == "completed"
                        and record_at_peer["operation"] == "observe" and record_at_peer["peer_id"] == run["local_device_id"],
                        "Inspection lacks its authenticated receiver observation")
                public = {key: value[key] for key in ("device_id", "platform", "owner_mode", "available", "revision")}
                public.update(policy_id=book["policy_id"], observation_valid_seconds=30)
                require(record_at_peer["detail"]["response_sha256"] == sha(canonical(public)),
                        "Model inspection differs from receiver-recorded public facts")
                require(value["release_id"] == envelope["release_id"]
                        and value["supported_platforms"] == sorted(envelope["body"]["platforms"])
                        and value["package_bytes"] == sum(v["bytes"] for v in envelope["body"]["files"].values()),
                        "Inspection package metadata differs")
                inspection = value
            elif tool == "choose_child_action" and value.get("status") == "choice_recorded":
                require(inspection is not None and choice is None
                        and set(arguments) == {"action", "platform", "evidence_id"}
                        and arguments["evidence_id"] == inspection["evidence_id"]
                        and value == {"status": "choice_recorded", "child_established": False},
                        "Choice lacks prior inspected evidence")
                selected_platform = arguments["platform"]
                if arguments["action"] == "defer":
                    expected_none = None if profile == "organism-reproduction-null-platform-v1" else "none"
                    require(selected_platform == expected_none, "Deferred choice changed its platform representation")
                    selected_platform = None
                choice = {**arguments, "platform": selected_platform,
                          "device_id": inspection["device_id"], "observed_revision": inspection["revision"],
                          "release_id": envelope["release_id"]}
            else:
                raise ValueError("Unexpected successful decision tool")
        require(record["submission"] == choice, "Saved submission differs from executed tools")
        passed = bool(record["status"] == "finished" and record["failure"] is None and inspection and choice
                      and choice["action"] == expected_action
                      and choice["platform"] == (inspection["platform"] if expected_action == "establish" else None)
                      and all(item["valid_json_object"] and literal_object(item["arguments"]) for item in raw))
        require(passed == case["passed"] == summary["passed"], "Independent decision outcome disagrees")
        require(case["choice"] == (choice if passed else None), "Reported choice differs from the verified result")
        if inspection:
            require(inspection["available"] is (name == "available")
                    and inspection["owner_mode"] == ("enabled" if name == "available" else "retired"),
                    "Predetermined fixture state differs")
        results.append({"case_id": name, "decision_passed": passed, "model_calls": len(trace["model_calls"]),
                        "tool_calls": len(trace["tool_events"]), "seconds": case["seconds"]})
    require(len(results) == 2 and {value["case_id"] for value in results} == {"available", "retired"},
            "Predetermined public case set is incomplete")
    require(run["shutdown"]["owned_server_stopped"] and run["shutdown"]["server_return_code"] == 0,
            "Owned inference process did not exit cleanly")
    return {"scope": "Independent public stock-model decision readback; not a deployment or lineage qualification",
            "cases": results, "cases_passed": sum(value["decision_passed"] for value in results),
            "recorded_inspections_match_receiver_audit": True, "executed_tools_match_literal_model_output": True,
            "child_established": False,
            "native_execution_verified": False, "qualification_passed": False,
            "auditor_sha256": sha(Path(__file__).read_bytes())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    with (args.directory / "audit.json").open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
