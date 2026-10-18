"""Check recorded reproduction choices before permitting any transfer."""

import hashlib
import json

from .capsule import canonical
from .reproduction_journal import ReproductionError


def require(condition, message):
    if not condition:
        raise ReproductionError(message)


def decode(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate literal reproduction argument")
            value[key] = item
        return value

    def invalid(_):
        raise ReproductionError("Nonfinite literal reproduction argument")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def verify_choice(job, *, peer_id, start_token, envelope):
    record = job["record"]
    require(hashlib.sha256(canonical(record)).hexdigest() == job["digest"]
            and record["status"] == "finished" and record["failure"] is None
            and record["device_id"] == peer_id and record["independently_verified"] is False,
            "Reproduction decision is not a matching finished job")
    trace = record["trace"]
    profile = trace.get("tool_profile", "organism-reproduction-null-platform-v1")
    require(profile in ("organism-reproduction-null-platform-v1", "organism-reproduction-platform-choice-v2"),
            "Unknown recorded reproduction protocol")
    count = len(trace["model_calls"])
    require(1 <= count <= 4 and len(trace["model_streams"]) == count
            and len(trace["model_responses"]) == count and len(trace["http_requests"]) == count
            and 2 <= len(trace["tool_events"]) <= 8, "Reproduction model trace is incomplete or exceeds its bound")
    literal = []
    for index, events in enumerate(trace["model_streams"], 1):
        active, blocks = None, []
        for event in events:
            if "contentBlockStart" in event:
                block = event["contentBlockStart"]
                require("contentBlockIndex" not in block, "Reproduction requires the pinned serial provider")
                active = block.get("start", {}).get("toolUse")
                if active:
                    active = {**active, "raw": "", "call": index}
                    blocks.append(active)
            if "toolUse" in event.get("contentBlockDelta", {}).get("delta", {}):
                require(active is not None, "Literal reproduction fragment has no tool start")
                active["raw"] += event["contentBlockDelta"]["delta"]["toolUse"]["input"]
            if "contentBlockStop" in event:
                active = None
        for block in blocks:
            arguments = decode(block["raw"])
            require(isinstance(arguments, dict), "Literal reproduction arguments are not an object")
            literal.append({"name": block["name"], "toolUseId": block["toolUseId"],
                            "input": arguments, "call": index})
    parsed = [{"call": index, **part["toolUse"]} for index, response in enumerate(trace["model_responses"], 1)
              for part in (response.get("message") or {}).get("content", []) if "toolUse" in part]
    require(literal == parsed and len(parsed) == len(trace["tool_events"]),
            "Literal reproduction choices differ from parsed or executed tools")
    for request in trace["http_requests"]:
        body = request["body"]
        require(request["method"] == "POST" and request["path"] == "/v1/chat/completions"
                and body["model"] == "organism-" + start_token and body["max_tokens"] == 192
                and body["temperature"] == 0 and body["stream"] is True,
                "Reproduction reasoning used another model incarnation or generation settings")
        if profile == "organism-reproduction-platform-choice-v2":
            choices = [item["function"] for item in body.get("tools", [])
                       if item.get("type") == "function" and item["function"].get("name") == "choose_child_action"]
            require(len(choices) == 1, "Versioned choice lacks its declared tool")
            schema = choices[0]["parameters"]
            require(schema["properties"]["platform"]["type"] == "string"
                    and schema["properties"]["platform"]["enum"] == ["linux-x86_64", "linux-arm64", "none"]
                    and "platform" in schema["required"], "Recorded choice schema does not match its protocol")
    observed, choice = None, None
    for parsed_tool, event in zip(parsed, trace["tool_events"]):
        name, arguments, result = event["tool"], event["arguments"], event["result"]
        require(parsed_tool["name"] == name and parsed_tool["input"] == arguments,
                "Executed reproduction arguments differ from literal model output")
        if result.get("status") == "rejected":
            continue
        require(choice is None, "Additional successful action followed a completed choice")
        if name == "inspect_candidate":
            fields = {"device_id", "platform", "owner_mode", "available", "revision", "supported_platforms",
                      "release_id", "package_bytes", "transport_receipt", "evidence_id"}
            require(set(result) == fields and arguments == {"device_id": peer_id}
                    and result["device_id"] == peer_id and result["release_id"] == envelope["release_id"]
                    and result["supported_platforms"] == sorted(envelope["body"]["platforms"])
                    and result["package_bytes"] == sum(value["bytes"] for value in envelope["body"]["files"].values())
                    and type(result["available"]) is bool
                    and result["evidence_id"] == hashlib.sha256(canonical(
                        {key: value for key, value in result.items() if key != "evidence_id"})).hexdigest(),
                    "Reproduction inspection differs from the carried release or evidence identity")
            observed = result
        elif name == "choose_child_action":
            require(observed is not None and set(arguments) == {"action", "platform", "evidence_id"}
                    and arguments["evidence_id"] == observed["evidence_id"]
                    and result == {"status": "choice_recorded", "child_established": False},
                    "Reproduction choice lacks its inspected evidence")
            if arguments["action"] == "establish":
                require(observed["owner_mode"] == "enabled" and observed["available"]
                        and arguments["platform"] == observed["platform"]
                        and arguments["platform"] in observed["supported_platforms"],
                        "Reproduction choice is not admitted or supported")
            else:
                none = None if profile == "organism-reproduction-null-platform-v1" else "none"
                require(arguments["action"] == "defer" and arguments["platform"] == none,
                        "Unexpected reproduction action")
            choice = {**arguments, "platform": arguments["platform"] if arguments["action"] == "establish" else None,
                      "device_id": peer_id, "observed_revision": observed["revision"],
                      "release_id": envelope["release_id"]}
        else:
            raise ReproductionError("Unknown successful reproduction tool")
    require(choice is not None and record["submission"] == choice, "Saved reproduction choice differs from its trace")
    return choice
