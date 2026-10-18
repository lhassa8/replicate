"""Reconstruct cross-device discovery, local model choice, and complete transfer."""

import json

from ..audit_application import canonical, match_streamed_tools, sha, streamed_arguments
from .audit_startup import moment, require


def chain(evidence, prefix):
    require(evidence.read(prefix + "peer-audit.enrolled", 128) == b"organism-peer-audit-v1\n",
            "Peer evidence lacks its original enrollment marker")
    name = prefix + "peer-audit.sqlite3"
    rows = evidence.rows(name,
                         "SELECT sequence,previous,digest,substr(body,1,2049) FROM records "
                         "ORDER BY sequence LIMIT 65537", maximum=192 * 1024**2)
    head = evidence.rows(name, "SELECT sequence,digest FROM head LIMIT 2", maximum=192 * 1024**2)
    require(len(rows) <= 65536 and len(head) == 1, "Peer audit exceeds its original allowance")
    previous, result = "0" * 64, []
    for index, (sequence, ancestor, digest, text) in enumerate(rows, 1):
        require(sequence == index and ancestor == previous and len(text.encode()) <= 2048
                and sha(bytes.fromhex(previous) + text.encode()) == digest,
                "Peer history is incomplete, reordered or changed")
        result.append({"sequence": sequence, "digest": digest, "record": json.loads(text)})
        previous = digest
    require(head[0] == (len(rows), previous), "Peer history head differs")
    return result


def anchored(history, anchor):
    require(isinstance(anchor, dict) and set(anchor) == {"sequence", "digest"}
            and type(anchor["sequence"]) is int and 1 <= anchor["sequence"] <= len(history),
            "Peer receipt points outside its actual history")
    row = history[anchor["sequence"] - 1]
    require(row["sequence"] == anchor["sequence"] and row["digest"] == anchor["digest"],
            "Peer receipt hash differs from its recorded history")
    return row["record"]


def transfer(history, ready, catalog, parent):
    birth, payloads = ready["birth"], {}
    completed, failures = [], []
    for item in history:
        record = item["record"]
        if record["operation_id"] != birth["operation_id"]:
            continue
        if record["phase"] in ("failed", "rejected", "response_lost"):
            failures.append(item)
        if record["phase"] != "completed":
            continue
        require(record["peer_id"] == parent, "Another identity contributed to this birth")
        completed.append(record)
        if record["operation"] == "put":
            detail = record["detail"]
            identity = {"bytes": detail["bytes"], "sha256": detail["sha256"]}
            require(detail["payload"] in catalog["files"]
                    and identity == catalog["files"][detail["payload"]],
                    "Received component differs from the complete signed package")
            payloads[detail["payload"]] = identity
    require(payloads == catalog["files"], "This child did not receive every package component from its parent")
    require(any(row["operation"] == "reserve" for row in completed)
            and any(row["operation"] == "commit" for row in completed),
            "Complete package transfer lacks its reservation or commit")
    return {"components": len(payloads), "bytes": sum(row["bytes"] for row in payloads.values()),
            "complete_package_received": True, "retained_failed_attempts": failures}


def decision(job, *, parent, child, catalog, child_history):
    record = job["record"]
    require(sha(canonical(record)) == job["digest"] and record["status"] == "finished"
            and record["failure"] is None and record["device_id"] == child["binding"]["device_id"]
            and record["independently_verified"] is False, "Parent local decision failed or changed")
    trace = record["trace"]
    require(trace["tool_profile"] == "organism-reproduction-platform-choice-v2",
            "Lineage choice does not use the frozen training protocol")
    raw = streamed_arguments(trace["model_streams"])
    match_streamed_tools(raw, trace)
    require(all(item["valid_json_object"] for item in raw), "Parent emitted malformed literal tool arguments")
    count = len(trace["model_calls"])
    require(1 <= count <= 4 and count == len(trace["http_requests"]) == len(trace["model_streams"])
            == len(trace["model_responses"]) and 2 <= len(trace["tool_events"]) <= 8,
            "Parent local inference trace is incomplete or exceeds its allowance")
    for request in trace["http_requests"]:
        body = request["body"]
        require(request["method"] == "POST" and request["path"] == "/v1/chat/completions"
                and body["model"] == "organism-" + parent["binding"]["start_token"]
                and body["max_tokens"] == 192 and body["temperature"] == 0 and body["stream"] is True,
                "Parent chose reproduction using another model incarnation or budget")
    inspection, choice = None, None
    for event in trace["tool_events"]:
        tool, arguments, value = event["tool"], event["arguments"], event["result"]
        if value.get("status") == "rejected":
            continue
        require(choice is None, "A successful tool followed the final reproduction decision")
        if tool == "inspect_candidate":
            require(arguments == {"device_id": child["binding"]["device_id"]}
                    and value["device_id"] == child["binding"]["device_id"]
                    and value["evidence_id"] == sha(canonical(
                        {key: item for key, item in value.items() if key != "evidence_id"}))
                    and value["release_id"] == catalog["release_id"]
                    and value["supported_platforms"] == sorted(catalog["platforms"])
                    and value["package_bytes"] == sum(item["bytes"] for item in catalog["files"].values())
                    and value["platform"] == "linux-x86_64" and value["owner_mode"] == "enabled"
                    and value["available"] is True, "Parent inspected a different or unavailable child")
            peer = anchored(child_history, value["transport_receipt"])
            public = {key: value[key] for key in
                      ("device_id", "platform", "owner_mode", "available", "revision")}
            public.update(policy_id=child["policy_id"], observation_valid_seconds=30)
            require(peer["phase"] == "completed" and peer["operation"] == "observe"
                    and peer["peer_id"] == parent["binding"]["device_id"]
                    and peer["detail"]["response_sha256"] == sha(canonical(public)),
                    "Model inspection is not backed by this child's authenticated receiver")
            inspection = value
        elif tool == "choose_child_action":
            require(inspection is not None and arguments == {
                "action": "establish", "platform": "linux-x86_64",
                "evidence_id": inspection["evidence_id"]}
                and value == {"status": "choice_recorded", "child_established": False},
                "The local model did not choose an admitted establishment")
            choice = {**arguments, "device_id": inspection["device_id"],
                      "observed_revision": inspection["revision"], "release_id": inspection["release_id"]}
        else:
            raise ValueError("Unexpected successful reproduction tool")
    require(choice is not None and record["submission"] == choice,
            "Saved reproduction action differs from literal executed model tools")
    return {"model_calls": count, "tool_calls": len(trace["tool_events"]),
            "choice": choice, "decision_sha256": job["digest"], "receiver_inspection_verified": True}


def link(parent_evidence, parent, child_evidence, child, catalog, *, child_enabled_at):
    """The two independently retained device histories must describe the same birth."""
    parent_work = "seed-evidence/data/work_root/" + parent["binding"]["individual_id"] + "/"
    identity = parent_evidence.json(parent_work + "reproduction/identity.json", 4096)
    require(identity["schema"] == "organism-reproduction-v1"
            and all(identity[key] == parent["binding"][key] for key in
                    ("device_id", "individual_id", "release_id"))
            and identity["policy_id"] == parent["policy_id"] == child["policy_id"],
            "Parent reproduction journal belongs to another individual or policy")
    boot_id = parent["readiness_records"]["intent"]["record"]["boot_id"]
    child_history = chain(child_evidence, "seed-evidence/control/state_directory/")
    reproduced = chain(parent_evidence, parent_work + "reproduction/")
    operation = child["birth"]["operation_id"]
    transitions = []
    for row in reproduced:
        record = row["record"]
        require(record["operation"] == "reproduce" and record["phase"] == "transition",
                "Parent reproduction history contains another record type")
        # With only one available fresh destination, a second attempted lineage
        # cannot be silently discarded in favor of the successful one.
        require(record["operation_id"] == operation and record["peer_id"] == child["binding"]["device_id"],
                "Parent history includes another attempted descendant")
        value = record["detail"]
        require(value["operation_id"] == operation and value["peer_id"] == record["peer_id"]
                and value["decision_token"] == parent["binding"]["start_token"]
                and value["boot_id"] == boot_id
                and value["deadline_clock"] == value["issued_clock"] + 1200
                and all(type(count) is int and 1 <= count <= 3 for count in value["attempts"].values())
                and moment(record["recorded_at"]) >= moment(child_enabled_at),
                "Parent intent changed its incarnation, original birth allowance, or availability ordering")
        if transitions:
            before = transitions[0]["record"]["detail"]
            require(all(before[key] == value[key] for key in
                        ("peer_id", "peer_revision", "boot_id", "issued_clock", "deadline_clock", "decision_token")),
                    "Parent reassigned an existing reproduction intent")
            require(transitions[-1]["record"]["detail"]["phase"] != "complete",
                    "Completed reproduction history was rewritten")
        transitions.append(row)
    require(transitions and transitions[0]["record"]["detail"]["phase"] == "proposed",
            "Parent lacks its pre-decision reproduction intent")
    final = transitions[-1]["record"]["detail"]
    require(final["phase"] == "complete" and final["failure"] is None
            and final["payload_index"] == len(catalog["files"]) and final["platform"] == "linux-x86_64",
            "Parent never observed the complete child's establishment")
    ready = final["receipt"]
    require(all(ready[key] == child["birth"][key] for key in
                ("operation_id", "parent_device_id", "device_id", "individual_id", "release_id"))
            and ready["status"] == "ready" and ready["ready_once"] is True
            and ready["ready_evidence"] == child["birth"]["ready_evidence"],
            "Parent completion describes another child or readiness result")
    anchor = anchored(child_history, ready["transport_receipt"])
    require(anchor["phase"] == "completed" and anchor["operation"] == "status"
            and anchor["operation_id"] == operation and anchor["peer_id"] == parent["binding"]["device_id"]
            and anchor["detail"]["response_sha256"] == sha(canonical(
                {key: value for key, value in ready.items() if key != "transport_receipt"})),
            "Parent readiness receipt is not in the child's actual receiver history")
    require(parent_evidence.read(parent_work + "reproduction/application.enrolled", 128)
            == b"organism-application-v1\n", "Parent decision journal marker differs")
    jobs = parent_evidence.rows(
        parent_work + "reproduction/application.sqlite3",
        "SELECT request_id,device_id,token,status,digest,substr(body,1,524289) FROM jobs LIMIT 5")
    require(len(jobs) == 1, "Parent local decision inventory differs from the single attempted child")
    request, device, token, status, digest, text = jobs[0]
    require(request == operation and device == child["binding"]["device_id"]
            and token == parent["binding"]["start_token"] and status == "finished"
            and len(text.encode()) <= 512 * 1024 and sha(text.encode()) == digest == final["decision_hash"],
            "Parent choice journal changes its request, target, incarnation or digest")
    checked = decision({"digest": digest, "record": json.loads(text)}, parent=parent, child=child,
                       catalog=catalog, child_history=child_history)
    discovered = chain(parent_evidence, parent_work + "discovery/")
    matches = []
    for row in discovered:
        record = row["record"]
        detail = record["detail"]
        if (record["operation"] == "discover" and record["phase"] == "observed"
                and record["peer_id"] == child["binding"]["device_id"]
                and detail["start_token"] == parent["binding"]["start_token"]
                and detail["boot_id"] == boot_id
                and detail["observation"]["revision"] == final["peer_revision"]
                and detail["observation"]["available"] is True
                and moment(detail["observed_at"]) <= moment(transitions[0]["record"]["recorded_at"])):
            peer = anchored(child_history, detail["observation"]["transport_receipt"])
            public = {key: value for key, value in detail["observation"].items() if key != "transport_receipt"}
            require(detail["authenticated_device_id"] == child["binding"]["device_id"]
                    and detail["tls"] in ("TLSv1.2", "TLSv1.3")
                    and peer["phase"] == "completed" and peer["operation"] == "observe"
                    and peer["peer_id"] == parent["binding"]["device_id"]
                    and peer["detail"]["response_sha256"] == sha(canonical(public))
                    and moment(child_enabled_at) <= moment(detail["observed_at"])
                    <= moment(transitions[0]["record"]["recorded_at"]),
                    "Fresh discovery lacks its authenticated, correctly ordered peer observation")
            matches.append(row)
    require(matches, "Parent did not independently discover the newly available child")
    return {
        "parent_device_id": parent["binding"]["device_id"], "child_device_id": child["binding"]["device_id"],
        "operation_id": operation, "decision": checked, "discovery_records": len(discovered),
        "reproduction_transitions": len(transitions),
        "transfer": transfer(child_history, child, catalog, parent["binding"]["device_id"]),
        "model_choice_and_peer_history_reconstructed": True,
        "external_cloud_and_resource_admission_required": True, "qualification_passed": False,
    }
