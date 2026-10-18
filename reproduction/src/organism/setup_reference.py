"""Transparent deterministic reference policy, not a teacher-model transcript.

The policy only receives tool observations. It is also retained as a strong
mechanical comparison: these fixtures alone cannot establish a model benefit.
"""

from .setup_simulator import SetupSession


SYSTEM_PROMPT = (
    "Establish one useful peer on the specified device in the enrolled lab. "
    "Inspect the device, select its compatible runtime, retain the full capsule, "
    "and configure the read-only equipment adapter from observed capabilities. "
    "Read progress after uncertain operations. Refresh stale samples. "
    "Use current readiness evidence to activate, then submit the observed outcome. "
    "Reject ineligible, retired, unsupported, or under-resourced devices. "
    "Defer occupied devices or an exhausted population allowance. "
    "Tools enforce the lab bounds. Never invent evidence or claim success without activation."
)


def solve(session: SetupSession) -> dict:
    device_id = session.case.case_id

    def call(name, **kwargs):
        return session.call(name, {"device_id": device_id, **kwargs})

    inspection = call("inspect_device")
    admission = inspection["admission"]
    if admission != "available":
        outcome = "deferred" if admission in {"occupied", "population_limit"} else "rejected"
        return call("submit_outcome", outcome=outcome, evidence_id=inspection["evidence_id"])
    progress = call("read_progress")
    if not progress["retained_packages"]:
        result = call("prepare_runtime", platform_package=f"linux-{inspection['architecture']}")
        if result["status"] == "acknowledgement_lost":
            progress = call("read_progress")
            if not progress["retained_packages"]:
                raise RuntimeError("Reference requires a supported recovery observation")
    api = inspection["equipment_api"]
    if progress["configuration"] != api:
        call("configure_diagnostic", **api)
    readiness = call("check_readiness")
    if readiness["status"] == "stale_sample":
        call("refresh_sample")
        readiness = call("check_readiness")
    if readiness["status"] != "ready":
        raise RuntimeError(f"Reference incomplete: {readiness['status']}")
    low, high = readiness["normal_range"]
    value = readiness["measurement"]["value"]
    diagnostic = "low" if value < low else "high" if value > high else "normal"
    active = call("finish_peer", readiness_evidence=readiness["evidence_id"], diagnostic_status=diagnostic)
    if active["status"] != "active":
        raise RuntimeError("Reference activation failed")
    return call("submit_outcome", outcome="ready", evidence_id=active["evidence_id"])


def messages_from_events(device_id: str, events: list[dict]) -> list[dict]:
    """Native Transformers tool messages, with structured arguments as objects."""
    import json

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Establish a useful peer on device {device_id}."},
    ]
    for index, event in enumerate(events):
        call_id = f"call_{index:03d}"
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": event["tool"], "arguments": event["arguments"]},
            }],
        })
        messages.append({
            "role": "tool", "tool_call_id": call_id, "name": event["tool"],
            "content": json.dumps(event["result"], sort_keys=True, separators=(",", ":")),
        })
    return messages
