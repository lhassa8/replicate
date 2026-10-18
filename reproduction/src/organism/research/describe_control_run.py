"""Describe any returned control outcome without prescribing a winning arm."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from organism.audit_application import canonical, match_streamed_tools, streamed_arguments
from organism.native_preflight.audit_startup import Evidence, require


ARMS = ("fixed-policy", "stock-control", "trained-merged")
SLOTS = ("seed", "child", "grandchild")


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def model_inventory(trace):
    names = ("model_calls", "http_requests", "model_streams", "model_responses")
    require(all(isinstance(trace.get(name), list) for name in names),
            "Model inventory is missing; absence is not a zero-call record")
    counts = {name: len(trace[name]) for name in names}
    require(len(set(counts.values())) == 1, "Model trace inventories differ")
    return counts


def describe(source, output):
    """Reconstruct inventories; birth validity remains the original audit's job."""
    result = {
        "schema": "replicate-control-description-v1",
        "analysis_prescribes_no_arm_outcome": True,
        "timing_boundary": "owner enable to useful readiness; diagnostics use application RPC and polling",
        "separate_machine_independence_claimed": False,
        "deployment_reliability_or_training_causality_claimed": False,
        "arms": {},
    }
    for arm in ARMS:
        directory = source / arm
        report_path = directory / "report.json"
        if not report_path.exists():
            result["arms"][arm] = {"not_executed": True}
            continue
        report = json.loads(report_path.read_text())
        require(report["arm"] == arm, "Report names another arm")
        row = {
            "report_identity": identity(report_path),
            "release_id": report["release_id"],
            "reported_failure": report["failure"],
            "owner_enable_to_useful_ready_seconds": {},
            "reproduction_jobs": [],
            "diagnostic_records": [],
            "native_application_receipts": [],
        }
        for slot in SLOTS:
            events = [event for event in report["events"] if event["slot"] == slot]
            enabled = [event for event in events if event["operation"] == "enabled"]
            ready = [event for event in events if event["operation"] == "useful_ready"]
            require(len(enabled) <= 1 and len(ready) <= 1,
                    "Repeated activation cannot be summarized as one attempt")
            if ready:
                require(len(enabled) == 1, "Useful readiness has no enabling event")
                elapsed = (datetime.fromisoformat(ready[0]["at"])
                           - datetime.fromisoformat(enabled[0]["at"])).total_seconds()
                require(elapsed >= 0, "Useful readiness precedes enabling")
                row["owner_enable_to_useful_ready_seconds"][slot] = elapsed
            work = directory / slot / "work"
            diagnostics = [directory / slot / "readiness.json",
                           *sorted((directory / slot).glob("task-*.json"))]
            for path in diagnostics:
                if not path.exists():
                    continue
                stored = json.loads(path.read_text())
                job = stored["job"]
                require(hashlib.sha256(canonical(job["record"])).hexdigest() == job["digest"],
                        "Diagnostic record does not match its stored identity")
                trace = job["record"]["trace"]
                counts = model_inventory(trace)
                if arm == "fixed-policy":
                    require(not any(counts.values()), "Fixed diagnostic contains model activity")
                row["diagnostic_records"].append({
                    "slot": slot, "case_id": stored["case"]["case_id"],
                    "identity": identity(path), **counts,
                })
            for path in sorted(work.glob("application-*.json")):
                receipt = json.loads(path.read_text())
                inference = receipt["inference"]
                row["native_application_receipts"].append({
                    "slot": slot, "identity": identity(path),
                    "model_started": inference.get("model_started"),
                    "pid": inference.get("pid"),
                    "owned_server_stopped": receipt["shutdown"]["owned_server_stopped"],
                })
                if arm == "fixed-policy":
                    require(inference["model_started"] is False and inference["pid"] is None,
                            "Fixed policy started an inference process")
            database = work / "reproduction/application.sqlite3"
            if not database.exists():
                continue
            evidence = Evidence(directory / slot)
            jobs = evidence.rows("work/reproduction/application.sqlite3",
                                 "SELECT request_id,token,status,digest,body FROM jobs ORDER BY request_id")
            for request, token, status, digest, text in jobs:
                job = json.loads(text)
                require(hashlib.sha256(text.encode()).hexdigest() == digest
                        and status == job["status"] and request == job["request_id"],
                        "Decision record does not match its stored identity")
                trace = job["trace"]
                streams = trace.get("model_streams", [])
                counts = model_inventory(trace)
                malformed = 0
                if arm == "fixed-policy":
                    require(not any(counts.values()), "Fixed policy made a model call")
                else:
                    raw = streamed_arguments(streams)
                    match_streamed_tools(raw, trace)
                    # Invalid model output is a recorded outcome, not evidence
                    # corruption, when its literal/parsed/executed joins agree.
                    malformed = sum(not item["valid_json_object"] for item in raw)
                    require(all(item["body"]["model"] == "organism-" + token
                                for item in trace["http_requests"]),
                            "Decision used another model incarnation")
                final = trace.get("model_responses", [])[-1:] or [{}]
                response = final[0]
                prose = "".join(part.get("text", "") for part in
                                response.get("message", {}).get("content", []))
                row["reproduction_jobs"].append({
                    "slot": slot, "request_id": request, "job_sha256": digest,
                    "status": status, "failure": job["failure"],
                    "submission": job["submission"], **counts,
                    "executed_tools": [event["tool"] for event in trace["tool_events"]],
                    "malformed_literal_tool_arguments": malformed,
                    "final_stop_reason": response.get("stop_reason"),
                    "final_text_contains_choice_markup": "<function=choose_child_action>" in prose,
                })
        row["total_reproduction_model_calls"] = sum(
            job["model_calls"] for job in row["reproduction_jobs"])
        row["total_diagnostic_model_calls"] = sum(
            job["model_calls"] for job in row["diagnostic_records"])
        result["arms"][arm] = row
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(describe(args.source, args.output), indent=2))
