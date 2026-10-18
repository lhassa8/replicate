"""Bounded live Mac stock-model check of public development interfaces."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import tempfile
import time
import uuid

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

from .application import AgentApplication, local_request
from .audit_equipment_diagnostics import inspect_job
from .capsule import canonical
from .equipment_baseline import solve
from .equipment_development import case
from .equipment_interfaces import InterfaceSession
from .inference_server import RuntimeConfig, digest
from .package_application import APPLICATION_MODULES


SELECTION = (("a", "nominal"), ("b", "fault"), ("c", "fault"), ("d", "fault"),
             ("e", "fault"), ("f", "fault"), ("g", "stale"), ("h", "invalid"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def baseline_job(session):
    record = {"device_id": session.device_id, "task_kind": "equipment_interface_v1",
              "status": "finished", "submission": session.submission,
              "trace": {"tool_events": session.events, "fixed_policy": True,
                        "tool_profile": session.tool_profile}}
    return {"record": record, "digest": hashlib.sha256(canonical(record)).hexdigest()}


def audit_saved(output):
    report = json.loads((output / "report.json").read_bytes())
    items = []
    for summary in report["cases"]:
        stored = json.loads((output / (summary["case_id"] + ".json")).read_bytes())
        raw = (output / "work-evidence/equipment-interfaces" / (summary["case_id"] + ".json")).read_bytes()
        baseline = inspect_job(stored["case"], raw, stored["baseline_job"])
        model = inspect_job(stored["case"], raw, stored["job"], live=True, clock_window=stored["clock_window"])
        if baseline != stored["baseline_evaluation"] or model != stored["evaluation"]:
            raise ValueError("Saved model or baseline evaluation differs")
        if model["semantic_passed"] != summary["model_passed"] or baseline["semantic_passed"] != summary["baseline_passed"]:
            raise ValueError("Case summary differs from independent readback")
        items.append({"case_id": summary["case_id"], "model_passed": model["semantic_passed"],
                      "baseline_passed": baseline["semantic_passed"], "input_and_trace_verified": True})
    if len(items) != len(SELECTION) or len({item["case_id"] for item in items}) != len(SELECTION):
        raise ValueError("Development probe case inventory is incomplete")
    for name, expected in report["source_hashes"].items():
        if digest(output / "source-snapshot" / name) != expected:
            raise ValueError("Retained probe source changed")
    return {"audit_passed": True, "scope": "Independent readback of live public Mac development cases",
            "cases": items, "model_successes": sum(item["model_passed"] for item in items),
            "fixed_policy_successes": sum(item["baseline_passed"] for item in items),
            "trained_for_organism": False, "native_execution_verified": False, "qualification_passed": False}


async def run(root, output):
    if platform.system() != "Darwin":
        raise ValueError("This preflight is a Mac development check")
    output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((root / "organism/experiment-contract.draft.json").read_bytes())
    model = contract["model"]
    config = RuntimeConfig(
        root / "models/granite-3b/granite-base-Q4_K_M.gguf", model["local_export_sha256"], model["local_export_bytes"],
        root / "tools/llama.cpp/b9850/llama-b9850/llama-server", contract["runtime"]["development_binary_sha256"],
        uuid.uuid4().hex, "d" * 64, uuid.uuid4().hex, "e" * 64, False, False)
    report = {
        "scope": "Live stock model in the Mac application; public development equipment files",
        "release_status": "unreleased_development_application", "started_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": config.model_sha256, "runtime_sha256": config.executable_sha256,
        "selection": [{"profile": profile, "variation": variation} for profile, variation in SELECTION],
        "baseline_status": "development_policy_candidate_not_frozen_Q5_baseline",
        "trained_for_organism": False, "native_execution_verified": False,
        "autonomous_reproduction_verified": False, "qualification_passed": False,
        "cases": [], "failure": None, "source_hashes": {},
    }
    sources = {name + ".py" for name in APPLICATION_MODULES} | {
        Path(__file__).name, "equipment_baseline.py", "equipment_development.py",
        "audit_equipment_diagnostics.py", "audit_application.py", "package_application.py"}
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name in sorted(sources):
        raw = (root / "organism" / name).read_bytes()
        (snapshot / name).write_bytes(raw)
        report["source_hashes"][name] = hashlib.sha256(raw).hexdigest()
    write(output / "request.json", report)
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="oi-live-", dir="/private/tmp") as temporary:
        work = Path(temporary)
        app = AgentApplication(config, work)
        try:
            report["initial_health"] = await app.start()
            report["server_pid"] = app.inference.process.pid
            print(json.dumps({"application": "serving", "server_pid": report["server_pid"]}), flush=True)
            for profile, variation in SELECTION:
                window_start = time.time()
                item = case(profile, variation, window_start)
                identity = item["case_id"]
                raw = canonical(item["document"])
                path = work / "equipment-interfaces" / (identity + ".json")
                path.write_bytes(raw)
                reference = InterfaceSession(app.equipment_interfaces, identity, canonical_temperature=True)
                solve(reference, identity)
                fixed_job = baseline_job(reference)
                fixed_check = inspect_job(item, raw, fixed_job)
                request_id = uuid.uuid4().hex
                request = {"operation": "diagnose", "request_id": request_id, "device_id": identity}
                first = await local_request(app.socket, request)
                retry = await local_request(app.socket, request)
                if first["request_id"] != request_id or retry["request_id"] != request_id:
                    raise ValueError("Diagnostic retry changed its identity")
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    state = await local_request(app.socket, {"operation": "status", "request_id": request_id})
                    if state["status"] != "running":
                        break
                    await asyncio.sleep(0.25)
                else:
                    raise TimeoutError("Diagnostic did not settle within its bounded allowance")
                job = app.store.get(request_id)
                window = [window_start - 1, time.time() + 1]
                try:
                    check = inspect_job(item, raw, job, live=True, clock_window=window)
                except Exception as exc:
                    check = {"semantic_passed": False, "audit_failure": type(exc).__name__ + ": " + str(exc),
                             "model_calls": len(job["record"]["trace"].get("model_calls", [])),
                             "tool_calls": len(job["record"]["trace"].get("tool_events", [])),
                             "qualification_passed": False}
                if path.read_bytes() != raw:
                    raise ValueError("Read-only diagnostic changed the equipment input")
                stored = {"case": item, "job": job, "clock_window": window, "request_id": request_id,
                          "evaluation": check, "baseline_job": fixed_job, "baseline_evaluation": fixed_check,
                          "input_unchanged": True, "retry_preserved_identity": True}
                write(output / (identity + ".json"), stored)
                summary = {"case_id": identity, "model_passed": check["semantic_passed"],
                           "baseline_passed": fixed_check["semantic_passed"],
                           "model_calls": check["model_calls"], "tool_calls": check["tool_calls"],
                           "job_status": job["record"]["status"], "failure": job["record"]["failure"],
                           "audit_failure": check.get("audit_failure"),
                           "seconds": job["record"]["trace"].get("seconds")}
                report["cases"].append(summary)
                write(output / "progress.json", report)
                print(json.dumps(summary), flush=True)
            report["final_health"] = app.health()
        except Exception as exc:
            report["failure"] = type(exc).__name__ + ": " + str(exc)
        finally:
            try:
                await app.close()
                report["shutdown"] = app.receipt.get("shutdown")
            except Exception as exc:
                report["shutdown_failure"] = type(exc).__name__ + ": " + str(exc)
            report["server_returncode"] = app.inference.process.returncode if app.inference.process else None
            report["socket_removed"] = not app.socket.exists()
            shutil.copytree(work, output / "work-evidence", ignore=shutil.ignore_patterns("*.sock"))
            logs = b"".join(path.read_bytes() for path in (output / "work-evidence").glob("model-*.log"))
            report["api_key_found_in_model_logs"] = app.inference.api_key.encode() in logs
    report["elapsed_seconds"] = round(time.monotonic() - began, 3)
    report["model_child_peak_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["model_successes"] = sum(item["model_passed"] for item in report["cases"])
    report["fixed_policy_successes"] = sum(item["baseline_passed"] for item in report["cases"])
    report["sources_unchanged_during_run"] = all(
        digest(root / "organism" / name) == expected for name, expected in report["source_hashes"].items())
    if not report["sources_unchanged_during_run"]:
        report["failure"] = report["failure"] or "Source files changed during the probe"
    write(output / "report.json", report)
    try:
        audit = audit_saved(output)
    except Exception as exc:
        audit = {"audit_passed": False, "failure": type(exc).__name__ + ": " + str(exc), "qualification_passed": False}
    write(output / "audit.json", audit)
    print(json.dumps({"report": str(output / "report.json"), "model_successes": report["model_successes"],
                      "fixed_policy_successes": report["fixed_policy_successes"], "failure": report["failure"],
                      "audit_passed": audit["audit_passed"]}), flush=True)
    return report, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, audit = asyncio.run(run(Path(__file__).resolve().parents[1], args.output.resolve()))
    if report["failure"] or not audit["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
