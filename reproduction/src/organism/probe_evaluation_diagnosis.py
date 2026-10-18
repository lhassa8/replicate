"""Two bounded public Mac stock-model trials of evaluator instrumentation."""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
import uuid

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

from .application import AgentApplication, local_request
from .audit_equipment_diagnostics import inspect_job
from .capsule import canonical
from .equipment_development import case
from .evaluation_diagnosis import EvaluationDiagnoser
from .evaluation_observation import FAULT, ObservationTrial
from .inference_server import RuntimeConfig, digest
from .package_application import APPLICATION_MODULES
from .qualification_inputs import REFERENCE_TIME, time_paths


SELECTION = ("ordinary", "configuration_interruption")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate(item, raw, job, window):
    try:
        return {"audit_passed": True, **inspect_job(item, raw, job, live=True, clock_window=window)}
    except Exception as exc:
        return {"audit_passed": False, "semantic_passed": False,
                "audit_failure": type(exc).__name__ + ": " + str(exc), "qualification_passed": False}


def audit_saved(output):
    report = json.loads((output / "report.json").read_bytes())
    for name, expected in report["sources"].items():
        if digest(output / "sources" / name) != expected:
            raise ValueError("Saved evaluator source differs")
    results = []
    for name in SELECTION:
        saved = json.loads((output / name / "result.json").read_bytes())
        raw = (output / name / "capture/observation.json").read_bytes()
        observed = output / name / "work/equipment-interfaces" / (saved["case"]["case_id"] + ".json")
        if raw != observed.read_bytes():
            raise ValueError("Retained application observation differs")
        checked = evaluate(saved["case"], raw, saved["job"], saved["clock_window"])
        if checked != saved["evaluation"]:
            raise ValueError("Saved live diagnostic reconstruction differs")
        result = {"case": name, "evaluation": checked, "trial": saved["trial_receipt"]}
        results.append(result)
    return {"saved_results_reconstructed": True, "cases": results,
            "all_model_traces_verified": all(value["evaluation"]["audit_passed"] for value in results),
            "native_linux_execution_verified": False, "qualification_passed": False}


async def run(root, output):
    if platform.system() != "Darwin":
        raise ValueError("This probe is confined to public development tasks on the Mac")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    contract = json.loads((root / "organism/experiment-contract.draft.json").read_bytes())
    source_names = {name + ".py" for name in APPLICATION_MODULES} | {
        "evaluation_observation.py", "evaluation_diagnosis.py", "probe_evaluation_diagnosis.py",
        "equipment_development.py", "qualification_inputs.py", "package_application.py",
    }
    (output / "sources").mkdir()
    report = {
        "schema": "organism-evaluation-diagnosis-probe-v1", "started_at": datetime.now(timezone.utc).isoformat(),
        "selection": list(SELECTION), "profile": "public development a", "condition": "fault",
        "sources": {}, "cases": [], "failure": None,
        "model_sha256": contract["model"]["local_export_sha256"],
        "runtime_sha256": contract["runtime"]["development_binary_sha256"],
        "harness_supplied_tool_choices_or_answers": False, "application_rpc_exercised": True,
        "trained_for_organism": False, "native_linux_execution_verified": False, "qualification_passed": False,
    }
    for name in sorted(source_names):
        raw = (root / "organism" / name).read_bytes()
        (output / "sources" / name).write_bytes(raw)
        report["sources"][name] = hashlib.sha256(raw).hexdigest()
    write(output / "request.json", report)
    try:
        for name in SELECTION:
            target = output / name
            target.mkdir(mode=0o700)
            saved = {"selection": name, "failure": None, "qualification_passed": False}
            binding = EvaluationDiagnoser()
            trial, request_id = None, None
            with tempfile.TemporaryDirectory(prefix="oe-live-", dir="/private/tmp") as temporary:
                work = Path(temporary)
                config = RuntimeConfig(
                    root / "models/granite-3b/granite-base-Q4_K_M.gguf",
                    report["model_sha256"], contract["model"]["local_export_bytes"],
                    root / "tools/llama.cpp/b9850/llama-b9850/llama-server",
                    report["runtime_sha256"], uuid.uuid4().hex, "d" * 64,
                    uuid.uuid4().hex, "e" * 64, False, False)
                app = AgentApplication(config, work, diagnoser=binding)
                try:
                    saved["initial_health"] = await app.start()
                    saved["server_pid"] = app.inference.process.pid
                    print(json.dumps({"case": name, "state": "model_ready", "pid": saved["server_pid"]}), flush=True)
                    item = case("a", "fault", REFERENCE_TIME)
                    item.update(time_binding={"fields": time_paths(item["document"])},
                                fault=deepcopy(FAULT) if name == "configuration_interruption" else None)
                    trial = ObservationTrial(item, app.equipment_interfaces.directory, target / "capture")
                    binding.bind(trial)
                    saved["case"] = {**item, "document": json.loads(trial.raw)}
                    request_id = uuid.uuid4().hex
                    request = {"operation": "diagnose", "request_id": request_id, "device_id": item["case_id"]}
                    saved["initial_request"] = await local_request(app.socket, request)
                    saved["repeated_request"] = await local_request(app.socket, request)
                    if saved["initial_request"].get("request_id") != request_id or saved["repeated_request"].get("request_id") != request_id:
                        raise ValueError("Application did not retain the exact diagnostic request")
                    deadline = time.monotonic() + trial.timing()["remaining_seconds"] + 10
                    while time.monotonic() < deadline:
                        status = await local_request(app.socket, {"operation": "status", "request_id": request_id})
                        if status["status"] != "running" and not app.busy:
                            break
                        await asyncio.sleep(0.25)
                    else:
                        raise TimeoutError("Application did not settle after its original trial deadline")
                    saved["job"] = app.store.get(request_id)
                    saved["binding"] = deepcopy(binding.receipt)
                    saved["clock_window"] = [trial.wall_start - 1, time.time() + 1]
                    saved["evaluation"] = evaluate(saved["case"], trial.raw, saved["job"], saved["clock_window"])
                    acceptable = (saved["evaluation"]["semantic_passed"] and binding.receipt["failure"] is None
                                  and binding.receipt["sources_unchanged"]
                                  and binding.receipt.get("observation_unchanged") is True)
                    saved["trial_receipt"] = trial.finish(task_success=acceptable)
                except Exception as exc:
                    saved["failure"] = type(exc).__name__ + ": " + str(exc)
                finally:
                    await app.close()
                    saved["shutdown"] = app.receipt.get("shutdown")
                    saved["server_returncode"] = app.inference.process.returncode if app.inference.process else None
                    saved["socket_removed"] = not app.socket.exists()
                    if request_id is not None:
                        saved["job"] = app.store.get(request_id)
                    saved["binding"] = deepcopy(binding.receipt)
                    shutil.copytree(work, target / "work", ignore=shutil.ignore_patterns("*.sock"))
                    saved["api_key_found_in_logs"] = any(app.inference.api_key.encode() in path.read_bytes()
                                                        for path in (target / "work").glob("model-*.log"))
                    write(target / "result.json", saved)
            job = saved.get("job", {}).get("record", {})
            trace = job.get("trace", {})
            summary = {"case": name, "failure": saved["failure"], "job_status": job.get("status"),
                       "job_failure": job.get("failure"), "model_calls": len(trace.get("model_calls", [])),
                       "tool_calls": len(trace.get("tool_events", [])), "evaluation": saved.get("evaluation"),
                       "trial_receipt": saved.get("trial_receipt"), "shutdown": saved["shutdown"]}
            report["cases"].append(summary)
            write(output / "progress.json", report)
            print(json.dumps(summary), flush=True)
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["sources_unchanged"] = all(digest(root / "organism" / name) == expected
                                           for name, expected in report["sources"].items())
        write(output / "report.json", report)
    try:
        audit = audit_saved(output)
    except Exception as exc:
        audit = {"saved_results_reconstructed": False, "failure": type(exc).__name__ + ": " + str(exc),
                 "qualification_passed": False}
    write(output / "audit.json", audit)
    print(json.dumps({"report": str(output / "report.json"), "saved_results_reconstructed":
                      audit["saved_results_reconstructed"], "qualification_passed": False}), flush=True)
    return report, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report, audit = asyncio.run(run(Path(__file__).resolve().parents[1], arguments.output.resolve()))
    raise SystemExit(0 if report["failure"] is None and audit["saved_results_reconstructed"] else 1)
