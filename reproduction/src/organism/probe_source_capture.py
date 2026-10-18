"""One public Mac stock diagnostic with an owner-updated observation file.

This probes evidence capture, not diagnostic quality or native qualification.
The file is updated once after an actual read event; no model answer is supplied.
"""

import argparse
import asyncio
from dataclasses import asdict, replace
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
from .equipment_development import case
from .inference_server import RuntimeConfig, digest
from .package_application import APPLICATION_MODULES
from .source_journal import SourceJournal


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def audit_saved(output):
    report = json.loads((output / "report.json").read_bytes())
    for name, expected in report["source_hashes"].items():
        if digest(output / "source-snapshot" / name) != expected:
            raise ValueError("Retained source changed")
    config = RuntimeConfig(**{**report["config"], "model": Path(report["config"]["model"]),
                             "executable": Path(report["config"]["executable"])})
    journal = SourceJournal(output / "work-evidence/source-evidence", replace(config, start_token=uuid.uuid4().hex))
    saved = json.loads((output / "diagnostic.json").read_bytes())
    retained = journal.get(report["request_id"])
    if retained is None or retained["job"] != saved["job"]:
        raise ValueError("Application job and retained source differ")
    raw = (output / "original-observation.json").read_bytes()
    identity = hashlib.sha256(raw).hexdigest()
    if retained["observations"].get(identity) != raw:
        raise ValueError("Exact original observation was not retained")
    if retained["independently_verified"] or retained["runtime_context"]["native"]:
        raise ValueError("Development capture claims verification")
    current = output / "work-evidence/equipment-interfaces" / (report["case_id"] + ".json")
    if not report["observation_rotated_after_read"] or current.read_bytes() == raw:
        raise ValueError("This probe did not establish capture across a later update")
    checked = inspect_job(saved["case"], raw, retained["job"], live=True, clock_window=saved["clock_window"])
    if checked != saved["evaluation"]:
        raise ValueError("Independent diagnostic readback differs")
    if report["failure"] or report["server_returncode"] is None or not report["socket_removed"]:
        raise ValueError("Probe completion or owned process shutdown is unconfirmed")
    return {"passed": True, "scope": "Exact source retention during one public Mac stock diagnostic",
            "request_id": report["request_id"], "source_state": retained["state"], "source_reason": retained["reason"],
            "retained_observations": len(retained["observations"]), "original_bytes_retained": True,
            "later_observation_differs": True, "job_retained_exactly": True, "reopened_after_application_exit": True,
            "independent_diagnostic_semantic_passed": checked["semantic_passed"],
            "model_calls": checked["model_calls"], "verifier_endorsements_issued": 0,
            "trained_for_organism": False, "native_execution_verified": False, "qualification_passed": False}


async def run(root, output):
    if platform.system() != "Darwin":
        raise ValueError("This is a Mac development probe")
    output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((root / "organism/experiment-contract.draft.json").read_bytes())
    model = contract["model"]
    config = RuntimeConfig(
        root / "models/granite-3b/granite-base-Q4_K_M.gguf", model["local_export_sha256"], model["local_export_bytes"],
        root / "tools/llama.cpp/b9850/llama-b9850/llama-server", contract["runtime"]["development_binary_sha256"],
        uuid.uuid4().hex, "d" * 64, uuid.uuid4().hex, "e" * 64, False, False)
    report = {"scope": "One public profile-b fault, stock model, exact source capture after file update",
              "started_at": datetime.now(timezone.utc).isoformat(),
              "config": {**asdict(config), "model": str(config.model), "executable": str(config.executable)},
              "source_hashes": {}, "failure": None, "observation_rotated_after_read": False,
              "trained_for_organism": False, "native_execution_verified": False, "qualification_passed": False,
              "verifier_endorsements_issued": 0, "source_release_attestation_claimed": False}
    sources = {name + ".py" for name in APPLICATION_MODULES} | {
        Path(__file__).name, "equipment_development.py", "audit_equipment_diagnostics.py",
        "audit_application.py", "package_application.py"}
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name in sorted(sources):
        raw = (root / "organism" / name).read_bytes()
        (snapshot / name).write_bytes(raw)
        report["source_hashes"][name] = hashlib.sha256(raw).hexdigest()
    write(output / "request.json", report)
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="source-live-", dir="/private/tmp") as temporary:
        work = Path(temporary)
        app = AgentApplication(config, work)
        try:
            report["initial_health"] = await app.start()
            report["server_pid"] = app.inference.process.pid
            print(json.dumps({"application": "serving", "server_pid": report["server_pid"]}), flush=True)
            started = time.time()
            item = case("b", "fault", started)
            identity = report["case_id"] = item["case_id"]
            raw = (json.dumps(item["document"], indent=3) + "\n").encode()
            (output / "original-observation.json").write_bytes(raw)
            path = work / "equipment-interfaces" / (identity + ".json")
            path.write_bytes(raw)
            request_id = report["request_id"] = uuid.uuid4().hex
            request = {"operation": "diagnose", "request_id": request_id, "device_id": identity}
            first = await local_request(app.socket, request)
            retry = await local_request(app.socket, request)
            if first["request_id"] != request_id or retry["request_id"] != request_id:
                raise ValueError("Retry changed the diagnostic request identity")
            report["retry_preserved_identity"] = True
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                job = app.store.get(request_id)
                events = job["record"]["trace"].get("tool_events", [])
                reads = [event for event in events if event.get("tool") == "read_equipment"
                         and event.get("result", {}).get("status") == "observed"]
                if reads and not report["observation_rotated_after_read"]:
                    replacement = json.loads(raw)
                    replacement["sequence"] += 1
                    temporary_path = path.with_suffix(".pending")
                    temporary_path.write_bytes(canonical(replacement))
                    os.replace(temporary_path, path)
                    report["observation_rotated_after_read"] = True
                    report["rotation_unix_seconds"] = time.time()
                    report["observed_read_before_rotation"] = reads[-1]["result"]
                if job["record"]["status"] != "running" and not app.busy:
                    break
                await asyncio.sleep(0.05)
            else:
                raise TimeoutError("Source diagnostic exceeded its declared allowance")
            if not report["observation_rotated_after_read"]:
                raise ValueError("Model did not execute a reading; capture objective was not demonstrated")
            window = [started - 1, time.time() + 1]
            checked = inspect_job(item, raw, job, live=True, clock_window=window)
            write(output / "diagnostic.json", {"case": item, "job": job, "clock_window": window,
                                              "evaluation": checked})
            report["sources"] = await local_request(app.socket, {"operation": "sources"})
            report["diagnostic_semantic_passed"] = checked["semantic_passed"]
            report["model_calls"] = checked["model_calls"]
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
    report["elapsed_seconds"] = round(time.monotonic() - began, 3)
    report["model_child_peak_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    report["sources_unchanged_during_run"] = all(
        digest(root / "organism" / name) == expected for name, expected in report["source_hashes"].items())
    if not report["sources_unchanged_during_run"]:
        report["failure"] = report["failure"] or "Source changed during execution"
    write(output / "report.json", report)
    try:
        audit = audit_saved(output)
    except Exception as exc:
        audit = {"passed": False, "failure": type(exc).__name__ + ": " + str(exc), "qualification_passed": False}
    write(output / "audit.json", audit)
    print(json.dumps({"report": str(output / "report.json"), "capture_passed": audit["passed"],
                      "diagnostic_passed": report.get("diagnostic_semantic_passed"), "failure": report["failure"]}),
          flush=True)
    return report, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, audit = asyncio.run(run(Path(__file__).resolve().parents[1], args.output.resolve()))
    if report["failure"] or not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
