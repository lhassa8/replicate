"""Matched complete development diagnostics with stock and admitted local LoRA.

This runs the existing application and Strands loop, with private local fixture
files. It does not use qualification data, launch descendants, alter training,
replace the frozen merged export, or claim native device qualification.
"""

import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import tempfile
import time
import uuid

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

from .application import AgentApplication, local_request
from .audit_trained_tasks import ARMS, audit, check_adapters, evaluate, with_clock
from .capsule import canonical
from .diagnostic_agent import EquipmentTool
from .equipment_baseline import solve
from .equipment_development import case, VARIATIONS
from .equipment_diagnostic_agent import SYSTEM_PROMPT, SPECS
from .equipment_interfaces import InterfaceSession
from .temperature_measurement import PROFILE
from .inference_server import RuntimeConfig, digest
from .package_application import APPLICATION_MODULES
from .probe_equipment_interfaces import baseline_job


LIMITS = {"arm_seconds": 1800, "case_seconds": 180, "maximum_new_bytes": 1024**3,
          "sampled_server_rss_bytes": 6 * 1024**3}
PREVIOUS = "organism/evidence/local-trained-adapter-20261015-01"
PREVIOUS_REPORT_SHA256 = "22a8809161fee3a68e6990774ca19586995b10d31851c812b2adca9d39857610"


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def registered_specs():
    from strands.tools.registry import ToolRegistry
    registry = ToolRegistry()
    registry.process_tools([EquipmentTool(None, spec) for spec in SPECS])
    return registry.get_all_tool_specs()


def make_plan(root, output, admission_path):
    previous = root / PREVIOUS
    old_report = previous / "report.json"
    require(digest(old_report) == PREVIOUS_REPORT_SHA256, "Prior conversion report identity changed")
    old = json.loads(old_report.read_text())
    admission = json.loads(admission_path.read_text())
    require(admission == old["training_admission"], "Current training admission differs from conversion")
    adapter = previous / "trained-adapter-f32.gguf"
    model = root / "models/granite-3b/granite-base-Q4_K_M.gguf"
    binary = root / "tools/llama.cpp/b9850/llama-b9850/llama-server"
    require(identity(adapter) == old["converted_adapter"] and identity(model) == old["stock_model"],
            "Model or converted adapter changed")
    require(identity(previous / "tensor-audit.json") == old["tensor_audit"],
            "Prior tensor audit changed")
    runtime_inputs = {name: expected for name, expected in old["tool_inputs"]["files"].items()
                      if name.startswith("tools/llama.cpp/") and not name.endswith("llama-quantize")}
    for name, expected in runtime_inputs.items():
        require(identity(root / name) == expected, "Pinned inference runtime changed")
    shutil.copy2(admission_path, output / "training-admission.json")
    plan = {"schema": "organism-trained-development-tasks-plan-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Ten public development validation conditions, actual application and Strands tools",
            "cases": [case(profile, variation, now=0) for profile in ("g", "h") for variation in VARIATIONS],
            "system_prompt": SYSTEM_PROMPT, "tool_specs": registered_specs(), "tool_profile": PROFILE,
            "arms": list(ARMS),
            "adapter_path": str(adapter), "adapter": identity(adapter), "model_path": str(model),
            "model": identity(model), "binary_path": str(binary), "runtime_inputs": runtime_inputs,
            "prior_conversion_report": identity(old_report), "tensor_audit": old["tensor_audit"],
            "training_admission": identity(output / "training-admission.json"),
            "limits": LIMITS, "source_files": {}, "total_stored_parameters": old["total_stored_parameters"],
            "versions": {name: version(name) for name in ("strands-agents", "httpx", "cryptography", "jsonschema")},
            "baseline_status": "Existing unchanged development fixed policy; not a blind Q5 comparison",
            "matching": "Same fixtures except fresh per-task timestamps; stock then trained, same inference settings",
            "development_validation_only": True, "qualification_passed": False}
    require(plan["versions"]["strands-agents"] == "1.57.1", "Unexpected Strands version")
    require(all(item["split"] == "development_validation" for item in plan["cases"]),
            "Comparison includes non-validation cases")
    names = {name + ".py" for name in APPLICATION_MODULES} | {
        "probe_trained_tasks.py", "audit_trained_tasks.py", "probe_equipment_interfaces.py",
        "equipment_development.py", "equipment_baseline.py", "package_application.py"}
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name in sorted(names):
        path = root / "organism" / name
        shutil.copy2(path, snapshot / name)
        plan["source_files"][name] = identity(path)
    write(output / "plan.json", plan)
    return plan


async def monitor(app, output, work, record):
    while app.inference.process is None or app.inference.process.returncode is None:
        process = app.inference.process
        if process is not None:
            sample = await asyncio.create_subprocess_exec(
                "/bin/ps", "-p", str(process.pid), "-o", "rss=",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, _ = await asyncio.wait_for(sample.communicate(), timeout=3)
            require(sample.returncode == 0 and stdout.strip().isdigit(), "Cannot observe owned process RSS")
            rss = int(stdout.strip()) * 1024
            record["rss_samples"].append({"monotonic_seconds": time.monotonic(), "rss_bytes": rss})
            size = sum(path.stat().st_size for directory in (output, work)
                       for path in directory.rglob("*") if path.is_file())
            if rss > LIMITS["sampled_server_rss_bytes"] or size > LIMITS["maximum_new_bytes"]:
                app.inference.failure = "Development process memory or artifact allowance exhausted"
                if process.returncode is None:
                    process.terminate()
                return
        await asyncio.sleep(2)


async def run_arm(root, output, arm, plan, *, model_override=None):
    directory = output / arm
    directory.mkdir()
    scale = 0 if arm == ARMS[0] else 1
    launcher = directory / "launcher.sh"
    direct = model_override is not None
    model_path = model_override["path"] if direct else plan["model_path"]
    model = model_override if direct else plan["model"]
    launcher.write_text("#!/bin/sh\nexec " + shlex.quote(plan["binary_path"])
                        + ("" if direct else
                           " --lora-scaled " + shlex.quote(plan["adapter_path"] + ":" + str(scale)))
                        + ' "$@"\n')
    launcher.chmod(0o700)
    executable = Path(plan["binary_path"]) if direct else launcher
    config = RuntimeConfig(Path(model_path), model["sha256"], model["bytes"],
                           executable, digest(executable), uuid.uuid4().hex, "d" * 64, uuid.uuid4().hex,
                           "e" * 64, scale == 1, False)
    record = {"arm": arm, "adapter_scale": scale, "cases": [], "failure": None, "rss_samples": [],
              "started_at": datetime.now(timezone.utc).isoformat(), "qualification_passed": False}
    if direct:
        record.update(adapter_scale=None, model_representation="merged-q4", model_input=model_override)
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="otasks-", dir="/private/tmp") as temporary:
        work = Path(temporary)
        app = AgentApplication(config, work)
        watcher = None
        try:
            async with asyncio.timeout(LIMITS["arm_seconds"]):
                watcher = asyncio.create_task(monitor(app, output, work, record))
                record["initial_health"] = await app.start()
                record["model_alias"] = app.inference.alias
                record["server_pid"] = app.inference.process.pid
                record["server_start"] = app.receipt["inference"]
                record["loaded_adapters"] = await app.inference.json_get("/lora-adapters")
                if direct:
                    require(record["loaded_adapters"] == [], "Exported model unexpectedly loaded a separate adapter")
                else:
                    check_adapters(record["loaded_adapters"], plan["adapter_path"], scale)
                print(json.dumps({"arm": arm, "serving_pid": record["server_pid"]}), flush=True)
                write(directory / "progress.json", record)
                for template in plan["cases"]:
                    if watcher.done():
                        await watcher
                        require(False, "Resource monitor stopped during inference")
                    require(not app.inference.failure, app.inference.failure)
                    epoch = time.time()
                    item = with_clock(template, epoch)
                    raw = canonical(item["document"])
                    path = work / "equipment-interfaces" / (item["case_id"] + ".json")
                    path.write_bytes(raw)
                    fixed = InterfaceSession(app.equipment_interfaces, item["case_id"],
                                             canonical_temperature=plan.get("tool_profile") == PROFILE)
                    baseline_started = time.monotonic()
                    solve(fixed, item["case_id"])
                    fixed_job = baseline_job(fixed)
                    baseline_seconds = time.monotonic() - baseline_started
                    request_id = uuid.uuid4().hex
                    request = {"operation": "diagnose", "request_id": request_id, "device_id": item["case_id"]}
                    first = await local_request(app.socket, request)
                    retry = await local_request(app.socket, request)
                    require(first["request_id"] == retry["request_id"] == request_id, "Retry changed request identity")
                    async with asyncio.timeout(LIMITS["case_seconds"]):
                        while True:
                            state = await local_request(app.socket, {"operation": "status", "request_id": request_id})
                            if state["status"] != "running":
                                break
                            await asyncio.sleep(0.25)
                    job = app.store.get(request_id)
                    require(path.read_bytes() == raw, "Read-only diagnostic modified its equipment observation")
                    stored = {"case": item, "fixture_epoch": epoch, "clock_window": [epoch - 1, time.time() + 1],
                              "request_id": request_id, "job": job, "baseline_job": fixed_job,
                              "baseline_seconds": baseline_seconds, "input_unchanged": True,
                              "retry_preserved_identity": True}
                    check, fixed_check = evaluate(stored, raw), evaluate(stored, raw, baseline=True)
                    stored.update(evaluation=check, baseline_evaluation=fixed_check)
                    write(directory / (item["case_id"] + ".json"), stored)
                    summary = {"case_id": item["case_id"], "model_passed": check["semantic_passed"],
                               "baseline_passed": fixed_check["semantic_passed"],
                               "trace_audit_passed": check["audit_passed"],
                               "model_calls": check["model_calls"], "tool_calls": check["tool_calls"],
                               "seconds": job["record"]["trace"].get("seconds"),
                               "job_status": job["record"]["status"], "failure": job["record"]["failure"],
                               "audit_failure": check.get("audit_failure")}
                    record["cases"].append(summary)
                    write(directory / "progress.json", record)
                    print(json.dumps({"arm": arm, **summary}), flush=True)
                record["final_adapters"] = await app.inference.json_get("/lora-adapters")
                require(record["final_adapters"] == record["loaded_adapters"], "Adapter state changed")
                record["final_health"] = app.health()
        except BaseException as exc:
            record["failure"] = type(exc).__name__ + ": " + str(exc)
        finally:
            if watcher is not None:
                if watcher.done() and not watcher.cancelled() and watcher.exception() is not None:
                    record["failure"] = record["failure"] or "Resource observation failed: " + str(watcher.exception())
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            try:
                await app.close()
                record["shutdown"] = app.receipt.get("shutdown")
            finally:
                record["socket_removed"] = not app.socket.exists()
                shutil.copytree(work, directory / "work-evidence", ignore=shutil.ignore_patterns("*.sock"))
                record["api_key_found_in_model_logs"] = any(
                    app.inference.api_key.encode() in path.read_bytes()
                    for path in (directory / "work-evidence").glob("model-*.log"))
                record["peak_sampled_server_rss_bytes"] = max(
                    (item["rss_bytes"] for item in record["rss_samples"]), default=None)
                record["elapsed_seconds"] = time.monotonic() - began
                record["completed_at"] = datetime.now(timezone.utc).isoformat()
                write(directory / "report.json", record)
    return record


async def run(output, admission_path):
    require(platform.system() == "Darwin", "This is a local Mac development probe")
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    require(shutil.disk_usage(output).free > 2 * 1024**3, "Insufficient evidence space")
    plan = make_plan(root, output, admission_path.resolve())
    report = {"schema": "organism-trained-development-tasks-v1", "plan": identity(output / "plan.json"),
              "started_at": datetime.now(timezone.utc).isoformat(), "arms": {}, "failure": None,
              "qualification_passed": False}
    try:
        for arm in ARMS:
            record = await run_arm(root, output, arm, plan)
            report["arms"][arm] = record
            require(record["failure"] is None, "Arm failed: " + str(record["failure"]))
            require(record["shutdown"]["owned_server_stopped"] is True, "Owned process still running")
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        report["sources_unchanged"] = all(identity(root / "organism" / name) == expected
                                          for name, expected in plan["source_files"].items())
        report["inputs_unchanged"] = (
            identity(Path(plan["model_path"])) == plan["model"]
            and identity(Path(plan["adapter_path"])) == plan["adapter"]
            and all(identity(root / name) == expected for name, expected in plan["runtime_inputs"].items()))
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        write(output / "report.json", report)
    try:
        checked = audit(output)
    except Exception as exc:
        checked = {"evidence_reconstructed": False, "failure": type(exc).__name__ + ": " + str(exc),
                   "qualification_passed": False}
    write(output / "audit.json", checked)
    print(json.dumps({"report": str(output / "report.json"), "failure": report["failure"],
                      "evidence_reconstructed": checked["evidence_reconstructed"]}), flush=True)
    return report, checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    report, checked = asyncio.run(run(args.output, args.admission))
    if report["failure"] or not checked["evidence_reconstructed"]:
        raise SystemExit(1)
