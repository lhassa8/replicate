"""Real Mac stock-model application check; no native or lineage qualification."""

import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import tempfile
import time
import uuid

# Scope these settings to this fresh development process before SDK import.
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

import httpx

from .application import AgentApplication, local_request
from .diagnostics import DEVELOPMENT_CASES
from .inference_server import RuntimeConfig, digest
from .probe_habitat import ProbeHabitat


def evaluate(case, record, observation_bytes):
    """Evaluator owns expectations; input files contain no expected answers."""
    trace, submission = record["trace"], record["submission"]
    readings = [event["result"] for event in trace.get("tool_events", [])
                if event["tool"] == "read_measurement" and event["result"].get("status") == "ok"]
    capabilities = [event["result"] for event in trace.get("tool_events", [])
                    if event["tool"] == "inspect_device" and event["result"].get("status") == "ok"]
    if not submission or not readings or not capabilities:
        return {"passed": False, "reason": "No complete evidence-backed assessment"}
    observed = readings[-1]
    source = json.loads(observation_bytes)
    genuine = (observed["observation_sha256"] == hashlib.sha256(observation_bytes).hexdigest()
               and observed["value"] == source["value"] and observed["unit"] == source["unit"]
               and observed["observed_unix_seconds"] == source["observed_unix_seconds"]
               and observed["sequence"] == source["sequence"])
    expected_status, expected_temperature = case.expected_status, case.expected_celsius
    stale = observed["age_seconds"] > case.max_age_seconds
    if stale:
        # Evaluate actual freshness at the measurement call. A slow CPU can
        # make a previously fresh observation stale; never call that fresh.
        expected_status, expected_temperature = "unknown", None
    temperature = submission["temperature_celsius"]
    temperature_ok = (temperature is None if expected_temperature is None else
                      type(temperature) in (int, float) and abs(temperature - expected_temperature) <= 0.05)
    evidence_ok = set(submission["evidence_ids"]) == {readings[-1]["evidence_id"], capabilities[-1]["evidence_id"]}
    passed = genuine and evidence_ok and temperature_ok and submission["status"] == expected_status and record["status"] == "finished"
    return {"passed": bool(passed), "observed_input_matches": genuine, "evidence_ids_match": evidence_ok,
            "expected_status": expected_status, "expected_celsius": expected_temperature,
            "observed_age_seconds": observed["age_seconds"], "freshness_at_read": "stale" if stale else "current",
            "status_matches": submission["status"] == expected_status, "temperature_matches": temperature_ok}


async def run(root, output, *, cases=DEVELOPMENT_CASES, with_discovery=False):
    if platform.system() != "Darwin":
        raise ValueError("This development probe is specifically for the Mac")
    output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((root / "organism/experiment-contract.draft.json").read_text())
    cfg = contract["model"]
    config = RuntimeConfig(
        root / "models/granite-3b/granite-base-Q4_K_M.gguf", cfg["local_export_sha256"], cfg["local_export_bytes"],
        root / "tools/llama.cpp/b9850/llama-b9850/llama-server", contract["runtime"]["development_binary_sha256"],
        uuid.uuid4().hex, "d" * 64, uuid.uuid4().hex, "e" * 64, False, False,
    )
    report = {"scope": "Live Mac application with retained stock model and public file-backed development observations",
              "started_at": datetime.now(timezone.utc).isoformat(), "model_sha256": config.model_sha256,
              "runtime_sha256": config.executable_sha256, "trained_for_organism": False,
              "native_execution_verified": False, "autonomous_reproduction_verified": False,
              "qualification_passed": False, "selection": [case.case_id for case in cases],
              "with_discovery": with_discovery,
              "cases": [], "failure": None}
    # Preserve the actual sources at the start, including the separate probe
    # helpers. Private identity data is created elsewhere and is never copied.
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    sources = {Path(__file__)}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).parent == root / "organism" and Path(path).suffix == ".py":
            sources.add(Path(path))
    report["source_hashes"] = {}
    for path in sorted(sources):
        raw = path.read_bytes()
        (snapshot / path.name).write_bytes(raw)
        report["source_hashes"][path.name] = hashlib.sha256(raw).hexdigest()
    (output / "request.json").write_text(json.dumps(report, indent=2) + "\n")
    started = time.monotonic()
    with ExitStack() as cleanup:
        work = Path(cleanup.enter_context(tempfile.TemporaryDirectory(prefix="oa-probe-", dir="/private/tmp")))
        habitat = cleanup.enter_context(ProbeHabitat()) if with_discovery else None
        discovery = None
        if habitat is not None:
            config, discovery = habitat.configure(config, work)
        app = AgentApplication(config, work, discovery=discovery)
        try:
            report["initial_health"] = await app.start()
            report["server_pid"] = app.inference.process.pid
            print(json.dumps({"application": "serving", "server_pid": report["server_pid"]}), flush=True)
            async with httpx.AsyncClient(trust_env=False, timeout=3) as anonymous:
                listing = await anonymous.get(app.inference.base_url + "/v1/models")
                report["public_model_listing_status"] = listing.status_code
                response = await anonymous.post(app.inference.base_url + "/v1/chat/completions", json={
                    "model": app.inference.alias, "messages": [{"role": "user", "content": "Reply OK."}],
                    "max_tokens": 1,
                })
                report["anonymous_inference_request_status"] = response.status_code
                if response.status_code != 401:
                    raise RuntimeError("Unauthenticated inference request was not rejected")
            if habitat is not None:
                await habitat.before_arrival(discovery)
            for case in cases:
                observed = {"schema": "organism-temperature-v1", "device_id": case.case_id,
                            "sensor_id": case.sensor_id, "unit": case.unit, "normal_range_celsius": [10.0, 40.0],
                            "maximum_observation_age_seconds": case.max_age_seconds,
                            "value": case.value, "observed_unix_seconds": time.time() - case.age_seconds, "sequence": 1}
                raw = json.dumps(observed, sort_keys=True).encode()
                (work / "equipment" / (case.case_id + ".json")).write_bytes(raw)
                request_id = uuid.uuid4().hex
                request = {"operation": "assess", "request_id": request_id, "device_id": case.case_id}
                accepted = await local_request(app.socket, request)
                retry = await local_request(app.socket, request)
                if accepted["request_id"] != retry["request_id"]:
                    raise RuntimeError("Retry changed the admitted diagnostic request")
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    result = await local_request(app.socket, {"operation": "status", "request_id": request_id})
                    if result["status"] != "running":
                        break
                    await asyncio.sleep(0.25)
                else:
                    raise TimeoutError("Application diagnostic did not settle within its allowance")
                record = app.store.get(request_id)
                evaluation = evaluate(case, record["record"], raw)
                case_receipt = {"case": asdict(case), "observation": observed, "request_id": request_id,
                                "job": record, "evaluation": evaluation, "retry_preserved_request": True}
                (output / (case.case_id + ".json")).write_text(json.dumps(case_receipt, indent=2) + "\n")
                summary = {"case_id": case.case_id, "passed": evaluation["passed"],
                           "job_status": record["record"]["status"],
                           "failure": record["record"]["failure"],
                           "model_calls": len(record["record"]["trace"].get("model_calls", [])),
                           "tool_calls": len(record["record"]["trace"].get("tool_events", []))}
                report["cases"].append(summary)
                print(json.dumps(summary), flush=True)
            if habitat is not None:
                await habitat.introduce_and_observe(app)
            report["final_health"] = await local_request(app.socket, {"operation": "health"})
            if report["final_health"]["status"] != "serving":
                raise RuntimeError("Application lost its inference service during the check")
        except Exception as exc:
            report["failure"] = type(exc).__name__ + ": " + str(exc)
        finally:
            try:
                await app.close()
                report["shutdown"] = app.receipt.get("shutdown")
            except Exception as exc:
                report["shutdown_failure"] = type(exc).__name__ + ": " + str(exc)
            report["socket_removed"] = not app.socket.exists()
            if habitat is not None:
                habitat.export(output, discovery)
            # Copy only this temporary runtime's bounded evidence, after close.
            shutil.copytree(work, output / "work-evidence", ignore=shutil.ignore_patterns("*.sock"))
            raw_evidence = b"".join(path.read_bytes() for path in (output / "work-evidence").glob("model-*.log"))
            report["model_api_key_found_in_server_log"] = app.inference.api_key.encode() in raw_evidence
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["model_child_peak_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    report["cases_passed"] = sum(case["passed"] for case in report["cases"])
    report["cases_total"] = len(cases)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["sources_unchanged_during_run"] = all(digest(root / "organism" / name) == expected
                                                for name, expected in report["source_hashes"].items())
    if not report["sources_unchanged_during_run"]:
        report["failure"] = report["failure"] or "Source files changed while the probe was running"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(output / "report.json"), "cases_passed": report["cases_passed"],
                      "cases_total": report["cases_total"], "failure": report["failure"],
                      "shutdown": report.get("shutdown")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-id", choices=[case.case_id for case in DEVELOPMENT_CASES])
    parser.add_argument("--with-discovery", action="store_true")
    args = parser.parse_args()
    cases = tuple(case for case in DEVELOPMENT_CASES if args.case_id is None or case.case_id == args.case_id)
    report = asyncio.run(run(Path(__file__).resolve().parents[1], args.output.resolve(), cases=cases,
                             with_discovery=args.with_discovery))
    return 1 if report["failure"] or report.get("shutdown_failure") else 0


if __name__ == "__main__":
    raise SystemExit(main())
