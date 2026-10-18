"""Run the exact carried smoke archive against one actual Mac stock-model app."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import sys
import tempfile
import time
import uuid

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

import httpx

from .application import AgentApplication, local_request
from .inference_server import RuntimeConfig, digest
from .safe_archive import extract_archive


async def run(root, output, component, *, runtime_config=None, challenge_id=None):
    if platform.system() != "Darwin":
        raise ValueError("This development probe requires the Mac")
    output.mkdir(parents=True, exist_ok=False)
    if runtime_config is None:
        contract = json.loads((root / "organism/experiment-contract.draft.json").read_bytes())
        model = contract["model"]
        config = RuntimeConfig(root / "models/granite-3b/granite-base-Q4_K_M.gguf",
                               model["local_export_sha256"], model["local_export_bytes"],
                               root / "tools/llama.cpp/b9850/llama-b9850/llama-server",
                               contract["runtime"]["development_binary_sha256"],
                               uuid.uuid4().hex, "d" * 64, uuid.uuid4().hex, "e" * 64, False, False)
    else:
        if not isinstance(runtime_config, RuntimeConfig) or runtime_config.native:
            raise ValueError("An explicit smoke-probe runtime must be a local Mac configuration")
        config = runtime_config
    if challenge_id is not None and not re.fullmatch("[0-9a-f]{32}", challenge_id):
        raise ValueError("Invalid predeclared smoke challenge")
    packed = json.loads((component / "receipt.json").read_bytes())
    if digest(component / "self-test.tar.gz") != packed["archive_sha256"]:
        raise ValueError("Carried self-test archive changed")
    report = {"scope": "Exact carried public smoke worker and live Mac application; no native child or readiness admission",
              "started_at": datetime.now(timezone.utc).isoformat(), "model_sha256": config.model_sha256,
              "runtime_sha256": config.executable_sha256, "trained_for_organism": config.trained_for_organism,
              "native_execution_verified": False, "autonomous_reproduction_verified": False,
              "qualification_passed": False, "receiver_readiness_acknowledged": False,
              "checker_challenge_transport": "inherited_read_only_file_descriptor",
              "self_test_component": packed, "cases": [], "cases_total": 1, "cases_passed": 0,
              "rss_samples": [], "failure": None}
    if runtime_config is not None:
        report["runtime_binding"] = {
            "model_path": str(config.model), "model_sha256": config.model_sha256,
            "model_bytes": config.model_bytes, "runtime_path": str(config.executable),
            "runtime_sha256": config.executable_sha256, "release_id": config.release_id,
            "trained_for_organism": config.trained_for_organism,
        }
        from .probe_trained_tasks import monitor
    snapshots = output / "source-snapshot"
    snapshots.mkdir()
    files = {Path(__file__)}
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value and Path(value).parent == root / "organism" and Path(value).suffix == ".py":
            files.add(Path(value))
    report["source_hashes"] = {}
    for source in sorted(files):
        raw = source.read_bytes()
        (snapshots / source.name).write_bytes(raw)
        report["source_hashes"][source.name] = hashlib.sha256(raw).hexdigest()
    (output / "request.json").write_text(json.dumps(report, indent=2) + "\n")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="or-live-", dir="/private/tmp") as temporary:
        base = Path(temporary)
        work = base / "work"
        work.mkdir(mode=0o700)
        extracted = base / "self-test"
        report["extraction"] = extract_archive(component / "self-test.tar.gz", extracted, allowed_roots=("readiness.py",))
        worker = extracted / "readiness.py"
        if digest(worker) != packed["source_sha256"]:
            raise ValueError("Extracted smoke worker differs from its receipt")
        (snapshots / "carried-readiness.py").write_bytes(worker.read_bytes())
        app = AgentApplication(config, work)
        child, watcher = None, None
        try:
            report["initial_health"] = await app.start()
            report["server_pid"] = app.inference.process.pid
            if runtime_config is not None:
                watcher = asyncio.create_task(monitor(app, output, work, report))
            print(json.dumps({"application": "serving", "model_pid": report["server_pid"]}), flush=True)
            async with httpx.AsyncClient(trust_env=False, timeout=3) as anonymous:
                response = await anonymous.post(app.inference.base_url + "/v1/chat/completions", json={
                    "model": app.inference.alias, "messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 1})
                report["anonymous_inference_request_status"] = response.status_code
                if response.status_code != 401:
                    raise ValueError("Unauthenticated inference was not rejected")
            now = time.time()
            challenge = {"schema": "organism-readiness-challenge-v1",
                         "challenge_id": challenge_id if challenge_id is not None else uuid.uuid4().hex,
                         "device_id": config.device_id, "individual_id": config.individual_id,
                         "process_token": config.start_token, "release_id": config.release_id,
                         "model_path": str(config.model), "model_sha256": config.model_sha256, "model_bytes": config.model_bytes,
                         "runtime_path": str(config.executable), "runtime_sha256": config.executable_sha256,
                         "native": False, "issued_unix_seconds": now, "expires_unix_seconds": now + 240}
            challenge_file = output / "challenge.json"
            challenge_file.write_text(json.dumps(challenge, indent=2) + "\n")
            with (output / "checker-stdout.json").open("xb") as stdout, \
                    (output / "checker-stderr.log").open("xb") as stderr, challenge_file.open("rb") as stdin:
                child = await asyncio.create_subprocess_exec(
                    sys.executable, "-I", "-S", str(worker), "--work-directory", str(work),
                    "--challenge-stdin", cwd=extracted,
                    env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
                    stdin=stdin, stdout=stdout, stderr=stderr)
                report["checker_pid"] = child.pid
                print(json.dumps({"carried_checker": "running", "pid": child.pid,
                                  "challenge_id": challenge["challenge_id"]}), flush=True)
                try:
                    report["checker_return_code"] = await asyncio.wait_for(child.wait(), timeout=250)
                except asyncio.TimeoutError:
                    child.kill()
                    await child.wait()
                    raise RuntimeError("Carried checker exceeded its external process deadline")
            raw = (output / "checker-stdout.json").read_bytes()
            if len(raw) > 65536:
                raise ValueError("Checker output exceeds its bound")
            result = json.loads(raw)
            if (result["checker_sha256"] != packed["source_sha256"]
                    or result["challenge_id"] != challenge["challenge_id"]
                    or result["process_token"] != config.start_token
                    or result["qualification_passed"] is not False
                    or report["checker_return_code"] != (0 if result["passed"] else 1)):
                raise ValueError("Carried checker response identity or exit differs")
            report["smoke_result"] = result
            evidence = work / "readiness" / config.start_token
            if (evidence / "job.json").exists() and (evidence / "observation.json").exists():
                job = json.loads((evidence / "job.json").read_bytes())
                observation = json.loads((evidence / "observation.json").read_bytes())
                case_id = observation["device_id"]
                receipt = {"observation": observation, "request_id": challenge["challenge_id"], "job": job,
                           "evaluation": {"passed": result["passed"]}}
                (output / (case_id + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
                report["cases"] = [{"case_id": case_id, "passed": result["passed"],
                                    "job_status": job["record"]["status"]}]
            report["cases_passed"] = int(result["passed"])
            report["final_health"] = await local_request(app.socket, {"operation": "health"})
            print(json.dumps({"public_smoke_passed": result["passed"], "failure": result["failure"]}), flush=True)
        except Exception as exc:
            report["failure"] = type(exc).__name__ + ": " + str(exc)
        finally:
            if watcher is not None:
                if watcher.done() and not watcher.cancelled() and watcher.exception() is not None:
                    report["failure"] = report["failure"] or "Resource monitor failed: " + str(watcher.exception())
                if app.inference.failure:
                    report["failure"] = report["failure"] or app.inference.failure
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            if child is not None and child.returncode is None:
                child.kill()
                await child.wait()
            try:
                await app.close()
                report["shutdown"] = app.receipt.get("shutdown")
            except Exception as exc:
                report["shutdown_failure"] = type(exc).__name__ + ": " + str(exc)
            report["socket_removed"] = not app.socket.exists()
            shutil.copytree(work, output / "work-evidence", ignore=shutil.ignore_patterns("*.sock"))
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["maximum_child_peak_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if runtime_config is not None:
        report["peak_sampled_server_rss_bytes"] = max(
            (row["rss_bytes"] for row in report["rss_samples"]), default=0)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["sources_unchanged_during_run"] = all(digest(root / "organism" / name) == expected
                                                for name, expected in report["source_hashes"].items())
    if not report["sources_unchanged_during_run"]:
        report["failure"] = report["failure"] or "Application sources changed during the run"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(output / "report.json"), "failure": report["failure"],
                      "smoke_passed": report.get("smoke_result", {}).get("passed")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--component", required=True, type=Path)
    args = parser.parse_args()
    report = asyncio.run(run(Path(__file__).resolve().parents[1], args.output.resolve(), args.component.resolve()))
    return 1 if report["failure"] or report.get("shutdown_failure") else 0


if __name__ == "__main__":
    raise SystemExit(main())
