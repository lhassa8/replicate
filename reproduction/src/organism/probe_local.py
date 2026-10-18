"""Bounded local stock-model preflight; no deployment or physical device tools.

Run from the workspace with its existing .venv-local interpreter. The probe
owns one loopback-only llama.cpp process and stops it in a finally block.
"""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import time
import urllib.request
import uuid

from organism.diagnostics import (
    DEVELOPMENT_CASES, DiagnosticSession, evaluate_submission,
)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


async def probe_cases(base_url: str, alias: str, directory: Path) -> list[dict]:
    import httpx
    from strands import Agent, tool
    from strands.hooks import AfterModelCallEvent, BeforeModelCallEvent, HookProvider
    from strands.models import LlamaCppModel

    class Observer(HookProvider):
        def __init__(self):
            self.calls = []
            self.responses = []

        def register_hooks(self, registry, **kwargs):
            registry.add_callback(BeforeModelCallEvent, self.before)
            registry.add_callback(AfterModelCallEvent, self.after)

        def before(self, event):
            if len(self.calls) >= 4:
                event.cancel = "Development preflight model-call budget exhausted"
                return
            self.calls.append(deepcopy({
                "messages": event.agent.messages,
                "system_prompt": event.agent.system_prompt,
                "tools": event.agent.tool_registry.get_all_tool_specs(),
            }))

        def after(self, event):
            response = event.stop_response
            self.responses.append({
                "stop_reason": response.stop_reason if response else None,
                "message": deepcopy(response.message) if response else None,
                "exception": str(event.exception) if event.exception else None,
            })

    summaries = []
    for case in DEVELOPMENT_CASES:
        fixture = DiagnosticSession(case)
        observer = Observer()
        requests = []

        async def observe_request(request):
            expected = httpx.URL(base_url)
            if (
                request.url.host != expected.host
                or request.url.port != expected.port
                or request.url.scheme != "http"
            ):
                raise RuntimeError("Preflight request escaped the owned local endpoint")
            requests.append({
                "method": request.method, "url": str(request.url),
                "body": json.loads(request.content) if request.content else None,
            })

        @tool
        def inspect_device(device_id: str) -> dict:
            """Read a device's sensor capabilities and acceptable operating range."""
            return fixture.inspect_device(device_id)

        @tool
        def read_measurement(device_id: str, sensor_id: str) -> dict:
            """Read the specified sensor after inspecting its device capabilities."""
            return fixture.read_measurement(device_id, sensor_id)

        @tool
        def submit_assessment(
            device_id: str, status: str,
            temperature_celsius: float | None, evidence_ids: list[str],
        ) -> dict:
            """Record one assessment with observed capability and measurement IDs.

            Args:
                device_id: Exact device being assessed.
                status: normal, low, high, or unknown.
                temperature_celsius: Converted current reading, or null if stale.
                evidence_ids: Evidence IDs returned by both observation tools.
            """
            return fixture.submit_assessment(
                device_id, status, temperature_celsius, evidence_ids,
            )

        model = LlamaCppModel(
            base_url=base_url, timeout=75, model_id=alias,
            params={"temperature": 0, "max_tokens": 192},
        )
        await model.client.aclose()
        model.client = httpx.AsyncClient(
            base_url=base_url, timeout=httpx.Timeout(75),
            trust_env=False, event_hooks={"request": [observe_request]},
        )
        agent = Agent(
            model=model, tools=[inspect_device, read_measurement, submit_assessment],
            system_prompt=(
                "You are a read-only equipment diagnostic agent. Inspect the device, "
                "read its temperature sensor, then submit one evidence-backed assessment. "
                "Compare Celsius temperature against the stated normal range. Convert units "
                "when necessary. If the reading is older than the allowed age, report "
                "unknown with null temperature. Use only returned evidence IDs. After "
                "submission, end with a short acknowledgement. Do not invent observations."
            ),
            hooks=[observer], callback_handler=None, retry_strategy=None,
            context_manager=False, background_tasks=False, load_tools_from_directory=False,
        )
        started = time.monotonic()
        failure = None
        try:
            await asyncio.wait_for(
                agent.invoke_async(f"Assess the device {case.case_id}."),
                timeout=150,
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            await model.client.aclose()
        evaluation = evaluate_submission(case, fixture)
        if failure is not None:
            evaluation["passed"] = False
        receipt = {
            "case_id": case.case_id,
            "seconds": round(time.monotonic() - started, 3),
            "failure": failure, "evaluation": evaluation,
            "requests": requests, "model_calls": observer.calls,
            "model_responses": observer.responses, "tool_events": fixture.events,
            "submission": fixture.submission, "final_messages": agent.messages,
        }
        write_json(directory / f"{case.case_id}.json", receipt)
        summary = {k: receipt[k] for k in ("case_id", "seconds", "failure", "evaluation")}
        summary["model_calls"] = len(observer.calls)
        summary["http_requests"] = len(requests)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    return summaries


def run(
    root: Path, output: Path, *, case_runner=probe_cases,
    case_count=len(DEVELOPMENT_CASES),
    scope="macOS ARM64 CPU stock-model development preflight",
):
    if version("strands-agents") != "1.57.1":
        raise RuntimeError("This preflight is bound to the inspected Strands SDK 1.57.1")
    output.mkdir(parents=True, exist_ok=False)
    model = root / "models/granite-3b/granite-base-Q4_K_M.gguf"
    runtime = root / "tools/llama.cpp/b9850/llama-b9850/llama-server"
    expected = json.loads((model.parent / "manifest.json").read_text())["files"][0]
    actual_hash = digest_file(model)
    if actual_hash != expected["sha256"] or model.stat().st_size != expected["bytes"]:
        raise RuntimeError("Retained stock-model identity check failed")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    alias = "organism-phase0-" + uuid.uuid4().hex[:12]
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(runtime), "--model", str(model), "--alias", alias,
        "--host", "127.0.0.1", "--port", str(port), "--offline",
        "--ctx-size", "4096", "--parallel", "1", "--gpu-layers", "0",
        "--device", "none", "--no-kv-offload", "--no-op-offload",
        "--threads", "4", "--threads-batch", "4", "--poll", "0",
        "--poll-batch", "0", "--cache-ram", "0", "--jinja",
        "--chat-template-kwargs", '{"enable_thinking":false}',
        "--temp", "0", "--top-p", "1", "--top-k", "0", "--min-p", "0",
        "--repeat-penalty", "1.0",
    ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "qualification_passed": False,
        "model_path": str(model), "model_sha256": actual_hash,
        "model_bytes": model.stat().st_size,
        "runtime_path": str(runtime), "runtime_sha256": digest_file(runtime),
        "strands_version": version("strands-agents"), "command": command,
        "trained_for_organism": False, "adapter_loaded": False,
        "network_boundary": "loopback request guard and runtime offline flag; no OS-level egress audit",
        "cases": [],
    }
    write_json(output / "request.json", report)
    process = None
    started = time.monotonic()
    log = (output / "server.log").open("w")
    try:
        environment = {
            k: v for k, v in os.environ.items()
            if not k.startswith(("AWS_", "OTEL_", "LLAMA_", "HF_", "HUGGING"))
        }
        environment.update({
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "OTEL_SDK_DISABLED": "true",
        })
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        report["server_pid"] = process.pid
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 75
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Owned model server exited with {process.returncode}")
            try:
                with opener.open(base_url + "/health", timeout=1) as response:
                    if json.load(response).get("status") == "ok":
                        break
            except (OSError, ValueError):
                time.sleep(0.25)
        else:
            raise TimeoutError("Owned model server readiness timeout")
        report["ready_seconds"] = round(time.monotonic() - started, 3)
        with opener.open(base_url + "/props", timeout=3) as response:
            write_json(output / "server-properties.json", json.load(response))
        report["cases"] = asyncio.run(case_runner(base_url, alias, output))
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            report["server_exit_code"] = process.returncode
            report["owned_server_stopped"] = process.poll() is not None
        log.close()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        report["host_system"] = platform.system()
        # Linux reports KiB; Darwin reports bytes. Keep the old Darwin field
        # for consumers of earlier Mac development receipts.
        if platform.system() in ("Linux", "Darwin"):
            report["child_peak_rss_bytes"] = int(rss * (1024 if platform.system() == "Linux" else 1))
        if platform.system() == "Darwin":
            report["child_peak_rss_bytes_macos"] = int(rss)
        report["passed_development_cases"] = sum(x["evaluation"]["passed"] for x in report["cases"])
        report["total_development_cases"] = case_count
        write_json(output / "report.json", report)
        print(json.dumps({
            "report": str(output / "report.json"),
            "passed": report["passed_development_cases"],
            "total": report["total_development_cases"],
            "failure": report.get("failure"),
            "owned_server_stopped": report.get("owned_server_stopped", False),
        }), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    already_existed = output.exists()
    try:
        result = run(workspace, output)
    except Exception as exc:
        if output.is_dir() and not already_existed:
            write_json(output / "setup-failure.json", {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "scope": "preflight setup; no model-quality conclusion",
                "failure": f"{type(exc).__name__}: {exc}",
                "qualification_passed": False,
            })
        raise
    raise SystemExit(0 if not result.get("failure") else 1)
