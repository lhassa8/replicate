"""Supervise one owned Mac memory-loading probe; never train or launch AWS."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

from .training_inputs import load, require, sha256


MAX_RSS_BYTES = 16 * 1024**3
MAX_SECONDS = 1800
POLL_SECONDS = 1


def observe_rss(pid):
    result = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(pid)],
                            check=True, capture_output=True, text=True, timeout=5)
    raw = result.stdout.strip()
    require(re.fullmatch(r"[0-9]+", raw) is not None, "Owned process memory observation is unavailable")
    return int(raw) * 1024


def stop_owned(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
    require(child.poll() is not None, "Owned memory probe has not terminated")
    return child.returncode


def watch(child, persist):
    started, peak, failure = time.monotonic(), 0, None
    observations = 0
    while child.poll() is None:
        try:
            rss = observe_rss(child.pid)
        except Exception as exc:
            if child.poll() is not None:
                break
            failure = "Memory observation failed while the owned process remained live: " + type(exc).__name__
            break
        observations += 1
        peak = max(peak, rss)
        elapsed = time.monotonic() - started
        persist({"pid": child.pid, "observations": observations, "peak_observed_rss_bytes": peak,
                 "elapsed_seconds": round(elapsed, 3), "running": True})
        if rss > MAX_RSS_BYTES:
            failure = "Observed memory exceeded the 16 GiB stop threshold"
            break
        if elapsed >= MAX_SECONDS:
            failure = "Owned process exceeded the thirty-minute wall-clock limit"
            break
        time.sleep(POLL_SECONDS)
    if failure:
        stop_owned(child)
    else:
        child.wait(timeout=10)
    return {"pid": child.pid, "observations": observations, "peak_observed_rss_bytes": peak,
            "elapsed_seconds": round(time.monotonic() - started, 3), "running": False,
            "exit_code": child.returncode, "failure": failure}


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(platform.system() == "Darwin" and platform.machine() == "arm64", "Declared Mac required")
    memory = int(subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], check=True,
                                capture_output=True, text=True, timeout=5).stdout.strip())
    pressure = subprocess.run(["/usr/bin/memory_pressure", "-Q"], check=True,
                              capture_output=True, text=True, timeout=5).stdout
    match = re.search(r"System-wide memory free percentage: ([0-9]+)%", pressure)
    require(memory >= 32 * 1024**3 and match is not None and int(match.group(1)) >= 60,
            "Insufficient observed Mac memory for this development probe")
    # Check the observer before allocating a model or opening a source stream.
    observe_rss(os.getpid())
    output.mkdir(parents=True, exist_ok=False)
    sources = ("verify_training_memory.py", "probe_training_memory.py", "verify_training_stream.py",
               "training_inputs.py", "training_source.py")
    source_hashes = {}
    snapshot = output / "sources"
    snapshot.mkdir()
    for name in sources:
        path = root / "organism" / name
        (snapshot / name).write_bytes(path.read_bytes())
        source_hashes[name] = sha256(path)
    receipt = {
        "schema": "organism-memory-probe-supervision-v1", "started_at": datetime.now(timezone.utc).isoformat(),
        "physical_memory_bytes": memory, "initial_memory_free_percentage": int(match.group(1)),
        "maximum_observed_rss_bytes": MAX_RSS_BYTES, "maximum_seconds": MAX_SECONDS,
        "poll_seconds": POLL_SECONDS, "kernel_memory_cap": False, "source_hashes": source_hashes,
        "failure": None, "passed": False, "qualification_passed": False, "cloud_resources_created": False,
    }

    def persist(state=None):
        if state is not None:
            receipt["observation"] = state
        temporary = output / ".supervisor.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(output / "supervisor.json")

    interpreter = root / "organism/.venv-training/bin/python"
    code = ("import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
            "from organism.verify_training_memory import run; run(Path(sys.argv[1]),Path(sys.argv[2]))")
    environment = {"PATH": str(interpreter.parent) + ":/usr/bin:/bin:/usr/sbin:/sbin",
                   "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                   "AWS_EC2_METADATA_DISABLED": "true", "OTEL_SDK_DISABLED": "true", "WANDB_DISABLED": "true",
                   "TOKENIZERS_PARALLELISM": "false"}
    child = None
    persist()
    try:
        with (output / "worker.log").open("x") as log:
            child = subprocess.Popen(
                [str(interpreter), "-I", "-c", code, str(root), str(output / "worker")],
                cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt["observation"] = watch(child, persist)
        require(receipt["observation"]["failure"] is None and child.returncode == 0,
                "The owned memory probe did not complete successfully")
        worker = load(output / "worker/report.json")
        require(worker.get("passed") is True and worker.get("native_model_loaded_on_mac") is True
                and worker.get("optimizer_steps") == 0, "Worker evidence does not prove a successful non-training probe")
        for name, expected in source_hashes.items():
            require(sha256(root / "organism" / name) == expected, "A probe source changed during execution")
        receipt["worker_report_sha256"] = sha256(output / "worker/report.json")
        receipt["passed"] = True
    except BaseException as exc:
        receipt["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        if child is not None:
            try:
                receipt["final_child_exit_code"] = stop_owned(child)
                receipt["child_termination_confirmed"] = True
            except Exception as exc:
                receipt["child_termination_confirmed"] = False
                receipt["cleanup_failure"] = type(exc).__name__ + ": " + str(exc)
                receipt["passed"] = False
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        persist()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(Path(__file__).resolve().parents[1], args.output)
    print(json.dumps(result, indent=2))
