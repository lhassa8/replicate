"""Development inference with the admitted adapter over the existing stock Q4.

This is not the frozen merged-Q4 comparison or device qualification. It checks
conversion tensors and exercises the real completed pilot without downloading
full native base weights or modifying a release capsule.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tarfile
import time
import uuid

from .adapter_export import admit_training_export
from .audit_trained_adapter_probe import matches_served_target
from .gguf_export import run_command
from .inference_server import InferenceServer, RuntimeConfig
from .probe_model_export import COMMIT, DEPENDENCY, identity, source_pin
from .training_inputs import check_versions, offline_environment, require, sha256


LIMITS = {"conversion_seconds": 180, "request_seconds": 120, "inference_seconds": 1800,
          "new_artifact_bytes": 1024**3, "adapter_bytes": 256 * 1024**2,
          "response_bytes": 1024**2, "server_sampled_rss_bytes": 6 * 1024**3,
          "generation_tokens": 192}
ROLES = ("diagnostic", "equipment", "equipment_inherited", "reproduction")
PROJECTIONS = {
    ("self_attn", "q_proj"): "attn_q", ("self_attn", "k_proj"): "attn_k",
    ("self_attn", "v_proj"): "attn_v", ("self_attn", "o_proj"): "attn_output",
    ("mlp", "gate_proj"): "ffn_gate", ("mlp", "up_proj"): "ffn_up", ("mlp", "down_proj"): "ffn_down",
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def select_rows(rows):
    """First tool and final decision for each role, before seeing any output."""
    selected = []
    for role in ROLES:
        for kind in ("tool", "final"):
            row = next((value for value in rows if value["role"] == role and value["target_kind"] == kind), None)
            require(row is not None, "Missing development role or target kind")
            selected.append(row)
    return selected


def tensor_destination(name):
    match = re.fullmatch(
        r"base_model\.model\.model\.layers\.([0-9]+)\.(self_attn|mlp)\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_([AB])\.weight", name)
    require(match is not None and (match[2], match[3]) in PROJECTIONS, "Unexpected admitted adapter tensor name")
    return f"blk.{int(match[1])}.{PROJECTIONS[match[2], match[3]]}.weight.lora_{match[4].lower()}", match[3], match[4]


def rotary_rows(row_count, heads):
    require(type(row_count) is int and type(heads) is int and heads > 0 and row_count % (2 * heads) == 0,
            "Invalid rotary row grouping")
    width = row_count // heads
    return [head * width + half * (width // 2) + offset
            for head in range(heads) for offset in range(width // 2) for half in (0, 1)]


def inspect_converted(path, source_weights, config, expected_alpha):
    """Compare all F32 values independently of the converter's tensor methods."""
    import gguf
    import numpy as np
    from safetensors import safe_open

    reader = gguf.GGUFReader(path)
    for name, value in {"general.type": "adapter", "general.architecture": "granite",
                        "adapter.type": "lora", "adapter.lora.alpha": expected_alpha}.items():
        require(name in reader.fields and reader.fields[name].contents() == value,
                "Converted adapter metadata differs: " + name)
    actual = {tensor.name: tensor for tensor in reader.tensors}
    require(len(actual) == len(reader.tensors), "Duplicate converted tensor")
    checked, parameters = [], 0
    with safe_open(source_weights, framework="numpy") as source:
        require(len(source.keys()) == len(actual) == config["num_hidden_layers"] * 7 * 2,
                "Converted adapter tensor count differs")
        for name in source.keys():
            destination, projection, kind = tensor_destination(name)
            require(destination in actual, "Converted tensor is missing")
            expected = source.get_tensor(name)
            if kind == "B" and projection in ("q_proj", "k_proj"):
                heads = config["num_attention_heads" if projection == "q_proj" else "num_key_value_heads"]
                expected = expected[rotary_rows(expected.shape[0], heads)]
            tensor = actual[destination]
            require(tensor.tensor_type == gguf.GGMLQuantizationType.F32
                    and tuple(tensor.shape[::-1]) == expected.shape
                    and np.array_equal(tensor.data, expected), "Converted adapter tensor values differ: " + name)
            parameters += expected.size
            checked.append({"source": name, "destination": destination, "shape": list(expected.shape),
                            "sha256": hashlib.sha256(expected.tobytes()).hexdigest()})
    return {"tensor_count": len(checked), "parameters": parameters, "all_f32_values_equal": True,
            "rotary_permutation_checked": True, "tensors": checked, "qualification_passed": False}


async def sample_server(server, samples):
    while server.process is None or server.process.returncode is None:
        if server.process is not None:
            process = await asyncio.create_subprocess_exec(
                "/bin/ps", "-p", str(server.process.pid), "-o", "rss=",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=3)
            if process.returncode == 0 and stdout.strip().isdigit():
                value = int(stdout.strip()) * 1024
                samples.append({"monotonic_seconds": time.monotonic(), "rss_bytes": value})
                if value > LIMITS["server_sampled_rss_bytes"]:
                    server.failure = "Development server exceeded its sampled RSS allowance"
                    if server.process.returncode is None:
                        server.process.terminate()
                    return
        await asyncio.sleep(1)


async def serve_arm(root, output, arm, adapter, model, rows):
    import httpx
    directory = output / arm
    directory.mkdir()
    binary = root / "tools/llama.cpp/b9850/llama-b9850/llama-server"
    scale = 0 if arm == "stock-control" else 1
    launcher = directory / "launcher.sh"
    launcher.write_text("#!/bin/sh\nexec " + shlex.quote(str(binary))
                        + " --lora-scaled " + shlex.quote(str(adapter) + ":" + str(scale)) + ' "$@"\n')
    launcher.chmod(0o700)
    config = RuntimeConfig(model, sha256(model), model.stat().st_size, launcher, sha256(launcher),
                           uuid.uuid4().hex, "d" * 64, uuid.uuid4().hex, "e" * 64, scale == 1, False)
    server = InferenceServer(config, directory)
    record = {"arm": arm, "adapter_scale": scale, "requests": [], "rss_samples": [],
              "failure": None, "qualification_passed": False}
    monitor = None
    try:
        record["server_start"] = await server.start()
        monitor = asyncio.create_task(sample_server(server, record["rss_samples"]))
        adapters = await server.json_get("/lora-adapters")
        require(len(adapters) == 1 and adapters[0]["id"] == 0 and adapters[0]["path"] == str(adapter)
                and adapters[0]["scale"] == scale, "Loaded adapter or effective scale differs")
        record["loaded_adapters"] = adapters
        async with httpx.AsyncClient(base_url=server.base_url, headers=server.headers, trust_env=False,
                                    timeout=LIMITS["request_seconds"]) as client:
            for index, row in enumerate(rows):
                require(not server.failure, server.failure)
                prompt = row["input_ids"][:row["prompt_tokens"]]
                request = {"prompt": prompt, "n_predict": LIMITS["generation_tokens"], "temperature": 0,
                           "cache_prompt": False, "return_tokens": True, "seed": 20261015,
                           "repeat_penalty": 1, "top_k": 0, "top_p": 1, "min_p": 0,
                           "lora": [{"id": 0, "scale": scale}]}
                started = time.monotonic()
                raw = bytearray()
                async with asyncio.timeout(LIMITS["request_seconds"]):
                    async with client.stream("POST", "/completion", json=request) as response:
                        response.raise_for_status()
                        async for block in response.aiter_bytes():
                            raw.extend(block)
                            require(len(raw) <= LIMITS["response_bytes"], "Development response exceeds its bound")
                body = json.loads(raw)
                require(body["tokens_evaluated"] == len(prompt), "Development prompt token count changed")
                saved = directory / f"response-{index + 1:02d}.json"
                saved.write_bytes(raw)
                expected = row["input_ids"][row["prompt_tokens"]:]
                exact = matches_served_target(expected, body["tokens"], body["stop_type"], 100257)
                record["requests"].append({
                    "case_id": row["case_id"], "decision_index": row["decision_index"],
                    "role": row["role"], "target_kind": row["target_kind"],
                    "request": request, "response": identity(saved), "response_file": saved.name,
                    "expected_tokens": expected, "exact_target_tokens": exact,
                    "elapsed_seconds": time.monotonic() - started,
                })
                write(directory / "progress.json", record)
                print(json.dumps({"arm": arm, "decision": index + 1, "of": len(rows), "exact_target": exact}), flush=True)
        require(await server.json_get("/lora-adapters") == adapters, "Loaded adapters changed during inference")
    except BaseException as exc:
        record["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        record["shutdown"] = await server.stop()
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        record["peak_sampled_rss_bytes"] = max((sample["rss_bytes"] for sample in record["rss_samples"]), default=None)
        write(directory / "report.json", record)
    require(record["shutdown"]["owned_server_stopped"], "Development server shutdown is unverified")
    return record


async def run(output):
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    require(shutil.disk_usage(output).free >= 2 * 1024**3, "Insufficient local probe space")
    offline_environment()
    worker = root / "organism/evidence/admitted-l4-execution-20261015-01"
    report = {"schema": "organism-local-trained-adapter-probe-v1",
              "started_at": datetime.now(timezone.utc).isoformat(), "limits": LIMITS,
              "passed": False, "failure": None, "commands": [], "arms": {},
              "evaluation_scope": "Eight teacher-forced development decisions; no tools executed",
              "score_definition": "Exact target before EOS, with EOS stop; post-EOS template newline excluded",
              "deployment_format": "Existing stock Q4_K_M plus separate F32 LoRA",
              "frozen_merged_export_comparison": False, "native_linux_execution": False,
              "training_performed": False, "qualification_passed": False,
              "probe_sources": {name: identity(Path(__file__).with_name(name)) for name in (
                  "probe_trained_adapter.py", "audit_trained_adapter_probe.py", "adapter_export.py",
                  "inference_server.py", "runtime_environment.py", "gguf_export.py", "probe_model_export.py")}}
    write(output / "request-before-work.json", report)
    began = time.monotonic()
    try:
        report["versions"] = {**check_versions(), "numpy": version("numpy")}
        require(report["versions"]["numpy"] == "2.5.4", "Development NumPy version changed")
        admitted = admit_training_export(worker / "original-workspace", worker / "returned/workload")
        report["training_admission"] = admitted["receipt"]
        rows = select_rows([json.loads(line) for line in admitted["paths"]["tokens_validation"].read_text().splitlines()])
        plan = {"selection": "First tool and final row for each role in fixed role order",
                "source": identity(admitted["paths"]["tokens_validation"]), "rows": rows,
                "limits": LIMITS, "qualification_cases_accessed": False}
        write(output / "development-plan.json", plan)
        report["development_plan"] = identity(output / "development-plan.json")
        source = root / ("tools/export/llama.cpp-" + COMMIT)
        report["tool_inputs"] = source_pin(root, source, output)
        converter = source / "convert_lora_to_gguf.py"
        with tarfile.open(root / "tools/export/source.tar.gz") as archive:
            member = archive.getmember(source.name + "/convert_lora_to_gguf.py")
            require(member.isfile() and converter.read_bytes() == archive.extractfile(member).read(),
                    "Adapter converter differs from retained source")
        report["adapter_converter"] = identity(converter)
        config = output / "base-config"
        config.mkdir()
        shutil.copyfile(admitted["paths"]["base_config"], config / "config.json")
        converted = output / "trained-adapter-f32.gguf"
        wrapper = """
import pathlib, runpy, sys
dependency, script = (pathlib.Path(value).resolve() for value in sys.argv[1:3])
sys.argv = sys.argv[2:]
sys.path[:0] = [str(dependency), str(script.parent)]
def guard(event, arguments):
    if event.startswith(('socket.', 'subprocess.', 'os.exec', 'os.posix_spawn')) or event == 'os.system':
        raise RuntimeError('Offline adapter conversion cannot use network or child processes: ' + event)
sys.addaudithook(guard)
runpy.run_path(str(script), run_name='__main__')
"""
        run_command([sys.executable, "-I", "-c", wrapper, str(root / DEPENDENCY / "site-packages"), str(converter),
                     str(admitted["adapter_directory"]), "--base", str(config),
                     "--outfile", str(converted), "--outtype", "f32"],
                    output / "conversion.log", report["commands"], timeout_seconds=LIMITS["conversion_seconds"],
                    output_path=converted, maximum_output_bytes=LIMITS["adapter_bytes"])
        sys.path.insert(0, str(source / "gguf-py"))
        report["converted_adapter"] = identity(converted)
        audit = inspect_converted(converted, admitted["adapter_directory"] / "adapter_model.safetensors",
                                  admitted["config"], admitted["recipe"]["adapter"]["lora_alpha"])
        write(output / "tensor-audit.json", audit)
        report["tensor_audit"] = identity(output / "tensor-audit.json")
        require(audit["parameters"] == admitted["receipt"]["adapter"]["parameters"], "Adapter parameter count changed")
        model = root / "models/granite-3b/granite-base-Q4_K_M.gguf"
        require(sha256(model) == "17e29b1449d2d036155c82e9b32c0e2b51f28b4b2312d1d12ddc3f49ff2e07ee",
                "Stock model identity changed")
        report["stock_model"] = identity(model)
        report["total_stored_parameters"] = admitted["receipt"]["base_parameters"] + audit["parameters"]
        write(output / "request.json", report)
        async with asyncio.timeout(LIMITS["inference_seconds"]):
            for arm in ("stock-control", "trained-adapter"):
                report["arms"][arm] = await serve_arm(root, output, arm, converted, model, rows)
        require(admit_training_export(worker / "original-workspace", worker / "returned/workload")["receipt"]
                == report["training_admission"], "Training inputs changed during development inference")
        require(identity(model) == report["stock_model"] and identity(converted) == report["converted_adapter"],
                "Inference inputs changed")
        require(sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
                <= LIMITS["new_artifact_bytes"], "Development artifacts exceeded their allowance")
        report["passed"] = True
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = time.monotonic() - began
        write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(run(args.output))
