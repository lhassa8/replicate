"""Collect bounded complete-vocabulary comparisons on an export host."""

import asyncio
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shlex
import uuid

from .export_math import ordered_native_log_probabilities, require
from .gguf_export import identity
from .inference_server import InferenceServer, RuntimeConfig
from .training_inputs import sha256


async def serving_rows(tools, output, path, label, prompts, vocabulary, *,
                       reference_precision=False, synthetic=True, trained=False,
                       total_seconds=180, request_seconds=60):
    import httpx
    import numpy as np

    require(type(synthetic) is bool and type(trained) is bool and not (synthetic and trained),
            "Synthetic export cannot be labeled trained")
    require(0 < request_seconds <= 600 and request_seconds <= total_seconds <= 1900,
            "Invalid export inference deadline")
    require(type(vocabulary) is int and 0 < vocabulary <= 131072
            and 1 <= len(prompts) <= 16, "Export comparison inventory exceeds its bound")
    executable = tools.path(tools.manifest["server"])
    expected_executable = tools.manifest["files"][tools.manifest["server"]]["sha256"]
    require(sha256(executable) == expected_executable, "Pinned export server changed")
    directory = output / label
    directory.mkdir()
    precision = {"cache_type": "runtime default", "reference_precision_control": reference_precision}
    if reference_precision:
        launcher = directory / "f32-cache-launcher.sh"
        text = ("#!/bin/sh\nexec " + shlex.quote(str(executable))
                + ' --cache-type-k f32 --cache-type-v f32 --flash-attn off "$@"\n')
        with launcher.open("x") as target:
            target.write(text)
        launcher.chmod(0o700)
        precision.update(cache_type="f32", flash_attention="off", binary=str(executable),
                         binary_sha256=expected_executable, launcher=identity(launcher), launcher_text=text)
        executable, expected_executable = launcher, sha256(launcher)
    config = RuntimeConfig(
        path, sha256(path), path.stat().st_size, executable, expected_executable,
        uuid.uuid4().hex, "d" * 64, uuid.uuid4().hex, "e" * 64, trained, platform.system() == "Linux",
        tuple(tools.path(name) for name in tools.manifest["library_paths"]),
    )
    server = InferenceServer(config, directory)
    result = {"synthetic_model": synthetic, "trained_model": trained, "requests": [], "failure": None,
              "precision": precision, "execution_context": "export host, not device qualification",
              "total_seconds_limit": total_seconds, "request_seconds_limit": request_seconds,
              "qualification_passed": False}
    rows = []
    try:
        async with asyncio.timeout(total_seconds):
            result["server_start"] = await server.start()
            async with httpx.AsyncClient(base_url=server.base_url, headers=server.headers,
                                         trust_env=False, timeout=request_seconds) as client:
                for index, tokens in enumerate(prompts):
                    require(isinstance(tokens, list) and 0 < len(tokens) < 4096
                            and all(type(token) is int and 0 <= token < vocabulary for token in tokens),
                            "Invalid export comparison prompt")
                    request = {"prompt": tokens, "n_predict": 1, "n_probs": vocabulary,
                               "temperature": 0, "cache_prompt": False, "return_tokens": True,
                               "post_sampling_probs": False, "repeat_penalty": 1, "seed": 20261014,
                               "grammar": 'root ::= "a"'}
                    item = {"request": request}
                    result["requests"].append(item)
                    raw = bytearray()
                    async with client.stream("POST", "/completion", json=request) as response:
                        response.raise_for_status()
                        async for block in response.aiter_bytes():
                            raw.extend(block)
                            require(len(raw) <= 24 * 1024**2, "Export probability response exceeded its bound")
                    saved = directory / f"response-{index + 1}.json.gz"
                    with saved.open("xb") as target, gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as zipped:
                        zipped.write(raw)
                    body = json.loads(raw)
                    probabilities = body["completion_probabilities"]
                    require(len(probabilities) == 1 and body["tokens_evaluated"] == len(tokens)
                            and body["content"] == "a" and len(body["tokens"]) == 1,
                            "Serving prompt or output token count changed")
                    values, accounting = ordered_native_log_probabilities(probabilities[0]["top_logprobs"], vocabulary)
                    rows.append(values)
                    item.update({"response_file": saved.name, "response_bytes": len(raw),
                                 "response_sha256": hashlib.sha256(raw).hexdigest(),
                                 "generated_tokens": body["tokens"], "complete_vocabulary": len(rows[-1]),
                                 "native_probability_accounting": accounting})
    except BaseException as exc:
        result["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        result["shutdown"] = await server.stop()
        (directory / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    require(result["shutdown"]["owned_server_stopped"], "Export inference server remains active")
    return np.stack(rows), result
