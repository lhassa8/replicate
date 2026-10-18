"""Exercise merge, conversion and quantization on a small synthetic Granite.

No optimizer, pretrained inference, qualification cases or cloud resources.
The test tensors are deterministic, not a trained candidate or release.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
import zipfile

from .export_math import (
    check_merge, compare_native_probabilities, compare_probabilities,
    log_probabilities, require,
)
from .export_serving import serving_rows
from .gguf_export import (Toolchain, bind_toolchain, convert_f32, inspect_gguf,
                          quantize_q4, tokenizer_fingerprint)
from .training_inputs import check_versions, load, offline_environment, sha256


COMMIT = "4f31eedb0ccf546b7e8d6bb243b170f12522f54d"
DEPENDENCY = "organism/vendor/export-probe-sentencepiece-0.2.1"
DEPENDENCY_WHEEL_SHA256 = "d9381351182ff9888cc80e41c632e7e274b106f450de33d67a9e8f6043da6f76"
LIMITS = {"merge_maximum_logit_error": 0.0002, "f32_maximum_log_probability_error": 0.002,
          "quantized_maximum_mean_kl": 0.10, "minimum_adapter_logit_effect": 0.01,
          "maximum_work_bytes": 1536 * 1024**2, "command_seconds": 180,
          "response_bytes": 24 * 1024**2}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def source_pin(root, source, output):
    archive_path = root / "tools/export/source.tar.gz"
    selected = [source / "convert_hf_to_gguf.py", source / "LICENSE"]
    selected += list((source / "conversion").rglob("*.py"))
    selected += list((source / "gguf-py/gguf").rglob("*.py"))
    selected = sorted(path for path in selected if "__pycache__" not in path.parts)
    files = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for path in selected:
            name = str(path.relative_to(source))
            member = archive.getmember(source.name + "/" + name)
            require(member.isfile() and member.size <= 2 * 1024**2, "Converter source member differs")
            raw = path.read_bytes()
            require(raw == archive.extractfile(member).read(), "Converter source changed from retained archive: " + name)
            files[str(path.relative_to(root))] = identity(path)
    binary_root = root / "tools/llama.cpp/b9850/llama-b9850"
    for path in sorted([binary_root / "llama-server", binary_root / "llama-quantize",
                        *binary_root.glob("*.dylib")]):
        files[str(path.relative_to(root))] = identity(path)
    dependency = root / DEPENDENCY
    dependency_receipt = load(dependency / "receipt.json")
    wheel_path = dependency / dependency_receipt["wheel_file"]
    require(sha256(wheel_path) == DEPENDENCY_WHEEL_SHA256, "Export dependency wheel changed")
    files[str(wheel_path.relative_to(root))] = identity(wheel_path)
    with zipfile.ZipFile(wheel_path) as wheel:
        for name, expected in dependency_receipt["files"].items():
            path = dependency / "site-packages" / name
            require(path.resolve().is_relative_to(dependency.resolve())
                    and identity(path) == expected and path.read_bytes() == wheel.read(name),
                    "Extracted export dependency differs from its pinned wheel")
            files[str(path.relative_to(root))] = expected
    files[str((dependency / "receipt.json").relative_to(root))] = identity(dependency / "receipt.json")
    receipt = {"declared_commit": COMMIT, "source_archive": identity(archive_path), "files": files,
               "source_members_equal_retained_archive": True,
               "upstream_commit_independently_refetched": False}
    write(output / "tool-inputs.json", receipt)
    return receipt


def work_bytes(path):
    size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    require(size <= LIMITS["maximum_work_bytes"], "Synthetic export exceeded its disk allowance")
    return size


async def run(root, output):
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, GraniteConfig, GraniteForCausalLM

    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    require(shutil.disk_usage("/private/tmp").free >= 3 * 1024**3, "Insufficient space for the bounded export fixture")
    workspace = Path(tempfile.mkdtemp(prefix="organism-export-probe-", dir="/private/tmp"))
    report = {"schema": "organism-synthetic-export-probe-v3", "started_at": datetime.now(timezone.utc).isoformat(),
              "workspace": str(workspace), "passed": False, "limits": LIMITS, "commands": [], "failure": None,
              "fixture_seed": 20261014, "adapter_seed": 271828, "optimizer_steps": 0,
              "training_performed": False, "pretrained_inference": False, "native_linux_execution": False,
              "qualification_passed": False,
              "f32_comparison_runtime": "F32 K/V caches; flash attention disabled",
              "quantized_comparison_runtime": "Deployment runtime defaults",
              "probe_sources": {name: identity(root / "organism" / name) for name in (
                  "probe_model_export.py", "export_math.py", "adapter_export.py", "inference_server.py",
                  "runtime_environment.py", "training_inputs.py", "gguf_export.py", "export_serving.py")}}
    write(output / "request.json", report)
    began = time.monotonic()
    try:
        offline_environment()
        report["versions"] = check_versions()
        report["versions"]["numpy"] = version("numpy")
        torch.set_num_threads(4)
        source = root / ("tools/export/llama.cpp-" + COMMIT)
        pinned = source_pin(root, source, output)
        tool_manifest = bind_toolchain(
            pinned, source_root=str(source.relative_to(root)), source_archive="tools/export/source.tar.gz",
            converter=str((source / "convert_hf_to_gguf.py").relative_to(root)),
            quantizer="tools/llama.cpp/b9850/llama-b9850/llama-quantize",
            server="tools/llama.cpp/b9850/llama-b9850/llama-server",
            dependency_root=DEPENDENCY + "/site-packages",
            symlinks={name: os.readlink(root / name) for name in pinned["files"] if (root / name).is_symlink()},
        )
        write(output / "toolchain.json", tool_manifest)
        tools = Toolchain(root, tool_manifest)
        report["toolchain"] = tools.verify()
        gguf = tools.import_gguf()
        experiment = load(root / "organism/experiment-contract.draft.json")
        stock = root / "models/granite-3b/granite-base-Q4_K_M.gguf"
        require(sha256(stock) == experiment["model"]["local_export_sha256"], "Pinned stock export changed")
        token_fields = tokenizer_fingerprint(stock)
        configuration = load(root / "organism/evidence/source-check-20261013-01/config.json")
        configuration.update(hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                             dtype="float32", torch_dtype="float32", use_cache=False)
        tokenizer_path = root / "organism/evidence/tokenizer-source-20261013-01"
        tokenizer_receipt = load(tokenizer_path / "receipt.json")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        prompts = [[configuration["bos_token_id"]] + tokenizer.encode(text, add_special_tokens=False)
                   for text in ("Hello.", "Read the observation and report the temperature in C.")]
        prompts.append([configuration["bos_token_id"], *range(10, 73)])
        report["prompts"] = prompts
        write(output / "fixture-config.json", configuration)
        torch.manual_seed(report["fixture_seed"])
        base = GraniteForCausalLM(GraniteConfig(**configuration)).float().eval()
        report["fixture_base_parameters"] = sum(value.numel() for value in base.parameters())
        require(report["fixture_base_parameters"] < 60_000_000, "Synthetic model exceeded its declared size")

        def logits(model):
            with torch.inference_mode():
                return np.stack([model(torch.tensor([tokens]), use_cache=False).logits[0, -1].float().numpy().copy()
                                 for tokens in prompts])

        def copy_tokenizer(directory):
            for item in tokenizer_receipt["files"]:
                name = item["name"]
                path = tokenizer_path / name
                require(sha256(path) == item["sha256"], "Pinned tokenizer file changed")
                if name != "config.json":
                    shutil.copyfile(path, directory / name)
            return {str(path.relative_to(directory)): identity(path) for path in directory.iterdir() if path.is_file()}

        def save_model(model, directory):
            model.save_pretrained(directory, safe_serialization=True, max_shard_size="1GB")
            return copy_tokenizer(directory)

        base_dir, adapter_dir, merged_dir = (workspace / name for name in ("base", "adapter", "merged"))
        base_logits = logits(base)
        report["base_files"] = save_model(base, base_dir)
        recipe = load(root / "organism/training-recipe.current-v3.draft.json")
        active = get_peft_model(base, LoraConfig(**recipe["adapter"])).eval()
        generator = torch.Generator(device="cpu").manual_seed(report["adapter_seed"])
        adapter_count = 0
        with torch.no_grad():
            for name, value in active.named_parameters():
                if ".lora_B." in name:
                    value.copy_(torch.randn(value.shape, generator=generator) * 0.05)
                if ".lora_" in name:
                    adapter_count += value.numel()
        report["fixture_adapter_parameters"] = adapter_count
        report["adapter_value_origin"] = "Fixed seeded random test tensors; no loss or optimizer"
        active_logits = logits(active)
        effect = float(np.max(np.abs(base_logits.astype(np.float64) - active_logits.astype(np.float64))))
        require(effect >= LIMITS["minimum_adapter_logit_effect"], "Synthetic adapter had no measurable effect")
        report["maximum_adapter_logit_effect"] = effect
        active.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
        report["adapter_files"] = {path.name: identity(path) for path in adapter_dir.iterdir() if path.is_file()}
        del active, base
        gc.collect()
        from .adapter_export import merge_adapter
        reloaded, stage = merge_adapter(
            base_dir, adapter_dir, configuration, recipe["adapter"], merged_dir, prompts,
            expected_parameters=report["fixture_base_parameters"],
            maximum_logit_error=LIMITS["merge_maximum_logit_error"],
        )
        report["base_reload"] = check_merge(base_logits, reloaded["base"],
                                            maximum_error=LIMITS["merge_maximum_logit_error"])
        report["adapter_reload"] = check_merge(active_logits, reloaded["adapter"],
                                               maximum_error=LIMITS["merge_maximum_logit_error"])
        merged_logits = reloaded["merged"]
        report["merge_stage"] = stage
        report["merge"] = check_merge(active_logits, merged_logits, maximum_error=LIMITS["merge_maximum_logit_error"])
        report["merged_files"] = copy_tokenizer(merged_dir)
        for label, values in (("base", base_logits), ("adapter", active_logits), ("merged", merged_logits)):
            np.save(output / (label + "-reference-logits.npy"), values, allow_pickle=False)
        gguf_paths = {}
        for label, directory in (("base-f32", base_dir), ("merged-f32", merged_dir)):
            path = workspace / (label + ".gguf")
            convert_f32(tools, directory, path, output / (label + "-conversion.log"), report["commands"],
                        model_name="Synthetic export verification only", timeout_seconds=LIMITS["command_seconds"])
            gguf_paths[label] = path
            report["maximum_work_bytes_observed"] = work_bytes(workspace)
            print(json.dumps({"stage": label + "-converted"}), flush=True)
        for prefix in ("base", "merged"):
            quantized = workspace / (prefix + "-q4.gguf")
            quantize_q4(tools, gguf_paths[prefix + "-f32"], quantized,
                        output / (prefix + "-quantization.log"), report["commands"],
                        timeout_seconds=LIMITS["command_seconds"])
            gguf_paths[prefix + "-q4"] = quantized
        report["maximum_work_bytes_observed"] = work_bytes(workspace)
        report["gguf"] = {label: inspect_gguf(path, configuration, token_fields) for label, path in gguf_paths.items()}
        require(all(item["parameters"] == report["fixture_base_parameters"] for item in report["gguf"].values()),
                "Export parameter inventory changed")
        require(all(report["gguf"][label]["file_type"] == int(gguf.LlamaFileType.MOSTLY_Q4_K_M)
                    for label in ("base-q4", "merged-q4")),
                "Quantized export has the wrong file type")
        report["serving"], report["comparisons"], report["numerical_failures"] = {}, {}, []
        for label, path in gguf_paths.items():
            values, receipt = await serving_rows(tools, output, path, label, prompts, configuration["vocab_size"],
                                                reference_precision=label.endswith("f32"))
            np.save(output / (label + "-log-probabilities.npy"), values, allow_pickle=False)
            report["serving"][label] = {"requests": len(receipt["requests"]), "shutdown": receipt["shutdown"],
                                       "precision": receipt["precision"]}
            reference = base_logits if label.startswith("base-") else merged_logits
            comparison = compare_native_probabilities(reference, values)
            report["comparisons"][label] = comparison
            if label.endswith("f32"):
                if comparison["maximum_absolute_log_probability_error"] > LIMITS["f32_maximum_log_probability_error"]:
                    report["numerical_failures"].append(label + ": F32 raw log-probability error")
            else:
                if comparison["mean_kl_divergence"] > LIMITS["quantized_maximum_mean_kl"]:
                    report["numerical_failures"].append(label + ": Q4 mean divergence")
            if label == "merged-q4":
                unchanged_base = compare_probabilities(merged_logits, log_probabilities(base_logits))
                report["unchanged_base_against_merged"] = unchanged_base
                if comparison["mean_kl_divergence"] >= unchanged_base["mean_kl_divergence"]:
                    report["numerical_failures"].append(label + ": synthetic adapter effect not retained")
            print(json.dumps({"stage": label + "-served", "mean_kl": comparison["mean_kl_divergence"]}), flush=True)
        require(tools.verify() == report["toolchain"], "Export toolchain changed during the probe")
        for name, expected in pinned["files"].items():
            require(identity(root / name) == expected, "Export tool changed during the probe")
        require(all(identity(base_dir / name) == value for name, value in report["base_files"].items()),
                "The original synthetic base files changed")
        report["original_synthetic_base_files_unchanged"] = True
        require(not report["numerical_failures"],
                "Synthetic pair rejected: " + "; ".join(report["numerical_failures"]))
        report["passed"] = True
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        if report["passed"]:
            shutil.rmtree(workspace)
        report["synthetic_weight_workspace_removed"] = not workspace.exists()
        report["seconds"] = round(time.monotonic() - began, 3)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(run(Path(__file__).resolve().parents[1], args.output))
    print(json.dumps({key: result[key] for key in ("passed", "seconds", "fixture_base_parameters",
                                                  "fixture_adapter_parameters", "optimizer_steps")}, indent=2))
