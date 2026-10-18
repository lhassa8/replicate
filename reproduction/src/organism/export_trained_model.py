"""Export a matched stock/trained GGUF pair from an admitted completed merge.

This is an offline export-host stage, not a worker lease, result uploader,
checkpoint selector, device capsule builder or qualification run.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import time

from .adapter_export import admit_training_export
from .export_math import check_merge, compare_native_probabilities
from .export_serving import serving_rows
from .gguf_export import Toolchain, convert_f32, identity, inspect_gguf, quantize_q4, tokenizer_fingerprint
from .training_inputs import load, offline_environment, require, validate_row, verified_path
from .training_source import expected_shapes, inspect_headers, read_header


LIMITS = {
    "minimum_free_bytes": 64 * 1024**3, "maximum_output_bytes": 64 * 1024**3,
    "stage_seconds": 7200, "command_seconds": 1200, "serving_seconds": 1900, "request_seconds": 600,
    "merge_maximum_logit_error": 0.0002, "f32_maximum_log_probability_error": 0.002,
    "quantized_maximum_mean_kl": 0.10,
}
SOURCES = ("export_trained_model.py", "gguf_export.py", "export_serving.py", "adapter_export.py",
           "merge_trained_adapter.py", "export_math.py", "training_inputs.py", "training_source.py",
           "train_specialist.py", "inference_server.py", "runtime_environment.py",
           "quantization_calibration.py", "quantization_precision.py")


def write(path, value):
    with path.open("x") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_flat_inventory(directory, entries, *, maximum_bytes):
    require(isinstance(entries, dict) and 1 <= len(entries) <= 64
            and all(isinstance(name, str) and Path(name).name == name for name in entries),
            "Invalid export file inventory")
    actual = {path.name for path in directory.iterdir()}
    require(actual == set(entries), "Export file inventory has missing or undeclared entries")
    require(all(set(entry) == {"bytes", "sha256"} and type(entry["bytes"]) is int and entry["bytes"] > 0
                for entry in entries.values())
            and sum(entry["bytes"] for entry in entries.values()) <= maximum_bytes,
            "Export input exceeds its byte allowance")
    return {name: verified_path(directory, {"path": name, **entry}) for name, entry in entries.items()}


def verify_f32_weights(directory, config):
    """Check safe tensor headers and index against the complete Granite shape map."""
    shapes, seen, weight_map = expected_shapes(config), {}, {}
    shards = sorted(directory.glob("*.safetensors"))
    require(1 <= len(shards) <= 32, "Invalid merged weight shard inventory")
    total = 0
    for path in shards:
        raw, tensors = read_header(path)
        intervals = []
        for name, tensor in tensors.items():
            require(name in shapes and name not in seen and tensor.get("dtype") == "F32"
                    and tensor.get("shape") == shapes[name], "Merged tensor inventory or dtype differs")
            offsets = tensor["data_offsets"]
            require(isinstance(offsets, list) and len(offsets) == 2
                    and all(type(value) is int and value >= 0 for value in offsets)
                    and offsets[1] - offsets[0] == math.prod(shapes[name]) * 4,
                    "Merged tensor byte range differs")
            intervals.append(tuple(offsets))
            seen[name], weight_map[name] = tensor["shape"], path.name
        end = 0
        for start, next_end in sorted(intervals):
            require(start == end, "Merged tensor bytes overlap or contain gaps")
            end = next_end
        require(len(raw) + end == path.stat().st_size, "Merged weights have truncated or trailing bytes")
        total += end
    require(seen == shapes, "Merged weights omit model tensors")
    index = directory / "model.safetensors.index.json"
    if index.exists():
        saved = load(index)
        require(isinstance(saved, dict) and isinstance(saved.get("metadata"), dict),
                "Merged weight index metadata is invalid")
        metadata = saved["metadata"]
        expected_metadata = {"total_size": total}
        # The pinned Transformers saver also records the parameter count.
        # Validate it against the tensors instead of rejecting that field.
        if "total_parameters" in metadata:
            expected_metadata["total_parameters"] = total // 4
        require(all(type(value) is int for value in metadata.values())
                and saved == {"metadata": expected_metadata, "weight_map": weight_map},
                "Merged weight index differs from the actual tensors")
    else:
        require(len(shards) == 1 and shards[0].name == "model.safetensors",
                "Sharded merged weights require an exact loader index")
    return {"tensor_count": len(seen), "parameters": total // 4, "dtype": "F32",
            "tensor_payload_bytes": total}


def development_prompts(admitted):
    recipe, config, paths = admitted["recipe"], admitted["config"], admitted["paths"]
    rows = [json.loads(line) for line in paths["tokens_train"].read_text().splitlines()]
    require(len(rows) == recipe["counts"]["train"] and rows, "Development row count changed")
    indices = list(dict.fromkeys((0, max(range(len(rows)), key=lambda index: rows[index]["prompt_tokens"]),
                                 len(rows) - 1)))
    prompts, cases = [], []
    for index in indices:
        row = rows[index]
        validate_row(row, vocab_size=config["vocab_size"], eos_token_id=config["eos_token_id"])
        require(0 < row["prompt_tokens"] < 4096, "Development reference exceeds the export context")
        prompts.append(row["input_ids"][:row["prompt_tokens"]])
        cases.append({"case_id": row["case_id"], "decision_index": row["decision_index"],
                      "prompt_tokens": row["prompt_tokens"], "source_split": "train"})
    return prompts, cases


def verify_merge_report(report, admitted, prompts, cases):
    require(report.get("schema") == "organism-trained-adapter-merge-run-v1"
            and report.get("passed") is True and report.get("failure") is None
            and report.get("original_inputs_unchanged") is True
            and report.get("optimizer_steps_this_stage") == 0
            and report.get("qualification_passed") is False
            and report.get("quality_selection_performed") is False
            and report.get("deployment_export_complete") is False,
            "A completed unqualified merge is required")
    require(report["admission"] == admitted["receipt"], "Merge belongs to a different admitted training run")
    require(report["prompts"] == prompts and report["development_reference_cases"] == cases,
            "Merge references differ from the declared development rows")
    require(report["maximum_merge_logit_error"] == LIMITS["merge_maximum_logit_error"]
            and report["merge"]["comparison"].get("passed") is True
            and report["merge"]["comparison"]["maximum_allowed_error"] == LIMITS["merge_maximum_logit_error"],
            "Merge tolerance changed or the comparison failed")
    require(report["native_base"].get("passed") is True
            and report["native_base"].get("full_native_payload_verified") is True
            and report["native_base"]["parameters"] == admitted["recipe"]["base"]["expected_parameters"],
            "Merge did not verify the complete native base")
    require(report["source_hashes"] == {
        name: identity(Path(__file__).with_name(name))
        for name in ("merge_trained_adapter.py", "adapter_export.py", "export_math.py")
    }, "Merge implementation differs from the recorded source")


def verify_merge_bundle(admitted, directory):
    import numpy as np
    from transformers import GraniteConfig

    report_path = directory / "report.json"
    require(not report_path.is_symlink() and report_path.stat().st_size <= 8 * 1024**2,
            "Invalid merge report")
    report = load(report_path)
    recipe, paths, config = admitted["recipe"], admitted["paths"], admitted["config"]
    parameters = recipe["base"]["expected_parameters"]
    prompts, cases = development_prompts(admitted)
    verify_merge_report(report, admitted, prompts, cases)

    metadata = load(paths["base_metadata"])
    base_entries = {name: {key: entry[key] for key in ("bytes", "sha256")}
                    for name, entry in metadata["files"].items() if name != "model.safetensors.index.json"}
    # The retained loading view deliberately carries the corrected index.
    base_entries["model.safetensors.index.json"] = identity(paths["loader_index"])
    base_files = verify_flat_inventory(directory / "base-loading-view", base_entries,
                                       maximum_bytes=parameters * 2 + 1024**2)
    inspect_headers(metadata, load(paths["loader_index"]),
                    {name: path for name, path in base_files.items() if name.endswith(".safetensors")},
                    config, parameters)

    merged_files = verify_flat_inventory(directory / "merged", report["merged_files"],
                                         maximum_bytes=parameters * 4 + 256 * 1024**2)
    tokenizer_files = {}
    for entry in load(paths["tokenizer_receipt"])["files"]:
        if entry["name"] != "config.json":
            source = verified_path(paths["tokenizer_receipt"].parent,
                                   {"path": entry["name"], **{key: entry[key] for key in ("bytes", "sha256")}})
            tokenizer_files[entry["name"]] = source
            require(identity(merged_files[entry["name"]]) == identity(source), "Merged tokenizer source changed")
    require(set(report["merged_files"]) == set(report["merge"]["files"]) | set(tokenizer_files)
            and all(report["merged_files"][name] == value for name, value in report["merge"]["files"].items()),
            "Merged model files differ from the merge-stage inventory")
    require(all(name in ("config.json", "generation_config.json", "model.safetensors.index.json")
                or name.endswith(".safetensors")
                for name in report["merge"]["files"]), "Merged model contains an unexpected loader file")
    generation = load(merged_files["generation_config.json"])
    require(all(generation.get(key) == config[key] for key in ("bos_token_id", "eos_token_id", "pad_token_id")),
            "Merged generation token identities changed")
    expected_config = GraniteConfig(**config).to_dict()
    actual_config = GraniteConfig(**load(merged_files["config.json"])).to_dict()
    for key in ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "vocab_size", "hidden_act", "rms_norm_eps", "rope_parameters",
                "max_position_embeddings", "attention_multiplier", "embedding_multiplier",
                "residual_multiplier", "logits_scaling", "tie_word_embeddings", "attention_bias", "mlp_bias"):
        require(actual_config[key] == expected_config[key], "Merged computational configuration changed: " + key)
    inventory = verify_f32_weights(directory / "merged", config)
    require(inventory["parameters"] == parameters, "Merged model parameter count changed")

    require(set(report["references"]) == {label + "-reference-logits.npy" for label in ("base", "adapter", "merged")},
            "Merge reference inventory changed")
    references = {}
    for name, entry in report["references"].items():
        require(0 < entry["bytes"] <= 8 * 1024**2, "Oversized reference array")
        path = verified_path(directory, {"path": name, **entry})
        values = np.load(path, allow_pickle=False)
        require(values.shape == (len(prompts), config["vocab_size"]) and values.dtype == np.float32
                and np.isfinite(values).all(), "Invalid merge reference array")
        references[name.removesuffix("-reference-logits.npy")] = values
    comparison = check_merge(references["adapter"], references["merged"],
                             maximum_error=LIMITS["merge_maximum_logit_error"])
    require(comparison == report["merge"]["comparison"], "Saved references do not reproduce the merge result")
    return {"report": report, "base_files": base_files, "tokenizer_files": tokenizer_files,
            "references": references, "prompts": prompts, "cases": cases,
            "receipt": {"merge_report": identity(report_path), "base_files": base_entries,
                        "merged_files": report["merged_files"], "references": report["references"],
                        "tensor_inventory": inventory, "qualification_passed": False}}


async def run(workspace, worker_output, merge_output, output, tools, *, calibration=None,
              precision_profile=None):
    offline_environment()
    began = time.monotonic()
    workspace, worker_output, merge_output, output = (
        Path(path).resolve() for path in (workspace, worker_output, merge_output, output))
    # Admission comes before model loading, tool execution or output creation.
    admitted = admit_training_export(workspace, worker_output)
    require(not output.exists()
            and all(not output.is_relative_to(path) and not path.is_relative_to(output)
                    for path in (workspace, worker_output, merge_output)), "Export would overlap an input bundle")
    require(shutil.disk_usage(output.parent).free >= LIMITS["minimum_free_bytes"],
            "Matched full-size export requires at least 64 GiB free scratch space")
    bound = verify_merge_bundle(admitted, merge_output)
    verified_tools = tools.verify()
    gguf = tools.import_gguf()
    precision = None
    if precision_profile is not None:
        from .quantization_precision import inspect_precision, specification
        require(calibration is not None, "Precision experiment requires bound calibration")
        precision = specification(precision_profile)
    calibration_plan = None
    if calibration is not None:
        from transformers import AutoTokenizer
        from .quantization_calibration import verify_calibration
        calibration = Path(calibration).resolve()
        require(calibration == workspace / "export-calibration" and "imatrix" in tools.manifest,
                "Calibrated export requires the bound training-only input and executable")
        tokenizer = AutoTokenizer.from_pretrained(admitted["paths"]["tokenizer_receipt"].parent,
                                                  local_files_only=True)
        calibration_plan = verify_calibration(admitted, tokenizer, calibration)
    output.mkdir(exist_ok=False)
    report = {
        "schema": "organism-trained-gguf-pair-export-v1", "started_at": datetime.now(timezone.utc).isoformat(),
        "passed": False, "failure": None, "admission": admitted["receipt"], "merge": bound["receipt"],
        "tools": verified_tools, "limits": LIMITS, "commands": [], "models": {}, "comparisons": {}, "serving": {},
        "numerical_failures": [],
        "prompts": bound["prompts"], "development_reference_cases": bound["cases"],
        "export_pair_verified": False, "deployment_export_complete": False, "quality_selection_performed": False,
        "optimizer_steps_this_stage": 0, "qualification_passed": False,
        "execution_context": "export host, not an 8 GiB device test",
        "source_hashes": {name: identity(Path(__file__).with_name(name)) for name in SOURCES},
    }
    if precision is not None:
        report["schema"] = "organism-trained-gguf-pair-export-v2"
        report["quantization_precision"] = {"specification": precision, "models": {}}
    write(output / "request.json", report)
    write(output / "toolchain.json", tools.manifest)
    if calibration_plan is not None:
        report["calibration"] = {"plan": calibration_plan, "matrices": {}}
        write(output / "calibration-plan.json", calibration_plan)
        shutil.copyfile(calibration / "calibration.txt", output / "calibration.txt")

    def budget(limit):
        remaining = LIMITS["stage_seconds"] - (time.monotonic() - began)
        require(remaining > 0, "Export stage deadline exhausted")
        size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        report["maximum_output_bytes_observed"] = max(size, report.get("maximum_output_bytes_observed", 0))
        require(size <= LIMITS["maximum_output_bytes"], "Export scratch allowance exhausted")
        require(shutil.disk_usage(output).free >= 1024**3, "Export scratch reserve exhausted")
        return min(limit, remaining)

    try:
        stock = output / "stock-loading-view"
        stock.mkdir()
        for name, path in bound["base_files"].items():
            if name.endswith(".safetensors"):
                os.link(path, stock / name)
            else:
                shutil.copyfile(path, stock / name)
        for name, path in bound["tokenizer_files"].items():
            shutil.copyfile(path, stock / name)
        for label, values in bound["references"].items():
            import numpy as np
            np.save(output / (label + "-reference-logits.npy"), values, allow_pickle=False)
        files = {}
        for label, directory in (("stock-f32", stock), ("trained-f32", merge_output / "merged")):
            path = output / (label + ".gguf")
            convert_f32(tools, directory, path, output / (label + "-conversion.log"), report["commands"],
                        model_name="Organism " + label + " unqualified export",
                        timeout_seconds=budget(LIMITS["command_seconds"]))
            files[label] = path
        token_fields = tokenizer_fingerprint(files["stock-f32"])
        require(len(token_fields) >= 5, "Converted stock model lacks tokenizer metadata")
        for prefix in ("stock", "trained"):
            label, path = prefix + "-q4", output / (prefix + "-q4.gguf")
            matrix = None
            if calibration_plan is not None:
                from .quantization_calibration import collect_imatrix
                matrix = output / (prefix + "-imatrix.gguf")
                report["calibration"]["matrices"][prefix] = collect_imatrix(
                    tools, files[prefix + "-f32"], output / "calibration.txt", matrix, admitted["config"],
                    output / (prefix + "-calibration.log"), report["commands"],
                    process_output=calibration_plan["settings"]["process_output"],
                    timeout_seconds=budget(LIMITS["command_seconds"]))
            quantize_q4(tools, files[prefix + "-f32"], path, output / (label + "-quantization.log"),
                        report["commands"], timeout_seconds=budget(LIMITS["command_seconds"]), imatrix=matrix,
                        precision_profile=precision_profile)
            files[label] = path
            if precision is not None:
                report["quantization_precision"]["models"][prefix] = inspect_precision(path, precision_profile)
        if precision is not None:
            require(report["quantization_precision"]["models"]["stock"]
                    == report["quantization_precision"]["models"]["trained"],
                    "Matched models have different tensor precision")
        for label, path in files.items():
            report["models"][label] = inspect_gguf(path, admitted["config"], token_fields)
            expected_type = gguf.LlamaFileType.ALL_F32 if label.endswith("f32") else gguf.LlamaFileType.MOSTLY_Q4_K_M
            require(report["models"][label]["file_type"] == int(expected_type)
                    and report["models"][label]["parameters"] == admitted["recipe"]["base"]["expected_parameters"],
                    "Export file type or model size differs")
            total = budget(LIMITS["serving_seconds"])
            values, serving = await serving_rows(
                tools, output, path, label, bound["prompts"], admitted["config"]["vocab_size"],
                reference_precision=label.endswith("f32"), synthetic=False, trained=label.startswith("trained-"),
                total_seconds=total, request_seconds=min(LIMITS["request_seconds"], total),
            )
            np.save(output / (label + "-log-probabilities.npy"), values, allow_pickle=False)
            report["serving"][label] = {"requests": len(serving["requests"]), "shutdown": serving["shutdown"],
                                        "precision": serving["precision"]}
            reference = bound["references"]["base" if label.startswith("stock-") else "merged"]
            comparison = compare_native_probabilities(reference, values)
            report["comparisons"][label] = comparison
            if label.endswith("f32"):
                if comparison["maximum_absolute_log_probability_error"] > LIMITS["f32_maximum_log_probability_error"]:
                    report["numerical_failures"].append(label + ": F32 raw log-probability error")
            else:
                if comparison["mean_kl_divergence"] > LIMITS["quantized_maximum_mean_kl"]:
                    report["numerical_failures"].append(label + ": Q4 mean divergence")
        budget(1)
        require(tools.verify() == verified_tools, "Export tools changed during the stage")
        require(verify_merge_bundle(admitted, merge_output)["receipt"] == bound["receipt"]
                and admit_training_export(workspace, worker_output)["receipt"] == admitted["receipt"],
                "Export inputs changed during the stage")
        require(all(identity(Path(__file__).with_name(name)) == entry for name, entry in report["source_hashes"].items()),
                "Export implementation changed during the stage")
        if calibration_plan is not None:
            require(verify_calibration(admitted, tokenizer, calibration) == calibration_plan
                    and identity(output / "calibration.txt") == calibration_plan["text"],
                    "Calibration input changed during export")
        budget(1)
        require(not report["numerical_failures"],
                "Export pair rejected: " + "; ".join(report["numerical_failures"]))
        report["export_pair_verified"] = True
        report["passed"] = True
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["seconds"] = round(time.monotonic() - began, 3)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "worker-output", "merge-output", "output", "tools-root", "tools-manifest"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--precision-profile")
    args = parser.parse_args()
    result = asyncio.run(run(args.workspace, args.worker_output, args.merge_output, args.output,
                             Toolchain(args.tools_root.resolve(), load(args.tools_manifest)),
                             calibration=args.calibration, precision_profile=args.precision_profile))
    print(json.dumps({name: result[name] for name in ("passed", "seconds", "export_pair_verified",
                                                     "qualification_passed")}, indent=2))
