"""Check saved LoRA artifacts and merge them without executing saved trainer code.

Receipt checks link evidence; they do not independently establish human consent
or qualification. The numerical core also accepts explicitly synthetic fixtures.
Only admit_training_export accepts a completed real worker run.
"""

import gc
import json
import math
from pathlib import Path
import shutil
import stat

from .export_math import check_merge
from .training_inputs import (
    TARGET_MODULES, admission_errors, check_versions, load, require, sha256,
    validate_recipe, verified_path, verify_admission_evidence,
)
from .training_source import expected_shapes, read_header, unique_object


MAX_ADAPTER_BYTES = 256 * 1024**2


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def adapter_shapes(config, adapter):
    require(set(adapter) == {"r", "lora_alpha", "lora_dropout", "bias", "task_type", "target_modules"},
            "Export supports the declared pilot adapter fields only")
    require(type(adapter["r"]) is int and 1 <= adapter["r"] <= 64
            and adapter["target_modules"] == TARGET_MODULES
            and adapter["bias"] == "none" and adapter["task_type"] == "CAUSAL_LM",
            "Unsupported adapter architecture")
    result = {}
    for name, shape in expected_shapes(config).items():
        if name.split(".")[-2] not in TARGET_MODULES:
            continue
        require(len(shape) == 2, "An adapter target is not a matrix")
        prefix = "base_model.model." + name.removesuffix(".weight")
        result[prefix + ".lora_A.weight"] = [adapter["r"], shape[1]]
        result[prefix + ".lora_B.weight"] = [shape[0], adapter["r"]]
    require(len(result) == config["num_hidden_layers"] * len(TARGET_MODULES) * 2,
            "Adapter inventory omitted a projection")
    return result


def inspect_adapter(directory, config, adapter):
    """Check exact configuration, tensor layout, finite values and nonzero pairs."""
    directory = Path(directory)
    configuration, weights = directory / "adapter_config.json", directory / "adapter_model.safetensors"
    for path, limit in ((configuration, 64 * 1024), (weights, MAX_ADAPTER_BYTES)):
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and not path.is_symlink() and 0 < info.st_size <= limit,
                "Invalid or oversized adapter file")
    actual = json.loads(configuration.read_bytes(), object_pairs_hook=unique_object)
    from peft import LoraConfig
    expected_config = LoraConfig(**adapter).to_dict()
    require(set(actual) == set(expected_config), "Saved adapter has undeclared configuration fields")
    require(actual["base_model_name_or_path"] is None
            or isinstance(actual["base_model_name_or_path"], str)
            and len(actual["base_model_name_or_path"]) <= 4096, "Invalid saved base-model path")
    # The explicit verified base is used. Never follow the saved host path.
    expected_config["base_model_name_or_path"] = actual["base_model_name_or_path"]
    expected_config["inference_mode"] = True
    require(isinstance(actual["target_modules"], list)
            and all(isinstance(name, str) for name in actual["target_modules"]),
            "Invalid saved target modules")
    actual["target_modules"] = sorted(actual["target_modules"])
    expected_config["target_modules"] = sorted(expected_config["target_modules"])
    canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    require(canonical(actual) == canonical(expected_config), "Saved adapter configuration differs from the recipe")
    shapes = adapter_shapes(config, adapter)
    raw, tensors = read_header(weights)
    require(set(tensors) == set(shapes), "Adapter contains missing or unexpected tensors")
    intervals = []
    for name, tensor in tensors.items():
        require(set(tensor) == {"dtype", "shape", "data_offsets"}
                and tensor["dtype"] == "F32" and tensor["shape"] == shapes[name],
                "Adapter tensor dtype or shape differs: " + name)
        offsets = tensor["data_offsets"]
        require(isinstance(offsets, list) and len(offsets) == 2
                and all(type(value) is int and value >= 0 for value in offsets)
                and offsets[1] - offsets[0] == math.prod(shapes[name]) * 4,
                "Adapter tensor byte range differs")
        intervals.append(tuple(offsets))
    end = 0
    for begin, next_end in sorted(intervals):
        require(begin == end, "Adapter tensor bytes overlap or contain a gap")
        end = next_end
    require(len(raw) + end == weights.stat().st_size, "Adapter has truncated or trailing tensor bytes")
    import torch
    from safetensors import safe_open
    nonzero = set()
    with safe_open(weights, framework="pt", device="cpu") as saved:
        require(set(saved.keys()) == set(shapes), "Safe tensor loader inventory differs")
        for name in saved.keys():
            tensor = saved.get_tensor(name)
            require(tensor.dtype == torch.float32 and list(tensor.shape) == shapes[name]
                    and torch.isfinite(tensor).all().item(), "Adapter has nonfinite or changed tensor values")
            if torch.count_nonzero(tensor).item():
                nonzero.add(name)
    pairs = sum(name in nonzero and name.replace(".lora_A.", ".lora_B.") in nonzero
                for name in shapes if ".lora_A." in name)
    require(pairs > 0, "Adapter has no nonzero A/B projection pair")
    return {"files": {path.name: identity(path) for path in (configuration, weights)},
            "tensor_count": len(shapes), "parameters": sum(math.prod(shape) for shape in shapes.values()),
            "nonzero_projection_pairs": pairs, "all_tensor_values_finite": True,
            "saved_base_path_followed": False, "training_provenance_verified": False,
            "qualification_passed": False}


def validate_run_receipt(run, workload, manifest, recipe, recipe_sha, contract_sha):
    require(run.get("schema") == "organism-lora-training-run-v1"
            and run.get("status") == "training_completed_unqualified"
            and run.get("training_completed") is True and run.get("optimization_started") is True
            and run.get("qualification_passed") is False and not run.get("failure"),
            "A completed, unqualified training run is required")
    require(type(run["optimizer_steps"]) is int
            and run["optimizer_steps"] == recipe["training"]["maximum_optimizer_steps"],
            "Training did not complete the declared optimizer budget")
    require(run["recipe_sha256"] == recipe_sha and run["frozen_contract_sha256"] == contract_sha
            and run["base_metadata_sha256"] == recipe["inputs"]["base_metadata"]["sha256"]
            and run["derived_loader_index_sha256"] == recipe["inputs"]["loader_index"]["sha256"]
            and run["base_parameters"] == recipe["base"]["expected_parameters"]
            and run["versions"] == recipe["versions"], "Training provenance differs from the admitted inputs")
    require(run.get("model_source_verified") is True
            and run["native_source"].get("passed") is True
            and run["native_source"].get("full_native_payload_verified") is True
            and run["native_source"].get("parameters") == recipe["base"]["expected_parameters"],
            "Training did not verify the complete native base")
    expected_sources = {name: recipe["inputs"]["source/" + name]["sha256"]
                        for name in ("train_specialist.py", "training_inputs.py", "training_source.py")}
    require(run["source_hashes"] == expected_sources, "Recorded trainer source differs")
    require(workload.get("mode") == "train" and workload.get("passed") is True
            and workload.get("failure") is None and workload.get("network_isolation_observed") is True
            and workload.get("training_completed") is True
            and workload.get("optimizer_steps") == run["optimizer_steps"]
            and workload.get("training") == run
            and workload.get("recipe_sha256") == recipe_sha
            and workload.get("input_manifest_sha256") == manifest,
            "Worker and trainer completion evidence differ")


def validate_precheck_receipt(precheck, run, recipe, longest):
    """Link the required no-optimizer GPU check to this completed pilot."""
    require(isinstance(precheck, dict)
            and precheck.get("passed") is True
            and type(precheck.get("optimizer_steps")) is int
            and precheck["optimizer_steps"] == 0
            and precheck.get("optimization_started") is False
            and precheck.get("adapter_values_unchanged") is True
            and precheck.get("frozen_base_has_no_gradients") is True
            and precheck.get("training_completed") is False
            and precheck.get("qualification_passed") is False,
            "Completed training lacks its required no-optimizer GPU precheck")
    for name, value in (("base_parameters", recipe["base"]["expected_parameters"]),
                        ("adapter_parameters", run["trainable_parameters"]),
                        ("sequence_tokens", len(longest["input_ids"])),
                        ("decision_index", longest["decision_index"])):
        require(type(precheck.get(name)) is int and precheck[name] == value,
                "GPU precheck differs from the admitted pilot: " + name)
    require(precheck.get("case_id") == longest["case_id"]
            and precheck.get("gpu") == run["gpu"]
            and precheck.get("cuda_runtime") == run["cuda_runtime"],
            "GPU precheck used a different row or execution device")
    loss = precheck.get("loss")
    require(type(loss) in (int, float) and math.isfinite(loss) and loss >= 0,
            "GPU precheck loss is missing or nonfinite")
    # The trainer adds only these two index-audit annotations to the same
    # complete source inspection returned by the preceding verifier.
    expected_source = dict(run["native_source"])
    expected_source.pop("upstream_index_consistent", None)
    expected_source.pop("upstream_index_placement_mismatches", None)
    require(precheck.get("native_source") == expected_source,
            "GPU precheck and trainer native-source evidence differ")


def admit_training_export(workspace, worker_output):
    """Validate a completed worker's final adapter; never admit a draft or fixture."""
    workspace, worker_output = Path(workspace).resolve(), Path(worker_output).resolve()
    manifest_path = workspace / "worker-input.json"
    manifest = load(manifest_path)
    require(manifest.get("schema") == "organism-training-worker-input-v1" and manifest.get("mode") == "train"
            and not manifest.get("synthetic_fixture"), "A real training worker input is required")
    entries = manifest["files"]
    require(isinstance(entries, list) and 0 < len(entries) <= 128
            and len({item["path"] for item in entries}) == len(entries), "Invalid worker input inventory")
    files = {item["path"]: verified_path(workspace, item) for item in entries}
    require(manifest["recipe"] in files and manifest["contract"] in files, "Worker control files are not bound")
    recipe_path, contract_path = files[manifest["recipe"]], files[manifest["contract"]]
    recipe, contract = load(recipe_path), load(contract_path)
    validate_recipe(recipe)
    errors = admission_errors(contract, recipe, sha256(recipe_path))
    require(not errors, "Export training admission is missing: " + "; ".join(errors))
    require(not contract.get("synthetic_fixture"), "Synthetic admission cannot authorize a trained export")
    evidence = verify_admission_evidence(workspace, contract, recipe, sha256(recipe_path))
    for field in evidence:
        entry = contract["training_admission"][field]
        require(entry["path"] in files, "Admission evidence is absent from the worker input")
    paths = {name: verified_path(workspace, entry) for name, entry in recipe["inputs"].items()}
    run_path, workload_path = worker_output / "training/run.json", worker_output / "workload.json"
    run, workload = load(run_path), load(workload_path)
    validate_run_receipt(run, workload, sha256(manifest_path), recipe, sha256(recipe_path), sha256(contract_path))
    precheck_required = contract.get("training_preparation", {}).get("mandatory_gpu_precheck_before_optimizer")
    if precheck_required is not None:
        require(type(precheck_required) is bool, "Invalid native GPU precheck requirement")
    if precheck_required:
        rows = [json.loads(line) for line in paths["tokens_train"].read_text().splitlines()]
        require(len(rows) == recipe["counts"]["train"] and rows, "Prepared training row count changed")
        longest = max(rows, key=lambda row: len(row["input_ids"]))
        validate_precheck_receipt(workload.get("verification"), run, recipe, longest)
    training = run_path.parent
    declared = run["adapter_files"]
    require(isinstance(declared, dict) and 2 <= len(declared) <= 64, "Invalid saved adapter inventory")
    actual = {str(path.relative_to(training)) for path in (training / "adapter").rglob("*") if path.is_file()}
    require(actual == set(declared) and all(Path(name).parts[0] == "adapter" for name in declared),
            "Saved final adapter inventory differs from the completed run")
    require(sum(entry["bytes"] for entry in declared.values()) <= MAX_ADAPTER_BYTES,
            "Saved final adapter exceeds its byte allowance")
    for name, entry in declared.items():
        require(set(entry) == {"bytes", "sha256"}, "Saved adapter receipt contains undeclared fields")
        verified_path(training, {"path": name, **entry})
    config = load(paths["base_config"])
    inspected = inspect_adapter(training / "adapter", config, recipe["adapter"])
    require(inspected["parameters"] == run["trainable_parameters"], "Saved adapter parameter count differs from training")
    require(sum(math.prod(shape) for shape in expected_shapes(config).values())
            == recipe["base"]["expected_parameters"], "Base configuration parameter count differs")
    return {
        "recipe": recipe, "paths": paths, "config": config, "adapter_directory": training / "adapter",
        "receipt": {
            "schema": "organism-trained-adapter-export-admission-v1",
            "worker_input": identity(manifest_path), "worker_completion": identity(workload_path),
            "training_completion": identity(run_path), "recipe": identity(recipe_path),
            "frozen_contract": identity(contract_path), "adapter": inspected,
            "optimizer_steps": run["optimizer_steps"], "base_parameters": run["base_parameters"],
            "checkpoint": "final adapter from completed pilot; no quality selection implied",
            "training_receipts_linked": True, "human_consent_independently_verified_by_this_check": False,
            "required_native_precheck_linked": precheck_required is True,
            "qualification_passed": False,
        },
    }


def merge_adapter(base, adapter_directory, config, adapter, output, prompts, *, expected_parameters,
                  maximum_logit_error, device="cpu"):
    """Merge verified tensor files and save F32 weights; no optimizer or promotion."""
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, GraniteConfig

    check_versions()
    require(device in ("cpu", "cuda:0"), "Unsupported export device")
    if device == "cuda:0":
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One CUDA device is required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    require(type(expected_parameters) is int and expected_parameters > 0
            and math.isfinite(maximum_logit_error) and 0 < maximum_logit_error <= 0.01,
            "Invalid merge comparison contract")
    require(isinstance(prompts, list) and 1 <= len(prompts) <= 16
            and all(isinstance(tokens, list) and 0 < len(tokens) <= 4096
                    and all(type(token) is int and 0 <= token < config["vocab_size"] for token in tokens)
                    for tokens in prompts), "Invalid bounded export prompts")
    base, adapter_directory, output = (Path(path).resolve() for path in (base, adapter_directory, output))
    require(not output.exists() and not output.is_relative_to(base) and not output.is_relative_to(adapter_directory),
            "Merged output would overwrite an input")
    require(shutil.disk_usage(output.parent).free >= expected_parameters * 4 + 256 * 1024**2,
            "Insufficient free space for merged weights")
    inspected = inspect_adapter(adapter_directory, config, adapter)
    model = active = merged = None
    try:
        model, loading = AutoModelForCausalLM.from_pretrained(
            base, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=torch.float32, device_map={"": device}, attn_implementation="sdpa", use_cache=False,
            output_loading_info=True,
        )
        require(not any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
                "Base loading left missing or mismatched tensors")
        require(type(model).__name__ == "GraniteForCausalLM"
                and sum(value.numel() for value in model.parameters()) == expected_parameters,
                "Loaded merge base has the wrong architecture or parameter count")
        canonical = GraniteConfig(**config).to_dict()
        actual = model.config.to_dict()
        for key in ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "vocab_size", "hidden_act", "rms_norm_eps", "rope_parameters",
                    "max_position_embeddings", "attention_multiplier", "embedding_multiplier",
                    "residual_multiplier", "logits_scaling", "tie_word_embeddings", "attention_bias", "mlp_bias"):
            require(actual[key] == canonical[key], "Loaded merge configuration differs: " + key)

        def logits(candidate):
            candidate.eval()
            with torch.inference_mode():
                result = np.stack([candidate(torch.tensor([tokens], device=device), use_cache=False)
                                   .logits[0, -1].float().cpu().numpy().copy() for tokens in prompts])
            require(result.shape == (len(prompts), config["vocab_size"]) and np.isfinite(result).all(),
                    "Merge reference logits are incomplete or nonfinite")
            return result

        references = {"base": logits(model)}
        active = PeftModel.from_pretrained(model, adapter_directory, local_files_only=True, is_trainable=False).eval()
        references["adapter"] = logits(active)
        merged = active.merge_and_unload(safe_merge=True).eval()
        references["merged"] = logits(merged)
        comparison = check_merge(references["adapter"], references["merged"], maximum_error=maximum_logit_error)
        require(sum(value.numel() for value in merged.parameters()) == expected_parameters
                and all(".lora_" not in name for name, _ in merged.named_parameters()),
                "Merged weights retain adapter modules or changed parameter count")
        # A directory is reserved only after successful loading and comparison.
        output.mkdir(exist_ok=False)
        merged.save_pretrained(output, safe_serialization=True, max_shard_size="1GB")
        require(all(identity(adapter_directory / name) == expected
                    for name, expected in inspected["files"].items()), "Input adapter changed during merging")
        return references, {
            "schema": "organism-adapter-merge-v1", "adapter": inspected, "comparison": comparison,
            "base_parameters": expected_parameters, "device": device, "load_dtype": "float32",
            "versions": check_versions(), "files": {path.name: identity(path) for path in output.iterdir() if path.is_file()},
            "optimizer_steps": 0, "qualification_passed": False, "deployment_export_complete": False,
        }
    finally:
        del model, active, merged
        gc.collect()
