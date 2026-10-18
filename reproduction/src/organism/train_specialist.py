"""First LoRA pilot on verified prepared tokens; admission and native weights required."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from .training_inputs import (ExactTargetCollator, PreparedRows, admission_errors, check_versions,
                              load, load_inputs, offline_environment, require, sha256,
                              verified_path, verify_admission_evidence)


def target_token_loss(outputs, labels, num_items_in_batch=None):
    """One next-token loss mean across all supervised tokens in the update."""
    import torch
    shifted = labels[..., 1:].contiguous()
    logits = outputs.logits[..., :-1, :].contiguous().float()
    total = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), shifted.reshape(-1),
                                              ignore_index=-100, reduction="sum")
    require(torch.isfinite(total).item(), "Nonfinite target-token loss")
    denominator = shifted.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
    require(int(denominator) > 0, "Loss has no supervised target tokens")
    return total / denominator


def training_arguments(recipe, output, *, cpu_preflight=False):
    from transformers import TrainingArguments
    cfg = recipe["training"]
    return TrainingArguments(
        output_dir=str(output), per_device_train_batch_size=cfg["per_device_batch_size"],
        per_device_eval_batch_size=1, gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"], max_grad_norm=cfg["max_grad_norm"],
        num_train_epochs=cfg["epochs"], max_steps=cfg["maximum_optimizer_steps"],
        warmup_steps=cfg["warmup_steps"], lr_scheduler_type=cfg["lr_scheduler_type"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": cfg["use_reentrant_checkpointing"]},
        bf16=not cpu_preflight, fp16=False, use_cpu=cpu_preflight, optim="adamw_torch",
        eval_strategy="steps", eval_steps=cfg["updates_per_epoch"],
        save_strategy="steps", save_steps=cfg["updates_per_epoch"], save_total_limit=None,
        load_best_model_at_end=False, prediction_loss_only=True, remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=not cpu_preflight,
        logging_steps=1, report_to="none", push_to_hub=False, disable_tqdm=True,
        seed=cfg["seed"], data_seed=cfg["seed"], average_tokens_across_devices=False,
    )


def verify_native_base(base, metadata, recipe, loader_index):
    """Compare local tensors with hashes retrieved from the pinned upstream revision."""
    require(metadata["schema"] == "organism-native-training-source-v1"
            and metadata["repository"] == recipe["base"]["repository"]
            and metadata["revision"] == recipe["base"]["revision"], "Native model source revision differs")
    files = metadata["files"]
    require(set(files) == {"config.json", "model.safetensors.index.json",
                           "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"},
            "Unexpected native training file inventory")
    for name, entry in files.items():
        verified_path(base, {"path": name, "bytes": entry["bytes"], "sha256": entry["sha256"]})
    index = load(Path(base) / "model.safetensors.index.json")
    require(set(index["weight_map"].values()) == {name for name in files if name.endswith(".safetensors")},
            "Tensor index references an undeclared shard")
    from .training_source import inspect_headers
    return inspect_headers(metadata, load(loader_index), {name: Path(base) / name for name in files if name.endswith(".safetensors")},
                           load(Path(base) / "config.json"), recipe["base"]["expected_parameters"])


def prepare_loading_view(base, metadata, loader_index, view):
    """Create a distinct loader view while retaining original files and tensors."""
    view = Path(view)
    view.mkdir()
    for name in metadata["files"]:
        if name.endswith(".safetensors"):
            os.link(Path(base).resolve() / name, view / name)
    (view / "config.json").write_bytes((Path(base) / "config.json").read_bytes())
    (view / "model.safetensors.index.json").write_bytes(Path(loader_index).read_bytes())
    return view


def run(root, recipe_path, contract_path, base, metadata_path, output):
    offline_environment()
    root, output = Path(root).resolve(), Path(output)
    recipe, contract = load(recipe_path), load(contract_path)
    errors = admission_errors(contract, recipe, sha256(recipe_path))
    require(not errors, "Training is not admitted: " + "; ".join(errors))
    verify_admission_evidence(root, contract, recipe, sha256(recipe_path))
    check_versions()
    paths, rows, config = load_inputs(root, recipe)
    require(sha256(metadata_path) == recipe["inputs"]["base_metadata"]["sha256"], "Native source receipt is not bound to the recipe")
    metadata = load(metadata_path)
    require(metadata["files"]["config.json"]["sha256"] == sha256(paths["base_config"]),
            "Pinned source config differs from the audited architecture")
    from .training_source import inspect_recipe_source
    header_source = inspect_recipe_source(paths, recipe["base"]["expected_parameters"])
    native_source = verify_native_base(base, metadata, recipe, paths["loader_index"])
    native_source["full_native_payload_verified"] = True
    native_source["upstream_index_consistent"] = header_source["upstream_index_consistent"]
    native_source["upstream_index_placement_mismatches"] = header_source["upstream_index_placement_mismatches"]
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, set_seed
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
            "The pilot requires one native CUDA GPU with BF16 support")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {
        "schema": "organism-lora-training-run-v1", "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "loading_verified_base", "recipe_sha256": sha256(recipe_path),
        "frozen_contract_sha256": sha256(contract_path), "base_metadata_sha256": sha256(metadata_path),
        "versions": check_versions(), "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda, "training_completed": False, "qualification_passed": False,
        "optimization_started": False, "optimizer_steps": 0,
        "model_source_verified": True, "base_parameters": recipe["base"]["expected_parameters"],
        "native_source": native_source,
        "source_hashes": {name: sha256(Path(__file__).with_name(name))
                          for name in ("train_specialist.py", "training_inputs.py", "training_source.py")},
    }

    def persist():
        path = output / "run.json"
        temporary = output / ".run.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    class Bound(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if time.monotonic() - started >= recipe["training"]["maximum_training_seconds"]:
                raise TimeoutError("Pilot training time budget exhausted")

        def on_step_end(self, args, state, control, **kwargs):
            receipt["optimizer_steps"] = state.global_step
            receipt["optimization_started"] = state.global_step > 0
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
            persist()
            require(state.global_step <= recipe["training"]["maximum_optimizer_steps"], "Optimizer exceeded its step limit")
            self.on_step_begin(args, state, control)

        def on_log(self, args, state, control, logs=None, **kwargs):
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": state.global_step, "metrics": logs or {}}, allow_nan=False) + "\n")

    persist()
    trainer = None
    try:
        set_seed(recipe["training"]["seed"])
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer_receipt"].parent, local_files_only=True)
        # A distinct loading view preserves the upstream files and tensor values.
        # Hard links require the verified base and output to share a filesystem.
        view = prepare_loading_view(base, metadata, paths["loader_index"], output / "base-loading-view")
        receipt["derived_loader_index_sha256"] = sha256(view / "model.safetensors.index.json")
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            view.resolve(), local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa", use_cache=False,
            output_loading_info=True,
        )
        require(not any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
                "Native model loading left missing or mismatched tensors")
        require(type(model).__name__ == recipe["base"]["architecture"]
                and sum(p.numel() for p in model.parameters()) == recipe["base"]["expected_parameters"],
                "Loaded base architecture differs")
        model = get_peft_model(model, LoraConfig(**recipe["adapter"]))
        trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
        require(bool(trainable) and all(".lora_A." in name or ".lora_B." in name for name, _ in trainable),
                "Non-adapter base weights became trainable")
        receipt["trainable_parameters"] = sum(value.numel() for _, value in trainable)
        require(receipt["trainable_parameters"] > 0, "No trainable adapter parameters")
        receipt["status"] = "optimizing"
        persist()
        trainer = Trainer(model=model, args=training_arguments(recipe, output / "checkpoints"),
                          data_collator=ExactTargetCollator(tokenizer.pad_token_id),
                          train_dataset=PreparedRows(rows["train"]), eval_dataset=PreparedRows(rows["validation"]),
                          processing_class=tokenizer, compute_loss_func=target_token_loss, callbacks=[Bound()])
        result = trainer.train()
        require(trainer.state.global_step == recipe["training"]["maximum_optimizer_steps"],
                "Pilot ended before the complete declared optimization budget")
        trainer.save_model(str(output / "adapter"))
        tokenizer.save_pretrained(output / "adapter")
        receipt["metrics"] = result.metrics
        receipt["optimizer_steps"] = trainer.state.global_step
        receipt["adapter_files"] = {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                                    for path in sorted((output / "adapter").rglob("*")) if path.is_file()}
        receipt["status"] = "training_completed_unqualified"
        receipt["training_completed"] = True
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["failure"] = type(exc).__name__ + ": " + str(exc)
        receipt["optimizer_steps"] = trainer.state.global_step if trainer is not None else 0
        receipt["optimization_started"] = receipt["optimizer_steps"] > 0
        raise
    finally:
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        receipt["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        persist()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("recipe", "contract", "base", "base-metadata", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(Path(__file__).resolve().parents[1], args.recipe, args.contract,
                 args.base, args.base_metadata, args.output)
    print(json.dumps(result, indent=2))
