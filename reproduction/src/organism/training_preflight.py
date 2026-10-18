"""Local metadata, collator and gradient checks. Never calls an optimizer."""

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import tempfile
import time

from .training_inputs import (CORPORA, ExactTargetCollator, PreparedRows, admission_errors, check_versions, load,
                              load_inputs, make_recipe, offline_environment, require, sha256)
from .training_source import expected_shapes, inspect_recipe_source
from .train_specialist import target_token_loss, training_arguments


def architecture_inventory(config, adapter):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, GraniteConfig
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(GraniteConfig(**config))
        base_shapes = {name: list(value.shape) for name, value in model.named_parameters()}
        require(base_shapes == expected_shapes(config), "Installed architecture differs from source tensor shapes")
        base_parameters = sum(value.numel() for value in model.parameters())
        model = get_peft_model(model, LoraConfig(**adapter))
        trainable = {name: {"shape": list(value.shape), "parameters": value.numel()}
                     for name, value in model.named_parameters() if value.requires_grad}
    require(all(value.device.type == "meta" for value in model.parameters()), "Architecture inspection allocated real base weights")
    require(trainable and all(".lora_A." in name or ".lora_B." in name for name in trainable), "Unexpected trainable base parameter")
    target_count = sum(hasattr(module, "lora_A") for module in model.modules())
    return {"base_parameters": base_parameters, "adapter_parameters": sum(item["parameters"] for item in trainable.values()),
            "target_projections": target_count, "trainable_tensors": trainable,
            "base_shapes": base_shapes, "meta_only": True}


def inspect_collator(rows, config):
    collator = ExactTargetCollator(config["pad_token_id"])
    results = {}
    for split, examples in rows.items():
        checked = supervised = padded = supervised_eos = 0
        for begin in range(0, len(examples), 4):
            group = examples[begin:begin + 4]
            batch = collator([PreparedRows(group)[index] for index in range(len(group))])
            for index, row in enumerate(group):
                size = len(row["input_ids"])
                require(batch["input_ids"][index, :size].tolist() == row["input_ids"], "Collator changed real input tokens")
                require(batch["labels"][index, :size].tolist() == row["labels"], "Collator changed target labels")
                require(batch["attention_mask"][index, :size].tolist() == [1] * size, "Real input masked as padding")
                require(all(value == -100 for value in batch["labels"][index, size:].tolist()), "Padding contributes to loss")
                require(all(value == 0 for value in batch["attention_mask"][index, size:].tolist()), "Padding remains attended")
                require(all(value == config["pad_token_id"] for value in batch["input_ids"][index, size:].tolist()), "Invalid padding token")
                checked += 1
                supervised += sum(value != -100 for value in row["labels"])
                supervised_eos += sum(value == config["eos_token_id"] for value in row["labels"])
                padded += batch["input_ids"].shape[1] - size
        results[split] = {"rows": checked, "supervised_tokens": supervised,
                          "supervised_end_tokens": supervised_eos, "padding_tokens_checked": padded,
                          "pad_equals_eos": config["pad_token_id"] == config["eos_token_id"]}
    return results


def tiny_gradient_check(recipe, temporary):
    """The tiny random model verifies loss behavior, not specialist capability."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import GraniteConfig, GraniteForCausalLM, Trainer
    torch.manual_seed(recipe["training"]["seed"])
    config = GraniteConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, pad_token_id=2, eos_token_id=2,
                           max_position_embeddings=64, use_cache=False, attention_dropout=0.0)
    base = GraniteForCausalLM(config)
    model = get_peft_model(base, LoraConfig(**{**recipe["adapter"], "r": 4, "lora_alpha": 8, "lora_dropout": 0.0}))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    require(model.is_gradient_checkpointing, "Non-reentrant gradient checkpointing did not enable")
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    examples = [{"input_ids": [5, 6, 7, 8, 9, 2], "labels": [-100, -100, -100, -100, 9, 2]},
                {"input_ids": [7, 6, 5, 4, 3, 9, 8, 2], "labels": [-100, -100, -100, 4, 3, 9, 8, 2]}]
    collator = ExactTargetCollator(config.pad_token_id)
    args = training_arguments(recipe, temporary, cpu_preflight=True)
    trainer = Trainer(model=model, args=args, data_collator=collator,
                      train_dataset=PreparedRows(examples), eval_dataset=PreparedRows(examples),
                      compute_loss_func=target_token_loss)
    microbatches = [collator([row]) for row in examples]
    count = trainer._get_num_items_in_batch(microbatches, torch.device("cpu"))
    require(int(count) == 7, "Trainer counted prompt, padding or unshifted labels")
    full = collator(examples)
    loss = trainer.compute_loss(model, dict(full), num_items_in_batch=count)
    require(torch.isfinite(loss).item(), "Nonfinite tiny-model loss")
    loss.backward()
    full_gradients = {name: value.grad.detach().clone() for name, value in model.named_parameters() if value.grad is not None}
    require(full_gradients and all(".lora_" in name for name in full_gradients), "Frozen base accumulated gradients")
    require(any(value.abs().sum().item() > 0 for value in full_gradients.values()), "No adapter gradient")
    model.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    for batch in microbatches:
        # Exercise Trainer/Accelerate's actual backward path, without creating
        # or stepping an optimizer.
        part = trainer.training_step(model, dict(batch), num_items_in_batch=count)
        accumulated_loss += part.item()
    gradients_equal = all(torch.allclose(value.grad, full_gradients[name], atol=1e-7, rtol=2e-4)
                          for name, value in model.named_parameters() if name in full_gradients)
    require(gradients_equal and abs(loss.item() - accumulated_loss) < 1e-6, "Gradient accumulation changes target-token weighting")
    require(all(torch.equal(value, before[name]) for name, value in model.named_parameters()), "A test changed model parameters")
    require(trainer.optimizer is None and trainer.state.global_step == 0, "Local preflight must not run an optimizer")
    return {"passed": True, "scope": "Tiny randomly initialized CPU model; no pretrained or specialist weights",
            "tiny_total_parameters": sum(value.numel() for value in model.parameters()),
            "loss": loss.item(), "accumulated_loss": accumulated_loss, "supervised_tokens": int(count),
            "gradient_tensors": len(full_gradients), "accumulated_gradients_match_full_batch": gradients_equal,
            "base_frozen": True, "parameters_unchanged": True, "optimizer_steps": 0,
            "actual_trainer_backward_path_verified": True,
            "gradient_checkpointing_enabled": True, "gradient_checkpointing_use_reentrant": False,
            "accelerator_accumulation_steps": trainer.accelerator.gradient_accumulation_steps,
            "trainer_warmup_steps": args.get_warmup_steps(args.max_steps),
            "cuda_training_configuration_verified": False}


def run(root, output, *, corpus="legacy_runtime", recipe_path=None):
    offline_environment()
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"schema": "organism-training-preflight-v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Local input, architecture, collation and gradient checks only", "failure": None,
              "optimization_started": False, "optimizer_steps": 0, "training_completed": False,
              "qualification_passed": False, "native_cuda_verified": False}
    try:
        report["versions"] = check_versions()
        if recipe_path is None:
            recipe = make_recipe(root, corpus=corpus)
            recipe_bytes = (json.dumps(recipe, indent=2) + "\n").encode()
        else:
            recipe_bytes = Path(recipe_path).read_bytes()
            recipe = json.loads(recipe_bytes)
            corpus = recipe["corpus"]
        if "audit_versions" in recipe:
            report["audit_versions"] = recipe["audit_versions"]
        saved_recipe = output / "recipe.draft.json"
        saved_recipe.write_bytes(recipe_bytes)
        paths, rows, config = load_inputs(root, recipe)
        report["recipe_sha256"] = sha256(saved_recipe)
        report["admission_blockers"] = admission_errors(load(root / "organism/experiment-contract.draft.json"), recipe, sha256(saved_recipe))
        report["architecture"] = architecture_inventory(config, recipe["adapter"])
        require(report["architecture"]["base_parameters"] == recipe["base"]["expected_parameters"], "Selected parameter count differs")
        report["native_source_headers"] = inspect_recipe_source(paths, recipe["base"]["expected_parameters"])
        report["collation"] = inspect_collator(rows, config)
        with tempfile.TemporaryDirectory(prefix="organism-gradient-preflight-") as temporary:
            report["gradient_check"] = tiny_gradient_check(recipe, temporary)
        report["passed"] = True
    except BaseException as exc:
        report["passed"] = False
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        snapshot = output / "sources"
        snapshot.mkdir()
        sources = {name: root / "organism" / name for name in
                   ("training_inputs.py", "train_specialist.py", "training_source.py", "training_preflight.py",
                    "requirements-training.in", "requirements-training.lock")}
        if corpus in ("inherited_equipment_runtime", "startup_complete_runtime", "refinement_runtime"):
            sources.update({name: root / "organism" / name for name in
                            ("requirements-training-inheritance.in", "requirements-training-inheritance.lock")})
        if corpus == "refinement_runtime":
            sources.update({name: root / "organism" / name for name in
                            ("refinement_dataset.py", "audit_refinement_dataset.py")})
        if corpus == "first_lineage_runtime":
            sources.update({name: root / "organism" / name for name in
                            ("audit_first_lineage_dataset.py", "native_target_codec.py",
                             "training_parser_admission.py")})
        try:
            from transformers import Trainer
            from peft import LoraConfig
            sources.update({"transformers_trainer.py": Path(inspect.getsourcefile(Trainer)),
                            "peft_lora_config.py": Path(inspect.getsourcefile(LoraConfig))})
        except ImportError:
            report["dependency_source_snapshot_unavailable"] = True
        report["source_hashes"] = {}
        for name, path in sources.items():
            (snapshot / name).write_bytes(path.read_bytes())
            report["source_hashes"][name] = sha256(path)
        report["seconds"] = round(time.monotonic() - started, 3)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", choices=sorted(CORPORA), default="legacy_runtime")
    parser.add_argument("--recipe", type=Path, help="Check these exact recipe bytes instead of generating a default draft")
    args = parser.parse_args()
    report = run(Path(__file__).resolve().parents[1], args.output, corpus=args.corpus, recipe_path=args.recipe)
    print(json.dumps({key: report[key] for key in
                      ("passed", "seconds", "gradient_check", "collation", "admission_blockers", "optimizer_steps")}, indent=2))
