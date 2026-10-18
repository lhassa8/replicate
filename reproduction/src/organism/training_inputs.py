"""Immutable pilot inputs and a collator that preserves verified target labels."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import stat


VERSIONS = {"torch": "2.14.0", "transformers": "5.17.0", "peft": "0.21.0",
            "accelerate": "1.15.0", "jinja2": "3.1.6"}
INHERITANCE_AUDIT_VERSIONS = {"cryptography": "50.0.1", "cffi": "2.1.1", "pycparser": "3.0"}
ADMISSION_EVIDENCE_FIELDS = ("data_verification_evidence", "qualification_seal", "compute_authorization_evidence")
DATA_VERIFICATION_CHECKS = ("input_bindings", "runtime_replay", "token_reconstruction", "provenance_inventory")
SEAL_CHECKS = (
    "input_reconstruction", "source_input_reconstruction", "q6_schedule_reconstruction",
    "all_trial_seeds_committed", "baseline_development_review", "runtime_and_resource_contract",
    "development_qualification_exclusion", "worker_package_exclusion", "preoptimization_input_freeze",
)
SEAL_COMMITMENTS = (
    "qualification_inputs", "inheritance_source_inputs", "lifecycle_protocol", "q6_schedules",
    "fixed_baseline", "independent_oracles", "exclusion_audit", "custody_record",
)
SEAL_COUNTS = {
    "Q4_Q5": 24, "Q7": 20, "Q10": 64, "protected": 12,
    "inheritance_source_inputs": 32, "lifecycle_trials": 20, "q6_subcases": 15,
}
DERIVED_SEAL_CHECKS = (
    "parent_public_commitments_verified", "prior_observations_and_splits_preserved",
    "current_runtime_replay_verified", "independent_semantics_verified",
    "complete_native_targets_verified", "frozen_temperature_cases_excluded",
    "worker_package_allowlist_verified", "preoptimization_input_freeze",
)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CORPORA = {
    "legacy_runtime": {
        "source": "data/runtime-development-v2",
        "tokens": "data/runtime-development-v2-tokens-v3",
        "capture": "evidence/runtime-template-preflight-20261014-02",
        "row_audit": "evidence/runtime-template-parity-20261014-03.json",
        "runtime_audit": "evidence/runtime-data-audit-20261014-02.json",
    },
    "equipment_runtime": {
        "source": "data/equipment-runtime-development-v1",
        "tokens": "data/equipment-runtime-development-v1-tokens",
        "capture": "evidence/equipment-runtime-template-preflight-20261014-01",
        "row_audit": "evidence/equipment-runtime-template-parity-20261014-01.json",
        "runtime_audit": "evidence/equipment-dataset-audit-20261014-01.json",
    },
    "inherited_equipment_runtime": {
        "source": "data/equipment-runtime-development-v2-02",
        "tokens": "data/equipment-runtime-development-v2-02-tokens",
        "capture": "evidence/inheritance-runtime-template-preflight-20261014-02",
        "row_audit": "evidence/inheritance-runtime-template-parity-20261014-01.json",
        "runtime_audit": "evidence/inheritance-data-development-20261014-01/audit-02.json",
    },
    "startup_complete_runtime": {
        "source": "data/equipment-runtime-development-v4-01",
        "tokens": "data/equipment-runtime-development-v4-01-tokens",
        "capture": "evidence/startup-training-template-capture-20261014-01",
        "row_audit": "evidence/startup-training-coverage-20261014-01/row-audit.json",
        "runtime_audit": "evidence/startup-training-coverage-20261014-01/runtime-audit.json",
    },
    "refinement_runtime": {
        "source": "data/equipment-runtime-development-v5-refinement-01",
        "tokens": "data/equipment-runtime-development-v5-refinement-01-tokens",
        "capture": "evidence/refinement-training-integration-20261016-01/capture",
        "row_audit": "evidence/refinement-training-integration-20261016-01/row-audit.json",
        "runtime_audit": "evidence/refinement-training-integration-20261016-01/runtime-audit.json",
    },
    "first_lineage_runtime": {
        "source": "data/first-lineage-runtime-development-v6-01",
        "tokens": "data/first-lineage-runtime-development-v6-01-tokens",
        "capture": "evidence/first-lineage-training-preparation-20261016-01/capture-02",
        "row_audit": "evidence/first-lineage-training-preparation-20261016-01/row-audit.json",
        "runtime_audit": "evidence/first-lineage-training-preparation-20261016-01/runtime-audit.json",
    },
}
INHERITED_CORPORA = ("inherited_equipment_runtime", "startup_complete_runtime", "refinement_runtime")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def offline_environment():
    for name, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                        "TOKENIZERS_PARALLELISM": "false", "WANDB_DISABLED": "true", "OTEL_SDK_DISABLED": "true",
                        "AWS_EC2_METADATA_DISABLED": "true"}.items():
        os.environ[name] = value


def verified_path(root, entry):
    relative = Path(entry["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Artifact path escapes its bundle")
    path = Path(root) / relative
    require(path.resolve().is_relative_to(Path(root).resolve()) and not path.is_symlink()
            and stat.S_ISREG(path.stat().st_mode), "Artifact must be an owned bundle file")
    require(path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"], "Artifact identity changed: " + str(relative))
    return path


def make_recipe(root, *, corpus="legacy_runtime"):
    root = Path(root)
    require(corpus in CORPORA, "Unsupported runtime training corpus")
    choice = CORPORA[corpus]
    experiment = load(root / "organism/experiment-contract.draft.json")
    source = root / "organism" / choice["source"]
    tokens = root / "organism" / choice["tokens"]
    capture = root / "organism" / choice["capture"]
    tokenizer = root / "organism/evidence/tokenizer-source-20261013-01"
    selected = {
        "source_manifest": source / "manifest.json", "source_train": source / "train.jsonl",
        "source_validation": source / "validation.jsonl", "contracts": source / "contracts.json",
        "release": source / "release.json", "token_report": tokens / "report.json",
        "tokens_train": tokens / "train.tokens.jsonl", "tokens_validation": tokens / "validation.tokens.jsonl",
        "row_audit": root / "organism" / choice["row_audit"],
        "runtime_audit": root / "organism" / choice["runtime_audit"],
        "capture_report": capture / "report.json", "capture_prompts": capture / "server-prompts.jsonl",
        "base_config": root / "organism/evidence/source-check-20261013-01/config.json",
        "tokenizer_receipt": tokenizer / "receipt.json",
        "dependency_lock": root / "organism/requirements-training.lock",
        "base_metadata": root / "organism/evidence/training-source-20261014-01/metadata.json",
        "base_index": root / "organism/evidence/training-source-20261014-01/model.safetensors.index.json",
        "base_header_receipt": root / "organism/evidence/training-source-20261014-01/header-receipt.json",
        "loader_index": root / "organism/evidence/training-loader-index-20261014-01/model.safetensors.index.json",
        "index_repair_receipt": root / "organism/evidence/training-loader-index-20261014-01/receipt.json",
    }
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        selected["base_header/" + name] = root / "organism/evidence/training-source-20261014-01" / (name + ".header")
    for name in ("training_inputs.py", "train_specialist.py", "training_source.py", "training_preflight.py",
                 "check_runtime_templates.py", "tokenize_dataset.py", "audit_runtime_dataset.py",
                 "check_template_parity.py", "runtime_rows.py", "audit_equipment_dataset.py",
                 "audit_equipment_diagnostics.py", "audit_application.py"):
        selected["source/" + name] = root / "organism" / name
    if corpus in INHERITED_CORPORA:
        selected["dependency_lock"] = root / "organism/requirements-training-inheritance.lock"
        for name in ("audit_inheritance_dataset.py", "audit_inherited_diagnostics.py",
                     "experience.py", "experience_issue.py", "diagnostic_oracle.py", "enrollment.py", "capsule.py"):
            selected["source/" + name] = root / "organism" / name
    if corpus in ("startup_complete_runtime", "refinement_runtime"):
        for name in ("startup_dataset.py", "audit_startup_dataset.py"):
            selected["source/" + name] = root / "organism" / name
    if corpus == "refinement_runtime":
        for name in ("refinement_dataset.py", "audit_refinement_dataset.py"):
            selected["source/" + name] = root / "organism" / name
        selected["refinement_references"] = source / "references.jsonl"
        selected["refinement_parent_manifest"] = source / "parent-manifest.json"
    if corpus == "first_lineage_runtime":
        for name in ("audit_first_lineage_dataset.py", "native_target_codec.py",
                     "training_parser_admission.py"):
            selected["source/" + name] = root / "organism" / name
        evidence = root / "organism/evidence/first-lineage-training-preparation-20261016-01"
        for key, name in (
            ("native_parser_report", "native-target-roundtrip.json"),
            ("native_parser_input", "native-target-parser-input.jsonl"),
            ("native_parser_output", "native-target-parser-output.jsonl"),
            ("token_parser_binding", "token-parser-binding.json"),
        ):
            selected[key] = evidence / name
        parent = root / "organism/evidence/refinement-freeze-20261016-01/public"
        selected["parent_contract"] = parent / "experiment-contract.frozen.json"
        selected["parent_seal"] = parent / "qualification-seal.json"
        selected["parent_recipe"] = root / "organism/training-recipe.current-v5.refinement-01.draft.json"
    for item in load(tokenizer / "receipt.json")["files"]:
        selected["tokenizer/" + item["name"]] = tokenizer / item["name"]
    report = load(tokens / "report.json")
    epochs, batch, accumulation = 2, 1, 16
    count = report["splits"]["train"]["decisions"]
    updates_per_epoch = math.ceil(count / (batch * accumulation))
    recipe = {
        "schema": "organism-lora-pilot-recipe-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus,
        "status": "draft_pending_experiment_freeze_and_data_admission", "experiment_id": experiment["experiment_id"],
        "base": {"repository": experiment["model"]["repository"], "revision": experiment["model"]["revision"],
                 "expected_parameters": experiment["model"]["parameters_reported_by_pinned_upstream_api"],
                 "architecture": "GraniteForCausalLM", "load_dtype": "bfloat16",
                 "trust_remote_code": False, "local_files_only": True, "use_safetensors": True},
        "inputs": {name: {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
                   for name, path in selected.items()},
        "versions": VERSIONS, "method": "LoRA", "quantized_training": False,
        "adapter": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
                    "target_modules": TARGET_MODULES},
        "training": {"epochs": epochs, "per_device_batch_size": batch, "gradient_accumulation_steps": accumulation,
                     "updates_per_epoch": updates_per_epoch, "maximum_optimizer_steps": updates_per_epoch * epochs,
                     "learning_rate": 0.0001, "weight_decay": 0.0, "max_grad_norm": 1.0,
                     "warmup_steps": math.ceil(updates_per_epoch * epochs * 0.1), "lr_scheduler_type": "cosine",
                     "gradient_checkpointing": True, "use_reentrant_checkpointing": False,
                     "context_tokens": 4096, "maximum_target_tokens": 192, "seed": 20261014,
                     "maximum_training_seconds": 4 * 3600, "concurrent_devices": 1,
                     "validation_policy": "Record loss each epoch; preserve both checkpoints. Loss alone cannot select a qualified release.",
                     "loss_normalization": "Mean next-token cross-entropy over nonmasked target tokens across each accumulated batch."},
        "counts": {split: report["splits"][split]["decisions"] for split in ("train", "validation")},
        "training_admitted": False, "optimization_started": False, "qualification_passed": False,
        "limitations": ["Pilot hyperparameters require freezing before optimization.",
                        "This recipe does not authorize AWS spending or substitute for an externally bounded GPU worker.",
                        "Native weights, CUDA fit and training execution remain unverified.",
                        "Automated data verification, experiment freeze and withheld-API qualification remain separate requirements."],
    }
    if corpus in INHERITED_CORPORA:
        recipe["audit_versions"] = INHERITANCE_AUDIT_VERSIONS
    if corpus == "refinement_runtime":
        recipe["experiment_id"] += "-refinement-01"
        recipe["limitations"].append(
            "This data refinement starts a new adapter from the same pinned base; the original trained adapter remains a comparator.")
    if corpus == "first_lineage_runtime":
        recipe["experiment_id"] += "-first-lineage-v6-01"
        recipe["training"]["maximum_training_seconds"] = 2700
        recipe["target_codec"] = "granite-json-scalar-targets-v1"
        recipe["limitations"].append(
            "Fresh adapter on a narrow prior-observation subset with versioned tools; original qualification remains deferred.")
    return recipe


def worker_limits(corpus, mode):
    """Exact reviewed envelopes; the shorter v6 profile cannot extend a lease."""
    require(corpus in CORPORA and mode in ("verify", "train"), "Unknown worker envelope")
    focused = corpus == "first_lineage_runtime"
    return {
        "workload_seconds": 1800 if mode == "verify" else 3600 if focused else 14400,
        "worker_seconds": 7200 if focused else 21600,
        "output_bytes": 2 * 1024**3, "concurrent_gpus": 1,
    }


def check_versions():
    actual = {name: version(name) for name in VERSIONS}
    require(actual == VERSIONS, "Training package versions differ from the recipe")
    return actual


def validate_recipe(recipe):
    require(recipe["schema"] == "organism-lora-pilot-recipe-v1" and recipe["method"] == "LoRA"
            and recipe["quantized_training"] is False, "Unsupported pilot method")
    require(recipe["versions"] == VERSIONS, "Unexpected training versions")
    require(recipe.get("corpus", "legacy_runtime") in CORPORA, "Unsupported runtime training corpus")
    if recipe.get("corpus") in INHERITED_CORPORA:
        require(recipe.get("audit_versions") == INHERITANCE_AUDIT_VERSIONS,
                "Inherited source-verification dependencies differ")
    require(recipe["adapter"]["target_modules"] == TARGET_MODULES and recipe["adapter"]["bias"] == "none"
            and recipe["adapter"]["task_type"] == "CAUSAL_LM", "Unexpected adapter targets")
    require(type(recipe["adapter"]["r"]) is int and 1 <= recipe["adapter"]["r"] <= 64, "Invalid adapter rank")
    cfg = recipe["training"]
    for field in ("epochs", "per_device_batch_size", "gradient_accumulation_steps", "maximum_optimizer_steps", "maximum_training_seconds"):
        require(type(cfg[field]) is int and cfg[field] > 0, "Invalid positive training limit: " + field)
    expected = math.ceil(recipe["counts"]["train"] / (cfg["per_device_batch_size"] * cfg["gradient_accumulation_steps"]))
    require(cfg["updates_per_epoch"] == expected and cfg["maximum_optimizer_steps"] == expected * cfg["epochs"],
            "Optimizer step budget disagrees with the complete dataset")
    require(cfg["concurrent_devices"] == 1 and cfg["context_tokens"] == 4096 and cfg["maximum_target_tokens"] == 192,
            "Device or sequence contract changed")
    require(2_000_000_000 <= recipe["base"]["expected_parameters"] <= 4_000_000_000
            and recipe["base"]["architecture"] == "GraniteForCausalLM"
            and recipe["base"]["local_files_only"] is True and recipe["base"]["trust_remote_code"] is False
            and recipe["base"]["use_safetensors"] is True, "Base model contract changed")
    if recipe.get("corpus") == "first_lineage_runtime":
        require(recipe["counts"] == {"train": 716, "validation": 258}
                and cfg["maximum_optimizer_steps"] == 90
                and cfg["maximum_training_seconds"] == 2700
                and recipe.get("target_codec") == "granite-json-scalar-targets-v1",
                "First-lineage recipe changed its verified targets or bounded workload")


def validate_row(row, *, vocab_size, eos_token_id):
    ids, labels, prefix = row["input_ids"], row["labels"], row["prompt_tokens"]
    require(isinstance(ids, list) and 1 < len(ids) <= 4096 and len(labels) == len(ids), "Invalid row lengths")
    require(type(prefix) is int and 0 < prefix < len(ids) and len(ids) - prefix <= 192, "Invalid target boundary")
    require(all(type(value) is int and 0 <= value < vocab_size for value in ids), "Invalid input token")
    require(all(type(value) is int for value in labels) and labels == [-100] * prefix + ids[prefix:],
            "Target labels changed or prompt tokens entered the loss")
    require(eos_token_id in labels[prefix:], "Target lost its end token")


class PreparedRows:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {name: list(self.rows[index][name]) for name in ("input_ids", "labels")}


class ExactTargetCollator:
    """Right padding only; an EOS equal to pad remains supervised in real tokens."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        require(bool(features), "Cannot collate an empty batch")
        width = max(len(item["input_ids"]) for item in features)
        require(width <= 4096, "Batch exceeds context limit")
        ids, labels, masks = [], [], []
        for item in features:
            size = len(item["input_ids"])
            require(size == len(item["labels"]) and size > 0, "Invalid batch row")
            ids.append(item["input_ids"] + [self.pad_token_id] * (width - size))
            labels.append(item["labels"] + [-100] * (width - size))
            masks.append([1] * size + [0] * (width - size))
        return {key: torch.tensor(value, dtype=torch.long) for key, value in
                (("input_ids", ids), ("labels", labels), ("attention_mask", masks))}


def load_inputs(root, recipe, *, independent_audit=True):
    validate_recipe(recipe)
    if recipe.get("corpus") in INHERITED_CORPORA:
        require({name: version(name) for name in INHERITANCE_AUDIT_VERSIONS} == recipe["audit_versions"],
                "Installed source-verification dependencies differ from the recipe")
    paths = {name: verified_path(root, entry) for name, entry in recipe["inputs"].items()}
    token_report, row_audit = load(paths["token_report"]), load(paths["row_audit"])
    require(token_report["status"] == "serving_prefix_and_native_targets_verified_not_admitted"
            and token_report["source_manifest_sha256"] == sha256(paths["source_manifest"]),
            "Token provenance does not bind the reference dataset")
    require(row_audit["passed"] is True and row_audit["token_report_sha256"] == sha256(paths["token_report"])
            and row_audit["capture_report_sha256"] == sha256(paths["capture_report"]), "Independent row audit differs")
    config = load(paths["base_config"])
    rows, identities, profiles = {}, set(), set()
    for split in ("train", "validation"):
        path = paths["tokens_" + split]
        require(sha256(path) == token_report["splits"][split]["sha256"], "Token split hash differs")
        rows[split] = [json.loads(line) for line in path.read_text().splitlines()]
        require(len(rows[split]) == recipe["counts"][split], "Token split count differs")
        split_ids, split_profiles = set(), set()
        decisions = set()
        for row in rows[split]:
            validate_row(row, vocab_size=config["vocab_size"], eos_token_id=config["eos_token_id"])
            key = (row["case_id"], row["decision_index"])
            require(key not in decisions, "Duplicate assistant decision")
            decisions.add(key)
            split_ids.add(row["case_id"])
            split_profiles.add(row["profile_id"])
        require(not identities & split_ids and not profiles & split_profiles, "Development split leakage")
        identities.update(split_ids)
        profiles.update(split_profiles)
    if independent_audit:
        offline_environment()
        from .check_runtime_templates import check
        audit = check(paths["capture_report"].parent, paths["token_report"].parent,
                      paths["tokenizer_receipt"].parent, paths["source_manifest"].parent)
        require(audit["passed"] is True, "Independent training-row reconstruction failed")
    if recipe.get("corpus") == "first_lineage_runtime":
        from .training_parser_admission import verify
        verify(paths, recipe, rows)
    return paths, rows, config


def admission_errors(contract, recipe, recipe_sha):
    """A draft recipe is reviewable; optimization requires a separately frozen contract."""
    errors = []
    if contract.get("status") != "frozen" or not contract.get("frozen_at") or contract.get("phase0_completed") is not True:
        errors.append("Experiment contract is not frozen after Phase 0")
    if contract.get("experiment_id") != recipe["experiment_id"]:
        errors.append("Experiment identity differs")
    admission = contract.get("training_admission", {})
    expected = {"recipe_sha256": recipe_sha,
                "candidate_manifest_sha256": recipe["inputs"]["source_manifest"]["sha256"],
                "token_report_sha256": recipe["inputs"]["token_report"]["sha256"]}
    if any(admission.get(key) != value for key, value in expected.items()):
        errors.append("Frozen admission does not bind the exact recipe and training data")
    if (admission.get("data_verification_method") != "automated"
            or admission.get("data_verification_verified") is not True
            or not admission.get("data_verification_evidence")):
        errors.append("Automated data verification is not recorded")
    if admission.get("qualification_splits_sealed") is not True or not admission.get("qualification_seal"):
        errors.append("Independent qualification families are not sealed")
    if admission.get("compute_authorization_verified") is not True or not admission.get("compute_authorization_evidence"):
        errors.append("Bounded training compute authorization is not recorded")
    if contract.get("model", {}).get("revision") != recipe["base"]["revision"]:
        errors.append("Frozen base-model revision differs")
    return errors


def verify_admission_evidence(root, contract, recipe, recipe_sha):
    """Check frozen evidence identities without treating an automated audit as human review."""
    admission = contract["training_admission"]
    paths = {field: verified_path(root, admission[field]) for field in ADMISSION_EVIDENCE_FIELDS}
    data = load(paths["data_verification_evidence"])
    require(data.get("schema") == "organism-automated-data-verification-v1"
            and data.get("experiment_id") == recipe["experiment_id"]
            and data.get("recipe_sha256") == recipe_sha
            and data.get("input_artifacts") == recipe["inputs"]
            and data.get("counts") == recipe["counts"],
            "Automated data evidence does not bind the exact recipe and inputs")
    require(data.get("passed") is True
            and data.get("checks") == {name: True for name in DATA_VERIFICATION_CHECKS}
            and data.get("human_review_required") is False
            and data.get("human_review_verified") is False,
            "Automated data evidence is incomplete or misstates human review")
    parent = None
    if recipe.get("corpus") == "first_lineage_runtime":
        parent_paths = {name: verified_path(root, recipe["inputs"]["parent_" + name])
                        for name in ("contract", "seal", "recipe")}
        parent = (load(parent_paths["seal"]), load(parent_paths["contract"]),
                  load(parent_paths["recipe"]), sha256(parent_paths["recipe"]))
        runtime = load(verified_path(root, recipe["inputs"]["runtime_audit"]))
        require(runtime.get("passed") is True
                and runtime.get("source_manifest_sha256") == recipe["inputs"]["source_manifest"]["sha256"]
                and all(runtime.get(key) is True for key in (
                    "prior_observations_and_splits_preserved", "current_runtime_replay_verified",
                    "independent_semantics_verified", "frozen_temperature_cases_excluded"))
                and runtime.get("sealed_qualification_data_accessed") is False,
                "Derived admission lacks its bound subset and runtime verification")
    verify_qualification_seal(load(paths["qualification_seal"]), contract, recipe, recipe_sha, parent=parent)
    compute = load(paths["compute_authorization_evidence"])
    verify_compute_authorization(compute, recipe)
    if compute.get("schema") in ("organism-standing-gpu-allocation-v1", "organism-standing-gpu-allocation-v2"):
        approval = load(verified_path(root, compute["standing_authorization"]))
        require(" ".join(approval.get("answer", "").split()).lower()
                == "approve. and stop asking for anything less that $400"
                and approval.get("standing_in_scope_spending_approval_under_usd") == 400
                and approval.get("blanket_s3_upload_approval_already_received") is True,
                "Standing compute authorization is not bound to the user's recorded instruction")
    return paths


def verify_qualification_seal(seal, contract, recipe, recipe_sha, *, parent=None):
    """Validate the public commitment receipt, not private answers on a GPU.

    The receipt must be produced by a separate local audit of the private assets.
    Its identity is bound by the frozen contract. It is not proof of inaccessible
    custody or of completed native qualification.
    """
    derived = seal.get("schema") == "organism-derived-qualification-seal-v1"
    require(seal.get("schema") in ("organism-qualification-seal-v1", "organism-derived-qualification-seal-v1")
            and seal.get("status") == "sealed_before_optimization"
            and seal.get("experiment_id") == recipe["experiment_id"]
            and seal.get("recipe_sha256") == recipe_sha
            and seal.get("candidate_manifest_sha256") == recipe["inputs"]["source_manifest"]["sha256"]
            and seal.get("token_report_sha256") == recipe["inputs"]["token_report"]["sha256"],
            "Qualification seal does not bind the exact experiment, recipe and data")
    if derived:
        require(recipe.get("corpus") == "first_lineage_runtime" and parent is not None,
                "Derived seal requires the exact public parent admission")
        old_seal, old_contract, old_recipe, old_sha = parent
        require(old_seal.get("schema") == "organism-qualification-seal-v1"
                and old_recipe.get("corpus") == "refinement_runtime"
                and old_sha == recipe["inputs"]["parent_recipe"]["sha256"]
                and not admission_errors(old_contract, old_recipe, old_sha)
                and old_contract["training_admission"]["qualification_seal"] == recipe["inputs"]["parent_seal"],
                "Derived seal parent differs from the admitted refinement")
        verify_qualification_seal(old_seal, old_contract, old_recipe, old_sha)
        require(seal.get("parent_public_inputs") == {
                    name: recipe["inputs"]["parent_" + name] for name in ("contract", "seal", "recipe")}
                and seal.get("inherited_checks") == old_seal["checks"]
                and "checks" not in seal
                and seal.get("derived_checks") == {name: True for name in DERIVED_SEAL_CHECKS}
                and seal.get("private_assets_reaudited") is False
                and seal.get("sealed_qualification_data_accessed_for_derivation") is False
                and seal.get("qualification_deferred") is True,
                "Derived seal misstates inherited checks or new verification")
        require(all(seal.get(key) == old_seal[key] for key in (
                    "commitments", "counts", "qualification_requirements", "resource_targets"))
                and contract.get("qualification_deferred") is True
                and recipe["base"] == old_recipe["base"],
                "Derived seal changed an original qualification commitment")
    else:
        require(recipe.get("corpus") != "first_lineage_runtime"
                and seal.get("checks") == {name: True for name in SEAL_CHECKS},
                "Qualification seal has incomplete checks or wrong derivation")
    require(seal.get("counts") == SEAL_COUNTS,
            "Qualification seal has incomplete checks or trial allocation")
    commitments = seal.get("commitments")
    require(isinstance(commitments, dict) and set(commitments) == set(SEAL_COMMITMENTS),
            "Qualification seal commitment inventory is incomplete")
    for value in commitments.values():
        require(isinstance(value, dict) and set(value) == {"bytes", "sha256"}
                and type(value["bytes"]) is int and value["bytes"] > 0
                and isinstance(value["sha256"], str) and len(value["sha256"]) == 64
                and all(character in "0123456789abcdef" for character in value["sha256"]),
                "Qualification seal contains an invalid artifact commitment")
    try:
        sealed_at = datetime.fromisoformat(seal["sealed_at"])
        frozen_at = datetime.fromisoformat(contract["frozen_at"])
        require(sealed_at.utcoffset() is not None and frozen_at.utcoffset() is not None
                and sealed_at <= frozen_at <= datetime.now(timezone.utc),
                "Qualification seal must precede the actual experiment freeze")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Qualification seal has an invalid freeze time") from exc
    requirements = contract.get("qualification_requirements")
    resources = contract.get("resource_targets")
    require(isinstance(requirements, dict) and set(requirements) == {f"Q{index}" for index in range(1, 11)}
            and seal.get("qualification_requirements") == requirements,
            "Qualification seal differs from the experiment's Q1-Q10 requirements")
    require(isinstance(resources, dict) and seal.get("resource_targets") == resources
            and resources.get("required_device_memory_mib") == 8192
            and resources.get("peak_working_memory_fraction_of_device") == 0.75,
            "Qualification seal differs from the experiment's small-device resource contract")
    for field in ("total_package_staging_and_storage_budget_bytes", "transfer_bandwidth_budget_bytes_per_second"):
        require(type(resources.get(field)) is int and resources[field] > 0,
                "Qualification resource ceiling remains unfrozen: " + field)
    require(seal.get("boundary") == "training_and_deployment_input_allowlists"
            and seal.get("author_accessed_expected_answers") is True
            and seal.get("independent_human_blinding_claimed") is False
            and seal.get("inaccessible_custody_claimed") is False,
            "Qualification seal misstates its custody boundary")
    require(seal.get("optimizer_steps_at_seal") == 0
            and type(seal["optimizer_steps_at_seal"]) is int
            and seal.get("qualification_executed") is False
            and seal.get("qualification_passed") is False,
            "Qualification input seal overclaims execution or follows optimization")


def verify_compute_authorization(authorization, recipe):
    """Enforce recorded GPU ceilings; live lease use is checked by the launcher."""
    if authorization.get("schema") in ("organism-standing-gpu-allocation-v1", "organism-standing-gpu-allocation-v2"):
        focused = authorization["schema"] == "organism-standing-gpu-allocation-v2"
        expected_corpus = "first_lineage_runtime" if focused else "refinement_runtime"
        expected_scope = ("one separately admitted first-lineage GPU run" if focused
                          else "one separately admitted organism refinement GPU run")
        amount = authorization.get("maximum_total_spend")
        before, after = (authorization.get(k) for k in
                         ("project_reservations_before_usd", "project_reservations_after_usd"))
        require(authorization.get("authority") == "direct human reply in this chat"
                and authorization.get("currency") == "USD"
                and authorization.get("scope") == expected_scope
                and recipe.get("corpus") == expected_corpus,
                "Standing allocation does not cover this recipe")
        require(all(type(n) in (int, float) and math.isfinite(n) for n in (amount, before, after))
                and 0 < amount <= 35 and 0 <= before < after < 400 and after == before + amount,
                "GPU allocation exceeds its reserved project envelope")
        require(type(authorization.get("maximum_leases")) is int and authorization["maximum_leases"] == 1
                and type(authorization.get("maximum_hours_per_lease")) is int
                and authorization["maximum_hours_per_lease"] == (2 if focused else 6)
                and authorization.get("leases_must_be_sequential") is True
                and authorization.get("automatic_expiry_and_cleanup_required") is True
                and authorization.get("training_requires_frozen_experiment_contract") is True,
                "GPU allocation omitted its single-lease expiry limits")
        require(recipe["training"]["concurrent_devices"] == 1
                and recipe["training"]["maximum_training_seconds"] <= (2700 if focused else 14400)
                and isinstance(authorization.get("standing_authorization"), dict),
                "Training workload exceeds its allocation")
        return
    require(recipe.get("corpus") not in ("refinement_runtime", "first_lineage_runtime"),
            "The original pilot authorization cannot admit a new refinement run")
    require(authorization.get("authority") == "direct human reply in this chat"
            and authorization.get("currency") == "USD"
            and isinstance(authorization.get("question_item_id"), str)
            and bool(authorization["question_item_id"])
            and isinstance(authorization.get("user_reply"), str) and bool(authorization["user_reply"])
            and authorization.get("scope") ==
            "GPU proposal in organism/TRAINING_WORKER.md: model verification and one admitted fine-tuning pilot",
            "Compute evidence is not the recorded GPU pilot authorization")
    amount = authorization.get("maximum_total_spend")
    leases, hours = authorization.get("maximum_leases"), authorization.get("maximum_hours_per_lease")
    require(type(amount) in (int, float) and math.isfinite(amount) and 0 < amount <= 35
            and type(leases) is int and 1 <= leases <= 2
            and type(hours) is int and 1 <= hours <= 6
            and authorization.get("leases_must_be_sequential") is True
            and authorization.get("automatic_expiry_and_cleanup_required") is True
            and authorization.get("training_requires_frozen_experiment_contract") is True,
            "Compute evidence exceeds or omits the authorized spend, lease or cleanup limits")
    require(recipe["training"]["concurrent_devices"] == 1
            and recipe["training"]["maximum_training_seconds"] <= hours * 3600,
            "Training recipe exceeds its recorded compute authorization")
