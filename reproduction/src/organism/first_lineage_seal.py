"""Freeze the narrow v6 experiment using public parent commitments only.

No private qualification case is read. Exact parent observations and splits are
checked by the independent development auditor; the parent's private-asset
checks remain explicitly inherited, rather than reported as newly executed.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

from .training_inputs import (
    DATA_VERIFICATION_CHECKS, DERIVED_SEAL_CHECKS, admission_errors, load, load_inputs,
    make_recipe, require, sha256, validate_recipe, verified_path,
    verify_admission_evidence, verify_qualification_seal,
)
from .training_worker.build import build


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def artifact(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Freeze artifact escaped the workspace")
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def check_public_files(files, reserved):
    require(0 < len(files) <= 128, "Worker inventory exceeds its bound")
    pattern = re.compile(r"(?:qual-(?:source-|life-)?|q6-)[0-9a-f]{32}")
    for name, raw in files.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and not any(part in name for part in (
                    "organism/evaluation/", "temperature-cases.frozen.json", "/tests/",
                    "first_lineage_seal.py", "refinement_seal.py", "evaluation_seal.py",
                    "qualification_inputs.py", "qualification_protocol.py", "check_qualification_inputs.py")),
                "An evaluator, fixture or freeze compiler entered the worker")
        text = raw.decode("utf-8", errors="replace")
        require(not pattern.search(text) and not any(value in text for value in reserved),
                "A reserved evaluation identifier entered the worker")
    return {"passed": True, "files": len(files), "reserved_temperature_cases": len(reserved),
            "private_case_inventory_read": False, "private_case_exclusion_reaudit_claimed": False}


def derived_seal(recipe, recipe_sha, old_seal, *, at):
    seal = deepcopy(old_seal)
    inherited = seal.pop("checks")
    seal.pop("previous_seal", None)
    seal.update(
        schema="organism-derived-qualification-seal-v1",
        experiment_id=recipe["experiment_id"], recipe_sha256=recipe_sha,
        candidate_manifest_sha256=recipe["inputs"]["source_manifest"]["sha256"],
        token_report_sha256=recipe["inputs"]["token_report"]["sha256"],
        sealed_at=at, inherited_checks=inherited,
        derived_checks={name: True for name in DERIVED_SEAL_CHECKS},
        parent_public_inputs={name: deepcopy(recipe["inputs"]["parent_" + name])
                              for name in ("contract", "seal", "recipe")},
        private_assets_reaudited=False, sealed_qualification_data_accessed_for_derivation=False,
        qualification_deferred=True, optimizer_steps_at_seal=0,
        qualification_executed=False, qualification_passed=False,
        receipt_scope=("Unchanged public parent commitments; separately verified subset, runtime, targets "
                       "and package allowlist. Original private checks are inherited, not rerun."),
    )
    return seal


def compile_first_lineage(root, recipe_path, output, *, readmission=False):
    root, recipe_path, output = (Path(path).resolve() for path in (root, recipe_path, output))
    require(recipe_path.is_relative_to(root) and output.is_relative_to(root / "organism/evidence")
            and not output.exists(), "Freeze requires a new workspace evidence directory")
    recipe, recipe_sha = load(recipe_path), sha256(recipe_path)
    validate_recipe(recipe)
    require(recipe["corpus"] == "first_lineage_runtime" and recipe["optimization_started"] is False
            and recipe["training_admitted"] is False
            and recipe["inputs"] == make_recipe(root, corpus="first_lineage_runtime")["inputs"],
            "First-lineage recipe no longer binds the exact reviewed source inventory")
    paths, rows, _ = load_inputs(root, recipe)
    old_seal, old_contract, old_recipe = (load(paths["parent_" + name]) for name in ("seal", "contract", "recipe"))
    old_sha = sha256(paths["parent_recipe"])
    verify_qualification_seal(old_seal, old_contract, old_recipe, old_sha)
    require(all(recipe[key] == old_recipe[key] for key in ("base", "adapter", "versions")),
            "First-lineage preparation changed the pinned base or adapter method")
    changed = {key for key in recipe["training"] if recipe["training"][key] != old_recipe["training"][key]}
    require(changed == {"updates_per_epoch", "maximum_optimizer_steps", "warmup_steps", "maximum_training_seconds"},
            "First-lineage preparation changed another training parameter")
    admission_context = {
        "preflight_path": root / "organism/evidence/first-lineage-training-preparation-20261016-01/preflight/report.json",
        "reservation_before_usd": 275, "previous_attempt_evidence": {},
    }
    if readmission:
        from .first_lineage_readmission import review
        admission_context = review(root, recipe_path, output)
    before = admission_context["reservation_before_usd"]
    preflight_path = admission_context["preflight_path"]
    preflight = load(preflight_path)
    require(preflight["passed"] is True and preflight["optimizer_steps"] == 0
            and preflight["recipe_sha256"] == recipe_sha, "Trainer preflight does not bind the frozen recipe")
    # The application runtime and trainer have separately pinned environments.
    # Replay actual tools in the application environment without adding runtime
    # libraries to the network-isolated GPU training package.
    replay = subprocess.run(
        [str(root / ".venv-local/bin/python"), "-c",
         "import asyncio,json,sys; from pathlib import Path; "
         "from organism.audit_first_lineage_dataset import audit; "
         "print(json.dumps(asyncio.run(audit(Path(sys.argv[1])))))",
         str(paths["source_manifest"].parent)],
        cwd=root, capture_output=True, text=True, timeout=60, check=True)
    runtime = json.loads(replay.stdout)
    require(runtime == load(paths["runtime_audit"]), "Independent development subset or runtime replay changed")
    from .training_parser_admission import verify
    parser_check = verify(paths, recipe, rows)
    frozen = root / "organism/evidence/first-lineage-milestone-20261016-01/temperature-cases.frozen.json"
    require(sha256(frozen) == "aff7cf82494c5f1f194b0272245835b030990a978786c1ec51aac2d0889e9685",
            "Frozen development evaluation changed")
    reserved = {case["case_id"] for case in load(frozen)["cases"]}
    files = {item["path"]: verified_path(root, item).read_bytes() for item in recipe["inputs"].values()}
    public_check = check_public_files(files, reserved)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(paths["tokenizer/tokenizer.json"]))
    decoded = 0
    for key, field in (("tokens_train", "input_ids"), ("tokens_validation", "input_ids"), ("capture_prompts", "tokens")):
        for line in paths[key].read_text().splitlines():
            row = json.loads(line)
            check_public_files({key: tokenizer.decode(row[field], skip_special_tokens=False).encode()}, reserved)
            decoded += 1
    comparison = root / "organism/evidence/first-lineage-temperature-evaluation-20261016-01/audit.json"
    comparison_receipt = artifact(root, comparison)
    approval = root / "organism/evidence/cpu-export-execution-20261015-03/approval.json"
    status = load(root / "organism/evidence/first-lineage-milestone-20261016-01/status.json")
    require(status["historical_project_reservations_usd"] == 215
            and status["milestone_native_reservation_usd"] == 60
            and status["planned_project_reservations_usd"] == before
            and status["candidate_optimizer_started"] is False
            and status["temperature_comparison_successes"] == {"trained": 4, "stock": 19, "fixed_policy": 20},
            "Recorded model-failure or reservation baseline changed")
    output.mkdir(parents=True)
    public = output / "public"
    public.mkdir()
    compute = {
        "schema": "organism-standing-gpu-allocation-v2", "authority": "direct human reply in this chat",
        "currency": "USD", "scope": "one separately admitted first-lineage GPU run",
        "standing_authorization": artifact(root, approval),
        "maximum_total_spend": 35, "project_reservations_before_usd": before,
        "project_reservations_after_usd": before + 35, "maximum_leases": 1, "maximum_hours_per_lease": 2,
        "leases_must_be_sequential": True, "automatic_expiry_and_cleanup_required": True,
        "training_requires_frozen_experiment_contract": True, "new_gpu_allocations_at_freeze": 0,
        "proposed_cpu_export_reservation_usd": 20, "proposed_combined_reservations_usd": before + 55,
        "invoice_reconciliation_claimed": False,
    }
    if readmission:
        compute["previous_attempt_evidence"] = {
            name: artifact(root, path) for name, path in admission_context["previous_attempt_evidence"].items()}
        compute["previous_failed_allocation_released_from_reservations"] = False
        compute["reason"] = "Separately admitted attempt after diagnosed pre-optimizer managed-wrapper failure."
    compute_path = public / "compute-allocation.json"
    write(compute_path, compute)
    data = {
        "schema": "organism-automated-data-verification-v1", "experiment_id": recipe["experiment_id"],
        "recipe_sha256": recipe_sha, "input_artifacts": recipe["inputs"], "counts": recipe["counts"],
        "checks": {key: True for key in DATA_VERIFICATION_CHECKS}, "passed": True,
        "human_review_required": False, "human_review_verified": False,
        "evidence": {key: artifact(root, path) for key, path in (
            ("preflight", preflight_path), ("runtime", paths["runtime_audit"]), ("targets", paths["row_audit"]),
            ("native_parser", paths["native_parser_report"]), ("token_parser_binding", paths["token_parser_binding"]))},
        "native_parser_reconstruction": parser_check, "public_input_check": public_check,
        "decoded_rows_checked": decoded, "private_qualification_assets_read": False,
    }
    data_path = public / "data-verification.json"
    write(data_path, data)
    contract = deepcopy(old_contract)
    contract.pop("frozen_at")
    contract.pop("previous_draft", None)
    contract.update(experiment_id=recipe["experiment_id"], optimization_started=False, qualification_passed=False,
                    final_release_bound=False, qualification_deferred=True,
                    previous_frozen_experiment=recipe["inputs"]["parent_contract"],
                    phase0_scope="First-lineage development subset and corrected native targets; original qualification deferred.")
    contract["phase0_evidence"] = artifact(root, data_path)
    contract["training_preparation"].update(
        recipe=artifact(root, recipe_path), data_verification=artifact(root, data_path),
        maximum_optimizer_steps=90, maximum_total_gpu_spend_usd=35,
        maximum_leases=1, leases_consumed_at_freeze=0, leases_remaining_at_freeze=1,
        maximum_training_seconds=2700, maximum_workload_seconds=3600, maximum_lease_seconds=7200)
    contract["development_comparison"] = {
        "pre_refinement_evidence": comparison_receipt, "new_recipe_optimizer_steps_at_freeze": 0,
        "initialization": "fresh adapter on the same pinned base",
        "arms": ["stock base", "completed v6 first-lineage adapter", "fixed policy"],
        "candidate": "final second-epoch checkpoint; retain both epochs and all failures",
        "comparison": "Twenty unchanged development temperature cases; identical current tools, prompts and budgets.",
        "required_trained_successes": 20, "frozen_cases": artifact(root, frozen),
        "qualification_feedback_allowed_in_training": False, "thresholds_changed": False,
    }
    seal_path = public / "qualification-seal.json"
    seal = derived_seal(recipe, recipe_sha, old_seal, at=datetime.now(timezone.utc).isoformat())
    write(seal_path, seal)
    contract["training_admission"].update(
        recipe_sha256=recipe_sha, candidate_manifest_sha256=recipe["inputs"]["source_manifest"]["sha256"],
        token_report_sha256=recipe["inputs"]["token_report"]["sha256"],
        data_verification_method="automated", data_verification_verified=True,
        data_verification_evidence=artifact(root, data_path), qualification_splits_sealed=True,
        qualification_seal=artifact(root, seal_path), compute_authorization_verified=True,
        compute_authorization_evidence=artifact(root, compute_path))
    contract["frozen_at"] = datetime.now(timezone.utc).isoformat()
    contract_path = public / "experiment-contract.frozen.json"
    write(contract_path, contract)
    require(not admission_errors(contract, recipe, recipe_sha), "First-lineage admission is incomplete")
    verify_admission_evidence(root, contract, recipe, recipe_sha)
    kit = public / "training-kit"
    build(root, kit, mode="train", contract_path=contract_path, recipe_path=recipe_path,
          instance_type="g6.4xlarge", region="us-east-1")
    with tarfile.open(kit / "kit.tar.gz") as archive:
        members = archive.getmembers()
        require(all(member.isfile() for member in members) and len(members) <= 128
                and sum(member.size for member in members) <= 256 * 1024**2, "Unsafe training archive")
        packed = {member.name: archive.extractfile(member).read() for member in members}
        require(len(packed) == len(members), "Duplicate training archive path")
    manifest = json.loads(packed["worker-input.json"])
    require(set(packed) == {entry["path"] for entry in manifest["files"]} | {"worker-input.json"},
            "Packaged input inventory differs from its manifest")
    for entry in manifest["files"]:
        raw = packed[entry["path"]]
        require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"],
                "Packaged bytes differ from the reviewed artifact")
    package_check = check_public_files(packed, reserved)
    result = {
        "passed": True, "at": datetime.now(timezone.utc).isoformat(), "recipe": artifact(root, recipe_path),
        "contract": artifact(root, contract_path), "qualification_seal": artifact(root, seal_path),
        "training_archive_check": package_check, "runtime_subset_check": runtime,
        "native_parser_reconstruction": parser_check, "compiler": artifact(root, Path(__file__)),
        "public_parent_commitments_preserved": True, "private_qualification_assets_read": False,
        "optimizer_steps_for_this_recipe": 0, "cloud_resources_created": False,
        "training_completed": False, "qualification_passed": False,
    }
    write(output / "COMPLETE.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readmission", action="store_true")
    for name in ("root", "recipe", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = compile_first_lineage(args.root, args.recipe, args.output, readmission=args.readmission)
    print(json.dumps({key: result[key] for key in ("passed", "optimizer_steps_for_this_recipe",
                                                 "cloud_resources_created", "qualification_passed")}))
