"""Local evaluator-only seal compiler. Never shipped to a training worker.

Reconstruct retained inputs and evidence, commit them without rewriting history,
then validate a public receipt and the actual training archive. No model,
optimizer, AWS API or device activation is invoked.
"""

import argparse
import ast
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile

from .baseline_review import audit as audit_baseline
from .capsule import verify_capsule_directory
from .check_qualification_inputs import audit_capture, read_inputs, expectation_matches
from .audit_equipment_diagnostics import inspect_job
from .diagnostic_oracle import encoded, reconstruct
from .qualification_inputs import entry, materialize
from .qualification_protocol import check as check_protocol
from .recovery_schedule import generate as recovery_schedules
from .training_inputs import (
    DATA_VERIFICATION_CHECKS, SEAL_CHECKS, SEAL_COUNTS, admission_errors,
    load, require, sha256, validate_recipe, verified_path, verify_admission_evidence,
)
from .training_worker.build import build


QUAL = "organism/evaluation/qualification-inputs-20261014-01"
PROTO = "organism/evaluation/qualification-protocol-20261014-01"
Q6 = "organism/evaluation/q6-schedules-20261014-01"
EVIDENCE = "organism/evidence/qualification-protocol-20261014-01"
BASELINE = "organism/evidence/baseline-freeze-review-20261014-01"
RECIPE = "organism/training-recipe.current-v4.admission-v2.draft.json"
DRAFT = "organism/experiment-contract.current-v4.admission-v2.draft.json"


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def artifact(root, path):
    path = Path(path).absolute()
    require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
            and path.is_file(), "Seal artifact must be a regular workspace file")
    return {"path": str(path.relative_to(root)), **entry(path.read_bytes())}


def inventory(root, paths):
    values = {}
    for path in sorted(set(paths)):
        item = artifact(root, path)
        values[item["path"]] = {key: item[key] for key in ("bytes", "sha256")}
    require(bool(values), "Cannot commit an empty inventory")
    return values


def tree_files(path):
    return [item for item in path.rglob("*") if item.is_file()]


def audit_source_records(root):
    report = load(root / EVIDENCE / "source-reader-audit.json")
    require(report["passed"] is True and report["model_calls"] == 0
            and report["live_source_episodes_executed"] == 0 and report["signed_notes_created"] == 0,
            "Source input check was replaced by an execution claim")
    require(report["protocol_manifest"] == entry((root / PROTO / "manifest.json").read_bytes())
            and len(report["cases"]) == 32, "Source input audit inventory changed")
    ids = set()
    for row in report["cases"]:
        identity = row["case_id"]
        require(identity not in ids, "Duplicate source input record")
        ids.add(identity)
        source_path = root / PROTO / "source-inputs" / (identity + ".json")
        capture_path = root / EVIDENCE / "source-reader-checks" / (identity + ".json")
        require(entry(source_path.read_bytes()) == row["input"]
                and entry(capture_path.read_bytes()) == row["record"], "Source input or reader record changed")
        source, saved = load(source_path), load(capture_path)
        require(saved["observation"] == materialize(source, saved["epoch"])
                and saved["model_calls"] == 0 and saved["harness_selected_reader_and_answer"] is True,
                "Source reader capture has changed provenance or observations")
        raw = encoded(saved["observation"])
        result = reconstruct(raw, source["oracle_plan"], now=saved["epoch"])
        require(expectation_matches(source, result) and result == saved["reconstruction"],
                "Source expectation differs from independent interpretation")
        case = {"case_id": identity, "document": saved["observation"],
                **{key: result[key] for key in
                   ("expected_reader", "expected_status", "expected_value", "expected_unit")}}
        checked = inspect_job(case, raw, saved["job"])
        require(checked == saved["trace_check"] and checked["semantic_passed"] is True,
                "Saved source reader evidence fails reconstruction")
        reads = [event["result"] for event in saved["job"]["record"]["trace"]["tool_events"]
                 if event["tool"] == "read_equipment"]
        require(len(reads) == 1 and reads[0]["read_unix_seconds"] == saved["epoch"],
                "Source reader clock differs from its materialized observation")
    return {"source_inputs": len(ids), "tool_calls": report["tool_calls"],
            "live_source_episodes": 0, "passed": True}


def check_exclusion_inventory(root, recipe):
    report = load(root / EVIDENCE / "exclusion.json")
    require(report["passed"] is True and report["reserved_inputs"] == 152
            and report["reserved_markers"] == 360 and report["decoded_tokens"] == 42090477
            and report["marker_hits"] == report["normalized_document_matches"]
            == report["withheld_field_kind_layout_matches"] == [],
            "Bounded exclusion evidence is incomplete")
    require(report["recipe"] == entry((root / RECIPE).read_bytes())
            and report["protocol_manifest"] == entry((root / PROTO / "manifest.json").read_bytes()),
            "Exclusion report belongs to different inputs")
    for name, value in report["inventory"].items():
        verified_path(root, {"path": name, "bytes": value["bytes"], "sha256": value["sha256"]})
    for item in recipe["inputs"].values():
        require(item["path"] in report["inventory"]
                and {key: report["inventory"][item["path"]][key] for key in ("bytes", "sha256")}
                == {key: item[key] for key in ("bytes", "sha256")},
                "Recipe input lacks the recorded exclusion check")
    return {"files": len(report["inventory"]), "decoded_tokens": report["decoded_tokens"], "passed": True}


def markers(root, cases):
    values = {case[key] for case in cases for key in ("case_id", "case_seed")}
    values.add(load(root / QUAL / "seed.json")["seed"])
    for path in (root / PROTO / "source-inputs").glob("*.json"):
        case = load(path)
        values.update((case["case_id"], case["case_seed"]))
    values.update(case["case_id"] for case in load(root / Q6 / "schedules.json"))
    for trial in load(root / PROTO / "protocol.json")["lifecycle_trials"]:
        values.update((trial["trial_id"], trial["seed"]))
    require(len(values) == 360, "Reserved marker inventory changed")
    return values


def check_worker_files(files, reserved):
    pattern = re.compile(r"(?:qual-(?:source-|life-)?|q6-)[0-9a-f]{32}|[0-9a-f]{64}")
    require(1 <= len(files) <= 128, "Unexpected training file count")
    for name, raw in files.items():
        require(not any(part in name for part in
                        ("organism/evaluation/", "evaluation_seal.py", "qualification_protocol.py",
                         "qualification_inputs.py", "check_qualification_inputs.py", "recovery_schedule.py")),
                "Evaluator-only source entered the training package")
        found = {match.group() for match in pattern.finditer(raw.decode("utf-8", errors="replace"))} & reserved
        require(not found, "Reserved evaluation identifier or seed entered the training package")
    return {"passed": True, "files": len(files), "reserved_markers": len(reserved)}


def verify_archive(kit, reserved):
    with tarfile.open(kit / "kit.tar.gz") as archive:
        members = archive.getmembers()
        require(all(member.isfile() for member in members)
                and len(members) <= 128 and sum(member.size for member in members) <= 256 * 1024**2,
                "Unexpected training archive contents")
        files = {member.name: archive.extractfile(member).read() for member in members}
        require(len(files) == len(members), "Duplicate training archive member")
    manifest = json.loads(files["worker-input.json"])
    require(manifest["mode"] == "train"
            and set(files) == {item["path"] for item in manifest["files"]} | {"worker-input.json"},
            "Training archive inventory differs")
    for item in manifest["files"]:
        require(entry(files[item["path"]]) == {key: item[key] for key in ("bytes", "sha256")},
                "Training archive file differs from its bound input")
    return check_worker_files(files, reserved)


def compile_seal(root, output, decision_path, public_output):
    root, output, public_output = map(lambda path: Path(path).resolve(), (root, output, public_output))
    require(output.is_relative_to(root) and not output.exists(), "Seal output must be a fresh workspace directory")
    require(public_output.is_relative_to(root) and not public_output.exists()
            and not public_output.is_relative_to(output) and not output.is_relative_to(public_output)
            and not public_output.is_relative_to(root / "organism/evaluation"),
            "Public receipt output must be fresh and separate from private evaluator assets")
    recipe, draft = load(root / RECIPE), load(root / DRAFT)
    validate_recipe(recipe)
    recipe_sha = sha256(root / RECIPE)
    for item in recipe["inputs"].values():
        verified_path(root, item)
    require(draft["optimization_started"] is False and draft["status"] == "draft_not_frozen",
            "An already optimized or frozen experiment cannot be resealed")
    data_path = verified_path(root, draft["training_admission"]["data_verification_evidence"])
    data = load(data_path)
    require(data["input_artifacts"] == recipe["inputs"] and data["recipe_sha256"] == recipe_sha
            and data["counts"] == recipe["counts"] and data["passed"] is True
            and data["checks"] == {name: True for name in DATA_VERIFICATION_CHECKS},
            "Automated data admission does not bind this candidate")
    for item in data["evidence"].values():
        verified_path(root, item)
    preflight = load(verified_path(root, data["evidence"]["preflight"]))
    require(preflight["passed"] is True and preflight["optimizer_steps"] == 0
            and preflight["recipe_sha256"] == recipe_sha,
            "Preparation evidence differs from this unoptimized recipe")
    _, cases = read_inputs(root / QUAL)
    target_capture = root / "organism/evidence/qualification-input-check-20261014-03"
    target_check = audit_capture(root / QUAL, target_capture)
    protocol_check = check_protocol(root / QUAL, root / PROTO)
    source_check = audit_source_records(root)
    exclusion_check = check_exclusion_inventory(root, recipe)
    q6 = root / Q6
    require(recovery_schedules(load(q6 / "design.json"), load(q6 / "contract.json"),
                              load(q6 / "seed.json")["seed"]) == load(q6 / "schedules.json"),
            "Recovery schedules differ from their precommitted seed")
    q6_manifest = load(q6 / "manifest.json")
    require(q6_manifest["schedules_sha256"] == sha256(q6 / "schedules.json"),
            "Recovery schedule bytes changed")
    for name, expected in q6_manifest["inputs"].items():
        require(entry((q6 / name).read_bytes()) == expected, "Recovery authoring input changed")
    baseline_check = audit_baseline(root / BASELINE, root / "organism/data/equipment-runtime-development-v1")
    require(baseline_check["diagnostic_successes"] == 280 and baseline_check["ambiguity_declines"] == 8
            and baseline_check["expected_behavior_passed"] is True, "Baseline competence evidence differs")
    baseline_source = (root / "organism/equipment_baseline.py").read_text()
    require(not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(ast.parse(baseline_source)))
            and not any(value in baseline_source for value in ("qual-", "environment-", "expected_reader", "expected_status")),
            "Fixed baseline acquired external code, private lookups or family branches")
    capsule = root / "organism/artifacts/complete-stock-capsule-20261014-03"
    trust = load(capsule / "release-trust.json")
    public = {name: base64.b64decode(value, validate=True) for name, value in trust["public_keys"].items()}
    checked_capsule = verify_capsule_directory(
        capsule / "capsule", (capsule / "capsule/.manifest.json").read_bytes(), public, require_trained=False)
    require(checked_capsule["release_id"] == trust["release_id"], "Stock package release identity differs")
    native = load(root / "organism/evidence/native-startup-execution-20261014-03/native-inference-failure-readback.json")
    require(set(native["architectures"]) == {"x86_64", "arm64"}
            and all(value["model_inference_observed"] is True and value["qualification_passed"] is False
                    for value in native["architectures"].values()), "Native failed baseline evidence differs")
    gpu = load(root / "organism/evidence/startup-gpu-execution-20261014-01/failure-readback.json")
    require(gpu["native_source_files_verified"] is True and gpu["native_base_loading_observed"] is True
            and gpu["base_parameters_checked_before_first_forward"] == recipe["base"]["expected_parameters"]
            and gpu["optimizer_steps"] == 0 and gpu["forward_completed"] is False,
            "Native source-loading evidence differs")
    decision = artifact(root, decision_path)
    reserved = markers(root, cases)
    output.mkdir()
    write(output / "request.json", {"at": datetime.now(timezone.utc).isoformat(),
                                   "recipe_sha256": recipe_sha, "decision": decision,
                                   "scope": "input freeze, not model or native qualification"})
    custody = {
        "author": "same project agent", "author_accessed_expected_answers": True,
        "independent_human_blinding_claimed": False, "inaccessible_custody_claimed": False,
        "boundary": "training_and_deployment_input_allowlists",
        "qualification_model_runs_before_freeze": 0, "source_model_runs_before_freeze": 0,
        "public_descriptions_are_not_secret": True,
        "later_access_rule": "Record every exposure; forbid qualification feedback, cases and source notes in training.",
        "final_deployment_package_exclusion_required": True,
    }
    write(output / "custody.json", custody)
    groups = {
        "qualification_inputs": tree_files(root / QUAL),
        "inheritance_source_inputs": tree_files(root / PROTO / "source-inputs")
            + tree_files(root / EVIDENCE / "source-reader-checks") + [root / EVIDENCE / "source-reader-audit.json"],
        "lifecycle_protocol": [root / PROTO / "protocol.json", root / PROTO / "manifest.json"]
            + tree_files(root / PROTO / "sources"),
        "q6_schedules": tree_files(q6),
        "fixed_baseline": tree_files(root / BASELINE),
        "independent_oracles": tree_files(target_capture) + [root / "organism" / name for name in
                                ("diagnostic_oracle.py", "audit_equipment_diagnostics.py",
                                 "audit_inherited_diagnostics.py", "audit_application.py",
                                 "check_qualification_inputs.py", "experience_issue.py")],
        "exclusion_audit": [root / EVIDENCE / name for name in
                            ("exclusion.json", "audit_exclusion.py", "package-check.json", "check_package.py")],
        "custody_record": [output / "custody.json"],
    }
    commitments = {}
    for name, paths in groups.items():
        path = output / (name + ".index.json")
        write(path, inventory(root, paths))
        commitments[name] = entry(path.read_bytes())
    decision_record = {
        "source_and_data_preparation": artifact(root, data_path),
        "native_failed_baseline": artifact(root, root / "organism/evidence/native-startup-execution-20261014-03/native-inference-failure-readback.json"),
        "native_base_loading": artifact(root, root / "organism/evidence/startup-gpu-execution-20261014-01/failure-readback.json"),
        "stock_capsule": artifact(root, capsule / "receipt.json"),
        "stock_manifest": artifact(root, capsule / "capsule/.manifest.json"),
        "source_pins": artifact(root, capsule / "input-pins.json"),
        "retained_licenses_and_notices": artifact(root, capsule / "notice-index.json"),
        "runtime_dependency_lock": artifact(root, root / "organism/requirements-runtime.lock"),
        "native_startup_kit": artifact(root, root / "organism/launch/native-startup-20261014-08/build-receipt.json"),
        "training_dependency_lock": recipe["inputs"]["dependency_lock"],
        "cost_envelope": artifact(root, root / "organism/evidence/startup-gpu-execution-20261014-01/cost-envelope.json"),
        "gpu_lease_ledger": artifact(root, root / "organism/evidence/startup-gpu-execution-20261014-01/ledger.json"),
        "compiler": artifact(root, Path(__file__).resolve()),
        "decision": decision,
    }
    write(output / "phase0-evidence.json", decision_record)
    contract = {key: deepcopy(draft[key]) for key in
                ("schema_version", "experiment_id", "plan", "model", "habitat", "resource_targets",
                 "qualification_requirements", "data_review_policy", "plan_amendment", "training_admission")}
    contract.update(status="frozen", phase0_completed=True, optimization_started=False,
                    qualification_passed=False, final_release_bound=False,
                    phase0_scope="Experiment inputs and constraints frozen; model success and qualification are unproved.",
                    phase0_evidence=artifact(root, output / "phase0-evidence.json"),
                    previous_draft=artifact(root, root / DRAFT),
                    evaluation_protocol=artifact(root, root / PROTO / "protocol.json"),
                    final_qualification_admission=load(root / PROTO / "protocol.json")["release_admission_before_qualification"])
    contract["model"]["candidate_status"] = "selected for bounded mandatory fine-tuning pilot; not qualified"
    contract["habitat"]["native_instances_started"] = False
    contract["habitat"]["qualification_maximum_concurrent_native_hosts"] = 6
    contract["habitat"]["qualification_instance_counts_are_authorization"] = False
    contract["runtime"] = {
        "strands_version": "1.57.1", "deployment_quantization_family": "Q4_K_M",
        "context_tokens": 4096, "concurrent_generations": 1,
        "both_platform_package_pins": decision_record["source_pins"],
        "runtime_dependency_lock": decision_record["runtime_dependency_lock"],
        "final_model_and_runtime_release_requires_new_complete_manifest": True,
        "useful_native_readiness_passed": False, "native_four_generation_qualification_passed": False,
    }
    contract["training_preparation"] = {
        "recipe": artifact(root, root / RECIPE), "data_verification": artifact(root, data_path),
        "native_base_loading_observed": True, "native_forward_backward_passed": False,
        "mandatory_gpu_precheck_before_optimizer": True, "maximum_optimizer_steps": 210,
        "combined_precheck_and_training_workload_seconds": 14400,
        "maximum_total_gpu_spend_usd": 35, "maximum_leases": 2,
        "leases_consumed_at_freeze": 1, "leases_remaining_at_freeze": 1,
        "pilot_candidate_selection": "Use the final completed second-epoch adapter as the first development candidate; preserve both epoch checkpoints. No qualification result selects or trains a checkpoint.",
        "further_training_requires_new_explicit_development_recipe_and_admission": True,
    }
    sealed_at = datetime.now(timezone.utc).isoformat()
    seal = {
        "schema": "organism-qualification-seal-v1", "status": "sealed_before_optimization",
        "experiment_id": recipe["experiment_id"], "recipe_sha256": recipe_sha,
        "candidate_manifest_sha256": recipe["inputs"]["source_manifest"]["sha256"],
        "token_report_sha256": recipe["inputs"]["token_report"]["sha256"],
        "sealed_at": sealed_at, "checks": {name: True for name in SEAL_CHECKS},
        "counts": deepcopy(SEAL_COUNTS), "commitments": commitments,
        "qualification_requirements": contract["qualification_requirements"],
        "resource_targets": contract["resource_targets"],
        **{key: custody[key] for key in ("boundary", "author_accessed_expected_answers",
                                        "independent_human_blinding_claimed", "inaccessible_custody_claimed")},
        "optimizer_steps_at_seal": 0, "qualification_executed": False, "qualification_passed": False,
        "receipt_scope": "Automated input commitment; no claim of native qualification or inaccessible custody.",
    }
    seal_path = public_output / "qualification-seal.json"
    # Check every prospective worker payload before publishing the valid seal.
    worker_files = {item["path"]: verified_path(root, item).read_bytes() for item in recipe["inputs"].values()}
    for name in ("__init__.py", "training_worker/__init__.py", "training_worker/runner.py",
                 "training_worker/host.py", "training_worker/check_inputs.py", "training_worker/regions.py"):
        worker_files["organism/" + name] = (root / "organism" / name).read_bytes()
    worker_files[RECIPE] = (root / RECIPE).read_bytes()
    worker_files[str(seal_path.relative_to(root))] = (json.dumps(seal, indent=2) + "\n").encode()
    seal_identity = {"path": str(seal_path.relative_to(root)),
                     **entry(worker_files[str(seal_path.relative_to(root))])}
    contract["frozen_at"] = datetime.now(timezone.utc).isoformat()
    contract["training_admission"].update(qualification_splits_sealed=True, qualification_seal=seal_identity)
    worker_files["organism/worker-contract.json"] = (json.dumps(contract, indent=2) + "\n").encode()
    for field in ("data_verification_evidence", "compute_authorization_evidence"):
        item = contract["training_admission"][field]
        worker_files[item["path"]] = verified_path(root, item).read_bytes()
    worker_scope = check_worker_files(worker_files, reserved)
    public_output.mkdir(parents=True)
    write(seal_path, seal)
    contract_path = public_output / "experiment-contract.frozen.json"
    write(contract_path, contract)
    require(not admission_errors(contract, recipe, recipe_sha), "Frozen contract did not satisfy admission")
    verify_admission_evidence(root, contract, recipe, recipe_sha)
    kit = public_output / "training-kit"
    build(root, kit, mode="train", contract_path=contract_path, recipe_path=root / RECIPE,
          instance_type="g6e.4xlarge", region="us-east-1")
    archive_check = verify_archive(kit, reserved)
    final = {
        "at": datetime.now(timezone.utc).isoformat(), "schema": "organism-experiment-freeze-audit-v1",
        "passed": True, "recipe": artifact(root, root / RECIPE), "contract": artifact(root, contract_path),
        "qualification_seal": artifact(root, seal_path), "target_input_check": target_check,
        "protocol_check": protocol_check, "source_input_check": source_check,
        "q6_schedules_reconstructed": 15, "baseline_check": baseline_check,
        "exclusion_check": exclusion_check, "prospective_worker_scope": worker_scope,
        "actual_training_archive_scope": archive_check,
        "training_kit": artifact(root, kit / "build-receipt.json"),
        "admission_errors": [], "optimizer_steps": 0, "native_gpu_precheck_passed": False,
        "training_completed": False, "qualification_passed": False, "cloud_resources_created": False,
    }
    write(output / "COMPLETE.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    args = parser.parse_args()
    result = compile_seal(Path(__file__).resolve().parents[1], args.output, args.decision, args.public_output)
    print(json.dumps({key: result[key] for key in
                      ("passed", "contract", "qualification_seal", "admission_errors", "optimizer_steps")}, indent=2))


if __name__ == "__main__":
    main()
