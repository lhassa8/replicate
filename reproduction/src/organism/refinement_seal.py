"""Freeze one data-refinement experiment while retaining the existing evaluator.

This local-only compiler verifies every inherited commitment, scans complete new
worker inputs and decoded targets, and packages no private evaluator artifacts.
It does not invoke a model, optimizer, cloud API or device activation.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tarfile

from .training_inputs import (DATA_VERIFICATION_CHECKS, SEAL_CHECKS, SEAL_COUNTS, SEAL_COMMITMENTS,
                              load, require, sha256, verified_path, validate_recipe, verify_qualification_seal,
                              verify_admission_evidence, admission_errors)
from .training_worker.build import build


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def artifact(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Freeze input is outside the workspace")
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def content_identity(root, path):
    return {k: v for k, v in artifact(root, path).items() if k != "path"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def documents(value):
    if isinstance(value, dict):
        if value.get("schema") == "organism-equipment-interface-v1":
            yield value
        else:
            for item in value.values():
                yield from documents(item)
    elif isinstance(value, list):
        for item in value:
            yield from documents(item)


def normalized(document):
    result = deepcopy(document)
    for name in ("device_id", "family", "sequence"):
        result.pop(name, None)
    for source in result["sources"]:
        source["fields"].sort(key=canonical)
    result["sources"].sort(key=lambda value: value["source_id"])
    return canonical(result)


def shape(document):
    return canonical(sorted(sorted(f["kind"] for f in s["fields"]) for s in document["sources"]))


def inherited_commitments(root, old_seal, destination):
    parent = root / "organism/evaluation/sealed-20261014-02"
    require(set(old_seal["commitments"]) == set(SEAL_COMMITMENTS), "Original evaluator inventory differs")
    result, checked = {}, {}
    for group, expected in old_seal["commitments"].items():
        source = parent / (group + ".index.json")
        require(content_identity(root, source) == expected, "Original evaluator commitment changed")
        entries = load(source)
        for name, value in entries.items():
            verified_path(root, {"path": name, **value})
        checked[group] = len(entries)
        copied = destination / source.name
        copied.write_bytes(source.read_bytes())
        result[group] = content_identity(root, copied)
    return result, checked


def private_inventory(root):
    qual = root / "organism/evaluation/qualification-inputs-20261014-01"
    proto = root / "organism/evaluation/qualification-protocol-20261014-01"
    cases = []
    for item in load(qual / "manifest.json")["cases"]:
        path = qual / "cases" / (item["case_id"] + ".json")
        require(content_identity(root, path) == item["receipt"], "Sealed target input changed")
        cases.append(load(path))
    sources = [load(p) for p in sorted((proto / "source-inputs").glob("*.json"))]
    cases += sources
    reserved = {case[k] for case in cases for k in ("case_id", "case_seed")}
    reserved.add(load(qual / "seed.json")["seed"])
    reserved.update(case["case_id"] for case in load(
        root / "organism/evaluation/q6-schedules-20261014-01/schedules.json"))
    for trial in load(proto / "protocol.json")["lifecycle_trials"]:
        reserved.update((trial["trial_id"], trial["seed"]))
    require(len(cases) == 152 and len(reserved) == 360, "Sealed evaluator case allocation changed")
    return cases, reserved


def check_public_files(files, reserved):
    pattern = re.compile(r"(?:qual-(?:source-|life-)?|q6-)[0-9a-f]{32}|[0-9a-f]{64}")
    require(0 < len(files) <= 128, "Unexpected worker inventory size")
    for name, raw in files.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and not any(v in name for v in (
                    "organism/evaluation/", "refinement_seal.py", "evaluation_seal.py", "qualification_protocol.py",
                    "qualification_inputs.py", "check_qualification_inputs.py", "recovery_schedule.py")),
                "Private evaluator source or path entered the worker")
        found = {m.group() for m in pattern.finditer(raw.decode("utf-8", errors="replace"))} & reserved
        require(not found, "Sealed evaluator identifier entered a worker input")
    return {"passed": True, "files": len(files), "reserved_markers": len(reserved)}


def exclusion(root, recipe, reserved, cases):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(verified_path(root, recipe["inputs"]["tokenizer/tokenizer.json"])))
    files = {v["path"]: verified_path(root, v).read_bytes() for v in recipe["inputs"].values()}
    check_public_files(files, reserved)
    observed, shapes = set(), defaultdict(set)
    for key in ("source_train", "source_validation", "refinement_references"):
        for line in files[recipe["inputs"][key]["path"]].splitlines():
            for document in documents(json.loads(line)):
                observed.add(normalized(document))
                shapes[document["task"]["quantity"]].add(shape(document))
    require(not any(normalized(case["document"]) in observed for case in cases),
            "A sealed observation entered development data")
    require(not any(case["slot"]["suite"] not in ("protected", "protected_source")
                    and shape(case["document"]) in shapes[case["slot"]["domain"]] for case in cases),
            "A withheld equipment layout entered development data")
    pattern = re.compile(r"(?:qual-(?:source-|life-)?|q6-)[0-9a-f]{32}|[0-9a-f]{64}")
    decoded, count = 0, 0
    for key, field in (("tokens_train", "input_ids"), ("tokens_validation", "input_ids"),
                       ("capture_prompts", "tokens")):
        for line in files[recipe["inputs"][key]["path"]].splitlines():
            row = json.loads(line)
            text = tokenizer.decode(row[field], skip_special_tokens=False)
            require(not {m.group() for m in pattern.finditer(text)} & reserved,
                    "A decoded training or prompt token contains a sealed identifier")
            decoded += len(row[field])
            count += 1
    return {
        "passed": True, "reserved_inputs": len(cases), "reserved_markers": len(reserved),
        "input_files": len(files), "decoded_rows": count, "decoded_tokens": decoded,
        "marker_hits": [], "normalized_document_matches": [], "withheld_field_kind_layout_matches": [],
        "scope": "All exact refinement recipe inputs, complete targets and captured prefixes; original historical exclusion retained.",
        "limitations": ["Exact-content and field-layout exclusion does not prove semantic independence.",
                        "The project agent authored and accessed evaluator answers; no inaccessible custody is claimed."],
    }


def verify_archive(directory, reserved):
    with tarfile.open(directory / "kit.tar.gz") as archive:
        members = archive.getmembers()
        require(all(m.isfile() for m in members) and len(members) <= 128
                and sum(m.size for m in members) <= 256 * 1024**2, "Unsafe or oversized worker package")
        files = {m.name: archive.extractfile(m).read() for m in members}
        require(len(files) == len(members), "Duplicate archive path")
    manifest = json.loads(files["worker-input.json"])
    require(manifest["mode"] == "train"
            and set(files) == {r["path"] for r in manifest["files"]} | {"worker-input.json"},
            "Worker inventory does not match its manifest")
    import hashlib
    for entry in manifest["files"]:
        raw = files[entry["path"]]
        require(entry["bytes"] == len(raw) and entry["sha256"] == hashlib.sha256(raw).hexdigest(),
                "A packaged input differs from its frozen identity")
    return check_public_files(files, reserved)


def compile_refinement(root, recipe_path, output, public_output):
    root, recipe_path, output, public_output = map(Path, (root, recipe_path, output, public_output))
    root, recipe_path, output, public_output = (p.resolve() for p in (root, recipe_path, output, public_output))
    require(all(p.is_relative_to(root) for p in (recipe_path, output, public_output))
            and not output.exists() and not public_output.exists()
            and not public_output.is_relative_to(output) and not output.is_relative_to(public_output)
            and not public_output.is_relative_to(root / "organism/evaluation"),
            "Freeze outputs must be new and separate workspace directories")
    recipe, recipe_sha = load(recipe_path), sha256(recipe_path)
    validate_recipe(recipe)
    require(recipe["corpus"] == "refinement_runtime" and recipe["optimization_started"] is False,
            "Only a new unoptimized refinement recipe can be frozen")
    for item in recipe["inputs"].values():
        verified_path(root, item)
    parent_public = root / "organism/evidence/experiment-freeze-20261014-01/public"
    old_contract, old_seal = load(parent_public / "experiment-contract.frozen.json"), load(parent_public / "qualification-seal.json")
    old_recipe_path = root / "organism/training-recipe.current-v4.admission-v2.draft.json"
    old_recipe = load(old_recipe_path)
    verify_qualification_seal(old_seal, old_contract, old_recipe, sha256(old_recipe_path))
    require(recipe["base"] == old_recipe["base"] and recipe["adapter"] == old_recipe["adapter"]
            and recipe["versions"] == old_recipe["versions"] and recipe["counts"] == {"train": 2100, "validation": 576},
            "Refinement changed more than the admitted data comparison")
    changed_training = {k for k in recipe["training"] if recipe["training"][k] != old_recipe["training"][k]}
    require(changed_training == {"updates_per_epoch", "maximum_optimizer_steps", "warmup_steps"},
            "Refinement changed the original training method")
    evidence_root = root / "organism/evidence/refinement-training-integration-20261016-01"
    preflight_path = evidence_root / "preflight/report.json"
    preflight = load(preflight_path)
    require(preflight["passed"] is True and preflight["optimizer_steps"] == 0
            and preflight["recipe_sha256"] == recipe_sha, "Trainer preflight does not bind this recipe")
    runtime_path = verified_path(root, recipe["inputs"]["runtime_audit"])
    row_path = verified_path(root, recipe["inputs"]["row_audit"])
    require(load(runtime_path)["passed"] is True
            and load(runtime_path)["source_manifest_sha256"] == recipe["inputs"]["source_manifest"]["sha256"]
            and load(row_path)["passed"] is True
            and load(row_path)["token_report_sha256"] == recipe["inputs"]["token_report"]["sha256"],
            "Complete runtime or target reconstruction is missing")
    comparison = root / "organism/evidence/attention-export-development-tasks-20261016-01/audit.json"
    checked = load(comparison)
    require(checked["evidence_reconstructed"] is True
            and checked["arms"]["trained-merged"]["model_successes"] == 3
            and checked["arms"]["stock-control"]["model_successes"] == 4, "Pre-refinement comparison changed")
    output.mkdir(parents=True)
    public_output.mkdir(parents=True)
    commitments, inherited = inherited_commitments(root, old_seal, output)
    cases, reserved = private_inventory(root)
    exclusion_result = exclusion(root, recipe, reserved, cases)
    exclusion_path = output / "refinement-exclusion.json"
    write(exclusion_path, {"recipe": artifact(root, recipe_path), **exclusion_result})
    previous_index = load(output / "exclusion_audit.index.json")
    previous_index[str(exclusion_path.relative_to(root))] = content_identity(root, exclusion_path)
    new_index = output / "refinement-exclusion.index.json"
    write(new_index, previous_index)
    commitments["exclusion_audit"] = content_identity(root, new_index)
    approval = root / "organism/evidence/cpu-export-execution-20261015-03/approval.json"
    compute = {
        "schema": "organism-standing-gpu-allocation-v1", "authority": "direct human reply in this chat",
        "currency": "USD", "scope": "one separately admitted organism refinement GPU run",
        "standing_authorization": artifact(root, approval),
        "maximum_total_spend": 20, "project_reservations_before_usd": 135, "project_reservations_after_usd": 155,
        "maximum_leases": 1, "maximum_hours_per_lease": 6, "leases_must_be_sequential": True,
        "automatic_expiry_and_cleanup_required": True, "training_requires_frozen_experiment_contract": True,
        "prior_gpu_ledger": artifact(root, root / "organism/evidence/admitted-l4-execution-20261015-01/ledger.json"),
        "new_gpu_allocations_at_freeze": 0, "invoice_reconciliation_claimed": False,
    }
    compute_path = public_output / "compute-allocation.json"
    write(compute_path, compute)
    data = {
        "schema": "organism-automated-data-verification-v1", "experiment_id": recipe["experiment_id"],
        "recipe_sha256": recipe_sha, "input_artifacts": recipe["inputs"], "counts": recipe["counts"],
        "checks": {k: True for k in DATA_VERIFICATION_CHECKS}, "passed": True,
        "human_review_required": False, "human_review_verified": False,
        "evidence": {k: artifact(root, p) for k, p in
                     (("preflight", preflight_path), ("runtime", runtime_path), ("targets", row_path))},
    }
    data_path = public_output / "data-verification.json"
    write(data_path, data)
    contract = deepcopy(old_contract)
    contract.update(experiment_id=recipe["experiment_id"], optimization_started=False, qualification_passed=False,
                    final_release_bound=False, previous_frozen_experiment=artifact(root, parent_public / "experiment-contract.frozen.json"),
                    phase0_scope="Same base and evaluator; separately verified development-only data refinement.")
    contract.pop("frozen_at")
    contract.pop("previous_draft", None)
    contract["phase0_evidence"] = artifact(root, data_path)
    contract["training_preparation"].update(
        recipe=artifact(root, recipe_path), data_verification=artifact(root, data_path),
        maximum_optimizer_steps=264, maximum_total_gpu_spend_usd=20,
        maximum_leases=1, leases_consumed_at_freeze=0, leases_remaining_at_freeze=1)
    contract["development_comparison"] = {
        "pre_refinement_evidence": artifact(root, comparison), "prior_optimizer_steps": 210,
        "new_recipe_optimizer_steps_at_freeze": 0, "initialization": "fresh adapter on the same pinned base",
        "arms": ["stock base", "original completed v4 adapter", "completed v5 refinement adapter"],
        "candidate": "final second-epoch checkpoint; retain both epochs and all failures",
        "comparison": "Same providers, prompts, tool limits, export precision and public development tasks.",
        "qualification_feedback_allowed_in_training": False, "thresholds_changed": False,
    }
    seal = deepcopy(old_seal)
    seal.update(experiment_id=recipe["experiment_id"], recipe_sha256=recipe_sha,
                candidate_manifest_sha256=recipe["inputs"]["source_manifest"]["sha256"],
                token_report_sha256=recipe["inputs"]["token_report"]["sha256"],
                sealed_at=datetime.now(timezone.utc).isoformat(), commitments=commitments,
                receipt_scope="Unchanged original evaluator commitments verified file by file; new recipe inputs independently excluded before refinement.",
                previous_seal=artifact(root, parent_public / "qualification-seal.json"),
                optimizer_steps_at_seal=0, qualification_executed=False, qualification_passed=False)
    seal_path = public_output / "qualification-seal.json"
    write(seal_path, seal)
    contract["training_admission"].update(
        recipe_sha256=recipe_sha, candidate_manifest_sha256=recipe["inputs"]["source_manifest"]["sha256"],
        token_report_sha256=recipe["inputs"]["token_report"]["sha256"],
        data_verification_method="automated", data_verification_verified=True,
        data_verification_evidence=artifact(root, data_path), qualification_splits_sealed=True,
        qualification_seal=artifact(root, seal_path), compute_authorization_verified=True,
        compute_authorization_evidence=artifact(root, compute_path))
    contract["frozen_at"] = datetime.now(timezone.utc).isoformat()
    contract_path = public_output / "experiment-contract.frozen.json"
    write(contract_path, contract)
    require(not admission_errors(contract, recipe, recipe_sha), "Refinement admission failed")
    verify_admission_evidence(root, contract, recipe, recipe_sha)
    kit = public_output / "training-kit"
    build(root, kit, mode="train", contract_path=contract_path, recipe_path=recipe_path,
          instance_type="g6.4xlarge", region="us-east-1")
    archive = verify_archive(kit, reserved)
    result = {
        "passed": True, "at": datetime.now(timezone.utc).isoformat(), "recipe": artifact(root, recipe_path),
        "contract": artifact(root, contract_path), "qualification_seal": artifact(root, seal_path),
        "inherited_commitment_files_verified": inherited, "refinement_exclusion": exclusion_result,
        "training_archive_exclusion": archive, "compiler": artifact(root, Path(__file__)),
        "optimizer_steps_for_this_recipe": 0, "cloud_resources_created": False,
        "training_completed": False, "qualification_passed": False,
    }
    write(output / "COMPLETE.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "recipe", "output", "public-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = compile_refinement(args.root, args.recipe, args.output, args.public_output)
    print(json.dumps({k: result[k] for k in ("passed", "optimizer_steps_for_this_recipe",
                                           "cloud_resources_created", "qualification_passed")}))
