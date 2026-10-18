"""Evaluate the frozen temperature milestone using existing matched runners."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import shutil

from .audit_trained_tasks import audit as audit_exported_tasks
from .export_task_inputs import ARMS, identity, require, runtime_inventory
from .first_lineage_cases import load_frozen, validate_cases
from .probe_exported_tasks import make_plan
from .probe_trained_tasks import LIMITS, run_arm, write


MATCHED_FIELDS = (
    "system_prompt", "tool_specs", "tool_profile", "arms", "model_representation", "models",
    "export_pair", "training_admission", "previous_runtime_binding",
    "explicit_training_recipe", "binary_path", "runtime_inputs", "runtime_links",
    "limits", "source_files", "total_stored_parameters", "versions",
)


def original_comparison_binding(directory, recipe_path, admission_path):
    """Keep the completed v5 comparison as history for its exact v6 successor."""
    directory = Path(directory)
    original_plan = json.loads((directory / "plan.json").read_text())
    original_report = json.loads((directory / "report.json").read_text())
    original_admission = json.loads((directory / "training-admission.json").read_text())
    original_recipe = json.loads((directory / "training-recipe.json").read_text())
    recipe = json.loads(Path(recipe_path).read_text())
    admission = json.loads(Path(admission_path).read_text())
    original_recipe_id = identity(directory / "training-recipe.json")
    original_admission_id = identity(directory / "training-admission.json")
    require(original_report["plan"] == identity(directory / "plan.json")
            and original_report["failure"] is None
            and original_report["sources_unchanged"] is True
            and original_report["inputs_unchanged"] is True
            and original_report["qualification_passed"] is False
            and original_plan["explicit_training_recipe"] == original_recipe_id == original_admission["recipe"]
            and original_plan["training_admission"] == original_admission_id
            and admission["recipe"] == identity(recipe_path),
            "Original or current comparison recipe/admission binding changed")
    same_training = (original_recipe_id == identity(recipe_path)
                     and original_admission_id == identity(admission_path))
    if not same_training:
        require(recipe.get("corpus") == "first_lineage_runtime",
                "Only the explicit first-lineage successor may use a historical comparison")
        parent = recipe["inputs"]
        require({key: parent["parent_recipe"][key] for key in ("bytes", "sha256")} == original_recipe_id
                and {key: parent["parent_contract"][key] for key in ("bytes", "sha256")}
                == original_admission["frozen_contract"],
                "Historical comparison is not the candidate's exact parent recipe and contract")
        require(recipe["base"] == original_recipe["base"]
                and admission["base_parameters"] == original_admission["base_parameters"],
                "Historical and current comparisons changed the foundation model")
    return {
        "relationship": "same_training" if same_training else "preserved_parent_history",
        "original_files": {name: identity(directory / name) for name in
                           ("plan.json", "report.json", "training-recipe.json", "training-admission.json")},
        "candidate_recipe": identity(recipe_path),
        "candidate_admission": identity(admission_path),
        "original_results_counted_as_candidate_results": False,
    }


def batch_cases(cases, number):
    """Two balanced ten-case batches preserve the existing per-arm allowance."""
    validate_cases(cases)
    require(number in (1, 2), "Expected temperature batch one or two")
    offset = (number - 1) * 2
    return [cases[category * 4 + variant]
            for category in range(5) for variant in (offset, offset + 1)]


def check_batch_plan(plan, cases, expected_identity, number):
    require(plan["first_lineage"]["batch"] == number
            and plan["first_lineage"]["frozen_cases"] == expected_identity,
            "Temperature batch does not bind the original frozen cases")
    require(plan["cases"] == batch_cases(cases, number),
            "Temperature batch inputs or answers differ from the frozen cases")


def check_matched_batches(plans):
    require(len(plans) == 2, "Expected two complete temperature batches")
    for field in MATCHED_FIELDS:
        require(plans[0][field] == plans[1][field],
                "Temperature batches changed " + field)


async def run_batch(root, output, admission, export, recipe, frozen, expected_identity, number):
    output.mkdir()
    cases = load_frozen(frozen, expected_identity["sha256"])
    plan = make_plan(root, output, admission, export, training_recipe_path=recipe)
    shutil.copyfile(frozen, output / "temperature-cases.frozen.json")
    plan["scope"] = "Balanced batch of the twenty frozen first-lineage temperature development cases"
    plan["cases"] = batch_cases(cases, number)
    plan["first_lineage"] = {"batch": number, "frozen_cases": expected_identity,
                            "total_cases": 20, "full_qualification": False}
    for name in ("first_lineage_cases.py", "probe_first_lineage_tasks.py"):
        source = root / "organism" / name
        shutil.copyfile(source, output / "source-snapshot" / name)
        plan["source_files"][name] = identity(source)
    check_batch_plan(plan, cases, expected_identity, number)
    write(output / "plan.json", plan)
    report = {
        "schema": "organism-exported-development-tasks-v1",
        "plan": identity(output / "plan.json"),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "arms": {}, "failure": None, "qualification_passed": False,
    }
    try:
        for arm in ARMS:
            record = await run_arm(root, output, arm, plan, model_override=plan["models"][arm])
            report["arms"][arm] = record
            require(record["failure"] is None, "Arm execution failed: " + str(record["failure"]))
            require(record["shutdown"]["owned_server_stopped"] is True,
                    "Owned model server did not stop")
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        report["sources_unchanged"] = all(
            identity(root / "organism" / name) == expected
            for name, expected in plan["source_files"].items())
        try:
            runtime_inventory(root, plan["runtime_inputs"], expected_links=plan["runtime_links"])
            report["inputs_unchanged"] = (
                identity(frozen) == expected_identity
                and identity(recipe) == plan["explicit_training_recipe"]
                and all(identity(Path(entry["path"]))
                        == {key: entry[key] for key in ("bytes", "sha256")}
                        for entry in plan["models"].values()))
        except (OSError, ValueError) as exc:
            report["inputs_unchanged"] = False
            report["failure"] = report["failure"] or "Input check failed: " + str(exc)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        write(output / "report.json", report)
    checked = audit_batch(output, cases, expected_identity, number)
    write(output / "audit.json", checked)
    return checked


def audit_batch(directory, cases, expected_identity, number):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    require(identity(directory / "temperature-cases.frozen.json") == expected_identity,
            "Saved frozen temperature cases changed")
    check_batch_plan(plan, cases, expected_identity, number)
    # This auditor independently reconstructs each raw tool/model trace and
    # verifies the actual exported model, prompt, budgets and server shutdown.
    return audit_exported_tasks(directory, exported=True)


def milestone_outcome(outcome, stored, template):
    """Keep correct runtime answers separate from unchanged milestone conditions."""
    checked = stored["evaluation"]
    preserved = (checked.get("expected_status") == template["expected_status"]
                 and checked.get("expected_value") == template["expected_value"])
    window = stored["clock_window"]
    numeric = lambda value: type(value) in (int, float) and math.isfinite(value)
    valid_window = (isinstance(window, list) and len(window) == 2
                    and all(numeric(value) for value in window)
                    and numeric(stored["fixture_epoch"])
                    and window[0] == stored["fixture_epoch"] - 1)
    seconds = window[1] - window[0] - 2 if valid_window else None
    if seconds is not None and not math.isfinite(seconds):
        seconds = None
    within_budget = seconds is not None and 0 <= seconds <= LIMITS["case_seconds"]
    runtime_passed = outcome["model_passed"]
    return {
        **outcome, "runtime_model_passed": runtime_passed,
        "frozen_condition_preserved": preserved, "wall_clock_seconds": seconds,
        "wall_clock_budget_preserved": within_budget,
        "model_passed": bool(runtime_passed and preserved and within_budget),
    }


def audit(directory, frozen, expected_sha256):
    directory, frozen = Path(directory), Path(frozen)
    cases = load_frozen(frozen, expected_sha256)
    expected_identity = identity(frozen)
    require(identity(directory / "temperature-cases.frozen.json") == expected_identity,
            "Run-wide temperature case snapshot changed")
    plans = [json.loads((directory / f"batch-{number:02d}" / "plan.json").read_text())
             for number in (1, 2)]
    check_matched_batches(plans)
    checked = [audit_batch(directory / f"batch-{number:02d}", cases, expected_identity, number)
               for number in (1, 2)]
    run_report = json.loads((directory / "report.json").read_text())
    current_recipe = json.loads((directory / "batch-01/training-recipe.json").read_text())
    require(current_recipe.get("corpus") != "first_lineage_runtime"
            or "original_comparison_binding" in run_report,
            "First-lineage successor lost its historical comparison binding")
    if "original_comparison_binding" in run_report:
        reconstructed = original_comparison_binding(
            directory / "original-comparison-snapshot",
            directory / "batch-01/training-recipe.json",
            directory / "batch-01/training-admission.json")
        require(reconstructed == run_report["original_comparison_binding"],
                "Preserved historical comparison binding changed")
    results = {}
    expected_ids = {item["case_id"] for item in cases}
    templates = {item["case_id"]: item for item in cases}
    for arm in ARMS:
        outcomes = []
        for number, batch in enumerate(checked, 1):
            for item in batch["arms"][arm]["cases"]:
                stored = json.loads((directory / f"batch-{number:02d}" / arm
                                     / (item["case_id"] + ".json")).read_text())
                outcomes.append(milestone_outcome(item, stored, templates[item["case_id"]]))
        require(len(outcomes) == 20 and {item["case_id"] for item in outcomes} == expected_ids,
                "Temperature comparison is incomplete or repeats cases")
        results[arm] = {
            "cases": outcomes,
            "successes": sum(item["model_passed"] for item in outcomes),
            "runtime_successes": sum(item["runtime_model_passed"] for item in outcomes),
            "fixed_policy_successes": sum(item["baseline_passed"] for item in outcomes),
            "all_trace_audits_passed": all(item["trace_audit_passed"] for item in outcomes),
        }
    trained = results["trained-merged"]
    return {
        "schema": "organism-first-lineage-temperature-audit-v2",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "frozen_cases": expected_identity, "evidence_reconstructed": True,
        "arms": results,
        "temperature_model_gate_passed": (
            trained["successes"] == 20
            and all(row["all_trace_audits_passed"] and row["fixed_policy_successes"] == 20
                    for row in results.values())),
        "native_linux_execution": False, "fine_tuning_superiority_established": False,
        "qualification_passed": False,
    }


async def run(output, admission, export, recipe, frozen, expected_sha256, original_comparison):
    require(platform.system() == "Darwin", "This runner is the Mac development comparison")
    require(original_comparison.is_dir(), "Complete the original matched comparison first")
    original = audit_exported_tasks(original_comparison, exported=True)
    require(original["evidence_reconstructed"], "Original comparison evidence did not reconstruct")
    binding = original_comparison_binding(original_comparison, recipe, admission)
    cases = load_frozen(frozen, expected_sha256)
    require(len(cases) == 20, "Expected the complete frozen temperature suite")
    root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(frozen, output / "temperature-cases.frozen.json")
    snapshot = output / "original-comparison-snapshot"
    snapshot.mkdir()
    for name, expected in binding["original_files"].items():
        shutil.copyfile(original_comparison / name, snapshot / name)
        require(identity(snapshot / name) == expected, "Original comparison changed during preservation")
    report = {
        "schema": "organism-first-lineage-temperature-run-v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "frozen_cases": identity(frozen),
        "original_comparison": str(original_comparison),
        "original_comparison_report": identity(original_comparison / "report.json"),
        "original_comparison_binding": binding,
        "completed_batches": [], "failure": None, "qualification_passed": False,
    }
    write(output / "report.json", report)
    try:
        for number in (1, 2):
            await run_batch(root, output / f"batch-{number:02d}", admission, export,
                            recipe, frozen, identity(frozen), number)
            report["completed_batches"].append(number)
            write(output / "report.json", report)
        checked = audit(output, frozen, expected_sha256)
        write(output / "audit.json", checked)
        print(json.dumps({
            "temperature_model_gate_passed": checked["temperature_model_gate_passed"],
            "successes": {arm: row["successes"] for arm, row in checked["arms"].items()},
        }), flush=True)
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        write(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "admission", "export-directory", "training-recipe",
                 "frozen-cases", "original-comparison"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cases-sha256", required=True)
    args = parser.parse_args()
    asyncio.run(run(
        args.output.resolve(), args.admission.resolve(), args.export_directory.resolve(),
        args.training_recipe.resolve(), args.frozen_cases.resolve(), args.cases_sha256,
        args.original_comparison.resolve(),
    ))
