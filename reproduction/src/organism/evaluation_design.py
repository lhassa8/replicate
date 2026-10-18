"""Check a draft evaluation design and enumerate unpopulated trial slots.

This does not generate qualification fixtures, seal a split, grade a model,
launch a device or authorize training. Profile names do not prove implemented
API diversity. A later evaluator must bind actual schemas, cases and oracles.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


DOMAINS = ("temperature", "vibration", "inspection", "machine_log")
ARCHITECTURES = ("x86_64", "arm64")
SUITES = {
    "Q4_Q5": (24, 6, ["Q4", "Q5"], ["trained", "fixed_recipe"]),
    "Q7": (20, 5, ["Q7"], ["trained_without_experience", "trained_with_verified_experience"]),
    "Q10": (64, 16, ["Q10"], ["trained", "matched_stock"]),
    "protected": (12, None, ["Q4", "Q7", "Q10"],
                  ["trained", "matched_stock", "trained_with_verified_experience"]),
}
FAULT_SUBCASES = {
    "inference_crash": ["inference_process_crash"],
    "organism_crash": ["organism_process_crash_supervisor_alive"],
    "seed_loss": ["original_seed_disconnected"],
    "later_peer_loss": ["later_peer_lost_mid_task"],
    "interrupted_birth": ["delivery_interrupted_before_completion"],
    "simultaneous_parents": ["two_parents_one_eligible_target"],
    "partition_rejoin": ["partition_then_rejoin"],
    "stale_observation": ["stale_device_or_admission_observation"],
    "corrupt_package": ["corrupted_carried_component"],
    "inherited_note_rejection": ["inapplicable_note", "instruction_bearing_note"],
    "retirement_restart": ["retire_then_restart_host_and_peer"],
    "admission_resource_rejection": ["ineligible_identity", "population_exhausted", "resource_exhausted"],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=1):
    return type(value) is int and value >= minimum


def unique(items, field, message):
    values = [item[field] for item in items]
    require(all(isinstance(value, str) and value.strip() for value in values)
            and len(set(values)) == len(values), message)


def validate(design, contract):
    require(design["schema"] == "organism-evaluation-design-v1"
            and design["experiment_id"] == contract["experiment_id"], "Experiment or schema differs")
    require(design["status"] == "draft_not_sealed"
            and all(design[key] is False for key in (
                "qualification_sealed", "qualification_executed", "training_admitted")),
            "Draft design cannot claim a seal, execution or training admission")
    require(design["architectures"] == list(ARCHITECTURES)
            and type(design["memory_mib"]) is int
            and design["memory_mib"] == contract["qualification_requirements"]["Q8"]["required_device_memory_mib"] == 8192,
            "Native architecture or device memory target changed")
    profiles = design["profiles"]
    require(len(profiles) == 12, "Exactly twelve environment profiles are required")
    unique(profiles, "id", "Duplicate or empty environment identity")
    unique(profiles, "family", "An API family is duplicated or crosses a split")
    unique(profiles, "interface_shape", "Relabeling an identical interface is not a distinct profile")
    require(Counter(profile["split"] for profile in profiles) == {
        "development_train": 6, "development_validation": 2, "qualification": 4},
        "Development and qualification allocation differs")
    require(all(profile["domain"] in DOMAINS for profile in profiles), "Unknown task domain")
    for split in ("development_train", "qualification"):
        require({profile["domain"] for profile in profiles if profile["split"] == split} == set(DOMAINS),
                "Training development and withheld profiles must each cover all task domains")
    for profile in profiles:
        require(isinstance(profile["interpretation"], str) and profile["interpretation"].strip(),
                "Profile interpretation is missing")
        require(profile["native_runtime_receipt"] is None,
                "A draft label cannot claim native runtime qualification")
        require(profile["split"] != "qualification" or profile["legacy_simulator_family"] is None,
                "Withheld profile reuses an existing development simulator family")
    legacy = [profile["legacy_simulator_family"] for profile in profiles if profile["split"] != "qualification"]
    require(set(legacy) == {f"dev-api-{letter}" for letter in "abcdefgh"},
            "Existing development family accounting differs")

    suites = design["suites"]
    require(len(suites) == len(SUITES), "Missing or additional evaluation suite")
    unique(suites, "id", "Duplicate evaluation suite")
    require({suite["id"] for suite in suites} == set(SUITES), "Unknown evaluation suite")
    for suite in suites:
        count, per_profile, gates, arms = SUITES[suite["id"]]
        require(type(suite["total_cases"]) is int and suite["total_cases"] == count
                and suite["gates"] == gates and suite["arms"] == arms, "Suite size, gates or arms changed")
        require(suite["reset_experience"] is True and suite["cross_trial_sharing"] is False
                and suite["training_use"] is False, "Evaluation leakage or experience carryover is allowed")
        if per_profile is not None:
            require(type(suite["cases_per_profile"]) is int and suite["cases_per_profile"] == per_profile
                    and suite["profile_split"] == "qualification", "Withheld case allocation differs")
        else:
            require(type(suite["cases_per_domain"]) is int and suite["cases_per_domain"] == 3
                    and suite["profile_split"] == "development", "Protected known-task allocation differs")
        conditions = suite["condition_groups"]
        require(isinstance(conditions, list) and conditions
                and all(isinstance(value, str) and value.strip() for value in conditions)
                and len(set(conditions)) == len(conditions), "Condition inventory is empty or duplicated")
        require(len(conditions) == {"Q4_Q5": 6, "Q7": 5, "Q10": 4, "protected": 3}[suite["id"]],
                "Condition coverage was reduced")
        require((per_profile or 3) % len(conditions) == 0, "Conditions do not evenly divide scheduled repetitions")

    pairing = design["pairing"]
    require(pairing["discard_failed_cases"] is False and pairing["qualification_feedback_to_training"] is False
            and all(value is True for key, value in pairing.items()
                    if key not in ("discard_failed_cases", "qualification_feedback_to_training")),
            "Matched comparison or failure retention rules changed")
    required_pairing = {
        "same_fixture_and_starting_state", "same_hardware_and_limits", "same_tools_and_observations",
        "same_prompts_for_model_arms", "same_context_and_quantization_family_for_model_arms",
        "freeze_all_model_and_baseline_artifacts_before_qualification", "precommit_balanced_arm_order",
        "retry_within_original_case_budget", "discard_failed_cases", "qualification_feedback_to_training",
    }
    require(set(pairing) == required_pairing, "Pairing rule inventory changed")
    effort = design["bounded_effort"]
    require(effort["metric"] == "task_wall_clock_seconds"
            and effort["failed_or_timed_out_charge"] == "full_case_budget"
            and effort["denominator"] == "every_scheduled_case"
            and effort["q7_note_creation_cost_reported_separately"] is True
            and effort["q7_note_validation_and_application_charged_to_case"] is True
            and effort["q7_target_answers_or_target_observations_in_notes"] is False,
            "Effort denominator, failed-case charge or inheritance separation changed")
    limits = design["provisional_task_limits"]
    require(limits["status"] == "requires_development_measurement_before_freeze",
            "Unmeasured task limits cannot be frozen")
    for key in ("wall_clock_seconds", "tool_calls", "model_calls", "context_tokens",
                "response_tokens", "concurrent_generations"):
        require(integer(limits[key]), "Task limits must be positive integers")
    require(limits["context_tokens"] == contract["model"]["initial_active_context_tokens"] == 4096
            and limits["concurrent_generations"] == contract["model"]["concurrent_generations_per_individual"] == 1,
            "Context or concurrency target changed")
    baseline = design["baseline"]
    require(baseline["competence_review_before_seal"] is True
            and baseline["qualification_specific_lookup_or_tuning"] is False
            and baseline["artificially_disable_supported_behavior"] is False
            and baseline["current_simulator_reference_is_qualification_baseline"] is False,
            "Baseline competence or separation rule changed")
    faults = design["q6_families"]
    require(len(faults) == 12, "All twelve Q6 fault families are required")
    unique(faults, "id", "Duplicate fault family")
    require({fault["id"]: fault["subcases"] for fault in faults} == FAULT_SUBCASES,
            "Q6 subcases are missing or changed")
    require(all(isinstance(fault["expected"], str) and fault["expected"].strip() for fault in faults),
            "Expected Q6 outcomes are missing")
    sealing = design["sealing"]
    require(sealing["status"] == "not_ready"
            and all(sealing[key] is False for key in ("case_seeds_created", "case_payloads_created",
                                                     "oracle_created", "qualified_runtime_interfaces_present"))
            and all(sealing[key] is None for key in ("case_manifest", "oracle_manifest", "baseline_manifest",
                                                    "access_receipt", "training_exclusion_audit")),
            "An unpopulated design cannot stand in for sealed evidence")
    gates = contract["qualification_requirements"]
    require(gates["Q4"] == {"blind_task_minimum_successes": 22, "trials": 24, "protected_regression_successes": 12}
            and gates["Q5"] == {"minimum_additional_successes_over_fixed_recipes": 3, "trials": 24}
            and gates["Q6"] == {"fault_control_families": 12, "every_subcase_must_pass": True}
            and gates["Q7"] == {"paired_cases": 20, "success_must_not_decrease": True,
                                "minimum_median_bounded_effort_reduction": 0.2}
            and gates["Q10"] == {"paired_cases": 64, "minimum_additional_successes_over_stock": 4,
                                 "alternative": {"success_must_not_decrease": True,
                                                 "minimum_median_bounded_effort_reduction": 0.2}},
            "The plan's task, recovery or contribution gates changed")


def trial_slots(design):
    """Return disjoint *empty* slots; arms reuse each slot's eventual fixture."""
    profiles = design["profiles"]
    slots = []
    for suite in design["suites"]:
        selected = []
        if suite["id"] == "protected":
            for domain in DOMAINS:
                candidates = [profile for profile in profiles
                              if profile["domain"] == domain and profile["split"] != "qualification"]
                selected.extend((candidates[index % len(candidates)], index)
                                for index in range(suite["cases_per_domain"]))
        else:
            for profile in profiles:
                if profile["split"] == "qualification":
                    selected.extend((profile, index) for index in range(suite["cases_per_profile"]))
        for suite_index, (profile, repetition) in enumerate(selected):
            slots.append({
                "slot_id": f"{suite['id']}/{profile['id']}/{repetition + 1:02d}",
                "suite": suite["id"], "profile": profile["id"], "family": profile["family"],
                "domain": profile["domain"], "architecture": ARCHITECTURES[suite_index % 2],
                "condition_group": suite["condition_groups"][repetition % len(suite["condition_groups"])],
                "arms": list(suite["arms"]), "case_seed": None, "fixture_sha256": None,
                "oracle_sha256": None, "starting_state_sha256": None,
                "source_episode_sha256": None, "inherited_notes_sha256": None,
                "status": "unpopulated_not_a_qualification_case",
            })
    return slots


def report(design, contract):
    validate(design, contract)
    slots = trial_slots(design)
    require(len({slot["slot_id"] for slot in slots}) == len(slots), "Trial slot collision")
    counts = {}
    for suite in design["suites"]:
        selected = [slot for slot in slots if slot["suite"] == suite["id"]]
        architectures = dict(Counter(slot["architecture"] for slot in selected))
        require(len(selected) == suite["total_cases"]
                and architectures == {architecture: len(selected) // 2 for architecture in ARCHITECTURES},
                "Scheduled count or native architecture balance differs")
        counts[suite["id"]] = {
            "distinct_slots": len(selected), "arm_executions": sum(len(slot["arms"]) for slot in selected),
            "architectures": architectures, "families": dict(Counter(slot["family"] for slot in selected)),
        }
    return {
        "schema": "organism-evaluation-design-check-v1",
        "design_consistent": True,
        "scope": "Public draft design and empty schedule only; no fixture or model execution",
        "environment_profiles": len(design["profiles"]),
        "development_profiles": 8, "withheld_profiles": 4,
        "qualification_slots": 108, "protected_slots": 12,
        "scheduled_arm_executions": sum(value["arm_executions"] for value in counts.values()),
        "q6_families": len(FAULT_SUBCASES),
        "q6_subcases": sum(len(value) for value in FAULT_SUBCASES.values()),
        "suites": counts,
        "qualification_sealed": False, "ready_to_seal": False, "training_admitted": False,
        "native_profile_execution_verified": False, "model_benefit_established": False,
        "withheld_family_separation_verified": False, "fixture_independence_verified": False,
        "blockers": [
            "Native interfaces, withheld descriptors and frozen independent oracles remain unverified.",
            "The implemented development fixed policy still needs competence review before becoming the qualification baseline.",
            "Exact case seeds, fixtures, source episodes, expected outcomes and reset states are absent.",
            "Extended diagnostic budgets need measurement on development cases before freeze.",
            "Training exclusion and qualification custody need artifact-level audits.",
        ],
    }, slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=Path(__file__).with_name("evaluation-design.draft.json"))
    parser.add_argument("--contract", type=Path, default=Path(__file__).with_name("experiment-contract.draft.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_design, raw_contract = args.design.read_bytes(), args.contract.read_bytes()
    checked, slots = report(json.loads(raw_design), json.loads(raw_contract))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, raw in (("design.draft.json", raw_design), ("contract.draft.json", raw_contract)):
        (args.output / name).write_bytes(raw)
    schedule_path = args.output / "empty-schedule.json"
    schedule_path.write_text(json.dumps(slots, indent=2) + "\n")
    checked["recorded_at"] = datetime.now(timezone.utc).isoformat()
    checked["inputs"] = {
        name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in (("design", raw_design), ("contract", raw_contract),
                          ("checker_source", Path(__file__).read_bytes()),
                          ("empty_schedule", schedule_path.read_bytes()))
    }
    (args.output / "report.json").write_text(json.dumps(checked, indent=2) + "\n")
    print(json.dumps(checked, indent=2))


if __name__ == "__main__":
    main()
