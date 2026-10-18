"""Evaluator-only prospective trials, including separate inheritance inputs.

No model, baseline, worker, or deployed application imports this module.
Source episodes and signed notes are future outcomes, never fabricated by the
input builder. The original 120 qualification inputs remain byte-for-byte intact.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .check_qualification_inputs import read_inputs
from .diagnostic_oracle import encoded, reconstruct, require
from .equipment_interfaces import descriptor, validate_document
from .experience import capability_hash
from .qualification_inputs import REFERENCE_TIME, entry, scalar_encoding, time_paths, write


SCHEMA = "organism-qualification-protocol-v1"
SOURCE_SCHEMA = "organism-inheritance-source-input-v1"
SOURCE_NAMES = (
    "qualification_protocol.py", "qualification_inputs.py", "check_qualification_inputs.py",
    "diagnostic_oracle.py", "equipment_interfaces.py", "experience.py",
)
LIMITS = {
    "wall_clock_seconds": 150, "model_calls": 6, "tool_calls": 8,
    "context_tokens": 4096, "response_tokens": 192, "concurrent_generations": 1,
}


def derived(seed, domain):
    return hashlib.sha256(bytes.fromhex(seed) + b"\0" + domain.encode()).hexdigest()


def assign(source, selector, value):
    """Populate a fresh fixture through a declared selector; never copy readings."""
    if source["format"] == "register_map":
        bank, address = selector.split(":")
        source["payload"].setdefault(bank, {})[address] = value
        return
    parts = [part.replace("~1", "/").replace("~0", "~") for part in selector[1:].split("/")]
    current = source["payload"]
    for index, key in enumerate(parts):
        last = index == len(parts) - 1
        if isinstance(current, list):
            key = int(key)
            while len(current) <= key:
                current.append(None)
        if last:
            current[key] = value
            return
        following = [] if parts[index + 1].isdigit() else {}
        if isinstance(current, list):
            if current[key] is None:
                current[key] = following
        else:
            current.setdefault(key, following)
        current = current[key]


def source_case(target):
    """Share capabilities only. Synthesize all source observations from a new seed."""
    seed = derived(target["case_seed"], "separate-live-source-input-v1")
    number = int(seed, 16)
    identity = "qual-source-" + seed[:32]
    published = descriptor(target["document"])
    document = {
        "schema": "organism-equipment-interface-v1", "device_id": identity,
        "family": published["family"], "task": published["task"],
        "sources": [{**source, "payload": {}} for source in published["sources"]],
        "sequence": number % 100000 + 1,
    }
    oracle = deepcopy(target["oracle_plan"])
    sample_time = REFERENCE_TIME - 7
    domain = target["slot"]["domain"]
    truth = {"temperature": 21.137 + number % 17 / 1000,
             "vibration": 1.173 + number % 17 / 1000,
             "inspection": 2.713 + number % 17 / 1000,
             "machine_log": 1 + number % 2}[domain]
    for source in document["sources"]:
        for field in source["fields"]:
            if field["kind"] == "number":
                value = scalar_encoding(truth, field["unit"], field["scale"])
            elif field["kind"] == "timestamp":
                value = sample_time
            elif field["kind"] == "validity":
                value = True
            else:
                require(field["kind"] == "records", "Undeclared source fixture field kind")
                continue
            assign(source, field["selector"], value)
    selected = next(source for source in document["sources"] if source["source_id"] == oracle["source_id"])
    operation = oracle["operation"]["kind"]
    if operation == "windowed_scalar":
        field = next(field for field in selected["fields"] if field["selector"] == oracle["value_selectors"][0])
        assign(selected, oracle["value_selectors"][1], [{
            "began": sample_time - 19, "ended": sample_time, "sequence": document["sequence"],
            "unit": field["unit"], "scale": field["scale"],
        }])
    elif operation == "counter_pair":
        accepted, rejected = 913 + number % 37, 23 + number % 11
        truth = 100 * rejected / (accepted + rejected)
        assign(selected, oracle["value_selectors"][0], [
            {"outcome": "accepted", "count": accepted, "batch": "source-" + seed[:12]},
            {"outcome": "rejected", "count": rejected, "batch": "source-" + seed[:12]},
        ])
    elif operation == "event_count":
        truth = 1 + number % 2
        rows = [{"event_id": "source-" + seed[:12] + "-" + str(index),
                 "category": "error", "observed": sample_time - index - 1} for index in range(truth)]
        rows.append(deepcopy(rows[0]))
        rows.append({"event_id": "source-" + seed[:12] + "-left",
                     "category": "error", "observed": sample_time - 43})
        assign(selected, oracle["value_selectors"][0], rows)
        assign(selected, oracle["value_selectors"][1], [{"start": sample_time - 43, "end": sample_time}])
    else:
        require(operation == "scalar", "Unsupported source operation")
    validate_document(document, identity)
    require(capability_hash(descriptor(document)) == capability_hash(published),
            "Source and target capabilities differ")
    result = reconstruct(encoded(document), oracle, now=REFERENCE_TIME)
    low, high = document["task"]["normal_range"]
    expected = {"status": "low" if truth < low else "high" if truth > high else "normal",
                "value": truth, "unit": document["task"]["canonical_unit"]}
    require(result["expected_status"] == expected["status"] and result["expected_unit"] == expected["unit"]
            and abs(result["expected_value"] - truth) < 1e-7,
            "Separate source does not match independent reconstruction")
    slot = deepcopy(target["slot"])
    slot.update(suite=target["slot"]["suite"] + "_source",
                slot_id="source/" + target["slot"]["slot_id"], arms=["trained_source"])
    return {
        "schema": SOURCE_SCHEMA, "status": "prospective_input_no_source_episode",
        "case_id": identity, "case_seed": seed, "target_case_id": target["case_id"], "slot": slot,
        "condition": "separate_fresh_observation", "document": document, "oracle_plan": oracle,
        "expected": expected, "arm_order": ["trained_source"], "fault": None,
        "capability_sha256": capability_hash(published),
        "starting_state": {"experience": [], "reader_configuration": None, "prior_tasks": [],
                           "cross_case_sharing": False},
        "time_binding": {"kind": "per_arm_trial_origin_preserves_frozen_offsets",
                         "reference_unix_seconds": REFERENCE_TIME, "fields": time_paths(document)},
        "source_episode": None, "signed_note": None, "model_calls": 0,
        "training_use": False, "source_execution_verified": False, "qualification_passed": False,
    }


def lifecycle_trials(master_seed):
    trials = []
    for gate, count, generations in (("Q1", 6, 1), ("Q2", 10, 3), ("Q3", 3, 4), ("Q8", 1, 3)):
        for index in range(count):
            seed = derived(master_seed, f"lifecycle-v1/{gate}/{index + 1}")
            architectures = [("x86_64", "arm64")[(index + generation) % 2] for generation in range(generations)]
            trial = {
                "trial_id": "qual-life-" + seed[:32], "seed": seed, "gate": gate,
                "ordinal": index + 1, "architectures": architectures, "memory_mib": 8192,
                "maximum_enrolled_identities": 6, "owner_bootstraps_only_seed": True,
                "descendants_activated_only_by_authenticated_peer_reproduction": True,
                "starting_state": "empty substrate, empty caches, empty experience and durable work",
                "independent_native_hosts": True, "external_model_or_registry_access": False,
                "birth_deadline_seconds": 1200, "healthy_window_seconds": 30,
                "every_generation_requires_useful_diagnosis_and_both_platform_payloads": True,
                "retry_within_original_deadline": True, "replace_failed_trial": False,
            }
            if gate == "Q1":
                trial["events"] = ["owner_seed_bootstrap", "empty_cache_offline_cold_start",
                                   "independent_useful_diagnosis", "complete_package_audit"]
            elif gate in ("Q2", "Q3"):
                trial["events"] = ["owner_seed_bootstrap", "seed_useful_and_healthy"]
                for generation in range(1, generations):
                    if generation >= 2:
                        trial["events"].append(f"disconnect_generation_{generation - 2}_through_trial_end")
                    trial["events"].extend([
                        f"observe_generation_{generation - 1}_useful_and_healthy_for_30_seconds",
                        f"introduce_enrolled_empty_generation_{generation}",
                        f"observe_peer_initiated_complete_useful_generation_{generation}",
                    ])
            else:
                trial.update({
                    "initial_population": 2, "new_arrival_population": 3,
                    "clock_anchor": "both initial architecture profiles independently useful and healthy",
                    "sustained_seconds": 72 * 3600,
                    "arrival_seconds_after_anchor": 48 * 3600 + 60,
                    "useful_task_interval_seconds": 600,
                    "task_phase_seconds_by_generation": [0, 200, 400],
                    "gap_or_missing_observation_invalidates_continuous_health": True,
                    "all_diagnostic_tasks_must_pass": True,
                    "clock_reset_or_extending_failed_run_allowed": False,
                    "initial_second_peer_requires_autonomous_birth": True,
                })
            trials.append(trial)
    return trials


def protocol(master_seed, sources):
    return {
        "schema": SCHEMA, "status": "precommitted_inputs_pending_experiment_seal",
        "experiment_id": "organism-3b-20261013-01",
        "limits": deepcopy(LIMITS),
        "resource_requirements": {
            "device_memory_mib": 8192, "organism_aggregate_peak_memory_mib": 6144,
            "cores": 4, "swap_bytes": 0, "total_organism_storage_bytes": 32 * 1024**3,
            "network_bytes_per_second": 8 * 1024**2, "network_burst_bytes": 16 * 1024**2,
            "healthy_background_fraction_of_one_core": 0.05,
            "resource_measurements_include": ["runtime", "model", "receiver", "staging",
                                            "inference", "durable_state", "partial_transfers"],
            "storage_includes": ["incoming", "capsule_store", "prepared_installations", "state", "logs"],
            "measurement_and_enforcement_verified_before_qualification": True,
            "measurement_or_enforcement_currently_claimed": False,
        },
        "lifecycle_trials": lifecycle_trials(master_seed),
        "source_to_target": [{
            "source_case_id": source["case_id"], "target_case_id": source["target_case_id"],
            "source_case_sha256": hashlib.sha256(encoded(source)).hexdigest(),
            "target_suite": source["slot"]["suite"].removesuffix("_source"),
        } for source in sources],
        "inheritance": {
            "source_model": "same frozen trained deployment export used for targets",
            "source_tools_and_prompts": "ordinary equipment diagnostic tools; no prior notes",
            "source_task_limits": deepcopy(LIMITS), "source_attempts": 1,
            "source_cost_reported_separately": True,
            "source_success_requires_live_model_trace_and_independent_diagnosis": True,
            "source_notes_require_distinct_enrolled_producer_and_verifier": True,
            "source_execution_before_any_target_arm": True,
            "source_failure": "retain failure; no replacement source; inheritance gate cannot pass",
            "note_format": "organism-reader-experience-v1",
            "notes_may_contain": "only the existing typed reader procedure and signed source references",
            "target_observations_answers_and_traces_in_notes": False,
            "without_experience_arm": "same experienced tools and prompt with empty verified note store",
            "with_experience_arm": "same experienced tools and prompt with its one verified source note",
            "validation_transport_and_store_preparation_charged_to_target": True,
            "reset_each_arm": ["reader", "job_history", "conversation", "experience_store"],
            "protected_inherited_arm": "one separate live source per protected case; same validation rules",
            "source_episodes_or_notes_claimed_present": False,
        },
        "release_admission_before_qualification": [
            "bind one final release and matched stock export; neither may change between gates",
            "bind fixed baseline source, configuration and development competence evidence",
            "verify native clock materialization and all declared fault adapters on public fixtures",
            "verify whole-device resource observation and enforcement on both architectures",
            "bind native host identities, distinct placement, enrollment and mission expiry",
            "verify complete offline packages, prompts, tools, tokenizer and quantization identities",
            "bind immutable source episode and note receipts before the paired target arms",
            "audit evaluator-only files absent from each training and deployment package",
            "obtain any still-missing native qualification compute authorization",
        ],
        "failure_policy": {
            "all_attempts_retained": True, "case_replacement_or_resampling": False,
            "model_failure_timeout_wrong_answer_or_budget_exhaustion": "count failure; charge full task budget",
            "infrastructure_failure": "mark gate incomplete; preserve evidence; never convert to model success",
            "infrastructure_rerun": "at most one complete affected suite rerun, both arms, identical release, inputs and order",
            "rerun_authority": "independently recorded external infrastructure failure, never model behavior",
            "rerun_cost_authorized_in_advance": True,
            "second_infrastructure_failure": "gate incomplete; no further automatic retries",
            "corrected_release": "new release identity; rerun all gates on that single release",
            "qualification_feedback_to_training": False,
        },
        "custody": {
            "case_author": "same project agent",
            "author_has_accessed_expected_answers": True,
            "independent_human_blinding_claimed": False,
            "same_mac_user_is_security_boundary": False,
            "required_boundary": "exact allowlist of training inputs and deployed capsule contents",
            "case_generators_or_seeds_or_observations_or_answers_shipped_to_training": False,
            "source_and_target_model_runs": 0,
        },
        "qualification_executed": False, "qualification_passed": False, "training_admitted": False,
    }


def create(inputs, output):
    inputs, output = Path(inputs), Path(output)
    manifest, cases = read_inputs(inputs)
    # No new random seed: all added slots use domain-separated original seeds.
    master = json.loads((inputs / "seed.json").read_bytes())["seed"]
    sources = [source_case(case) for case in cases if case["slot"]["suite"] in ("Q7", "protected")]
    require(len(sources) == 32 and len({case["case_id"] for case in sources}) == 32,
            "Inheritance source inventory differs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source-inputs").mkdir()
    (output / "sources").mkdir()
    write(output / "request.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": entry((inputs / "manifest.json").read_bytes()),
        "seed_receipt": manifest["seed_receipt"], "new_random_seed": False,
    })
    for source in sources:
        write(output / "source-inputs" / (source["case_id"] + ".json"), source)
    write(output / "protocol.json", protocol(master, sources))
    for name in SOURCE_NAMES:
        (output / "sources" / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    files = {str(path.relative_to(output)): entry(path.read_bytes())
             for path in sorted(output.rglob("*")) if path.is_file()}
    result = {
        "schema": "organism-qualification-protocol-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": entry((inputs / "manifest.json").read_bytes()),
        "files": files, "source_inputs": len(sources), "lifecycle_trials": 20,
        "qualification_sealed": False, "source_episodes_executed": 0,
        "training_admitted": False, "qualification_passed": False,
    }
    write(output / "manifest.json", result)
    return result


def check(inputs, output):
    inputs, output = Path(inputs), Path(output)
    _, targets = read_inputs(inputs)
    manifest = json.loads((output / "manifest.json").read_bytes())
    require(manifest["schema"] == "organism-qualification-protocol-manifest-v1"
            and manifest["input_manifest"] == entry((inputs / "manifest.json").read_bytes()),
            "Prospective protocol input binding changed")
    require(manifest["source_inputs"] == 32 and manifest["lifecycle_trials"] == 20
            and manifest["source_episodes_executed"] == 0
            and all(manifest[key] is False for key in
                    ("qualification_sealed", "training_admitted", "qualification_passed")),
            "Prospective protocol scope changed")
    actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    require(actual == set(manifest["files"]) | {"manifest.json"}, "Protocol file inventory changed")
    for name, receipt in manifest["files"].items():
        path = output / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and path.resolve().is_relative_to(output.resolve()) and not path.is_symlink()
                and entry(path.read_bytes()) == receipt, "Protocol file changed or escapes its directory")
    for name in SOURCE_NAMES:
        require((output / "sources" / name).read_bytes() == Path(__file__).with_name(name).read_bytes(),
                "Protocol builder source changed")
    sources = [source_case(target) for target in targets if target["slot"]["suite"] in ("Q7", "protected")]
    for source in sources:
        require(json.loads((output / "source-inputs" / (source["case_id"] + ".json")).read_bytes()) == source,
                "Source input differs from original seeded reconstruction")
    seed = json.loads((inputs / "seed.json").read_bytes())["seed"]
    require(json.loads((output / "protocol.json").read_bytes()) == protocol(seed, sources),
            "Prospective protocol differs from seeded reconstruction")
    return {"passed": True, "source_inputs": 32, "lifecycle_trials": 20,
            "original_inputs": 120, "source_episodes_executed": 0,
            "model_calls": 0, "qualification_sealed": False, "qualification_passed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = check(args.inputs, args.output) if args.check else create(args.inputs, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}))


if __name__ == "__main__":
    main()
