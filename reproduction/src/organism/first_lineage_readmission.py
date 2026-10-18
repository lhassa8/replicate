"""Admit one fresh allocation after the retained pre-training wrapper failure."""

from copy import deepcopy
from pathlib import Path

from .training_inputs import load, require, sha256


def review(root, recipe_path, output):
    root = Path(root)
    previous = root / "organism/evidence/first-lineage-managed-training-20261016-01"
    diagnostic = root / "organism/evidence/first-lineage-isolation-diagnostic-20261016-01"
    original_path = root / "organism/training-recipe.first-lineage-v6-01.draft.json"
    require(sha256(original_path) == "9de362da8f14b75f82fb0e962ba8411c1d855e68ed60360eae209a9101ccfddd",
            "Previous recipe changed")
    expected = deepcopy(load(original_path))
    expected["experiment_id"] = expected["experiment_id"].removesuffix("-01") + "-02"
    require(load(recipe_path) == expected
            and Path(recipe_path) == root / "organism/training-recipe.first-lineage-v6-02.draft.json"
            and Path(output) == root / "organism/evidence/first-lineage-freeze-20261016-02",
            "Readmission changed more than the separate attempt identity")
    from .training_worker.managed_control import verify_job
    from .training_worker.managed_bootstrap import identity
    job = load(previous / "job-terminal.json")
    config_path = previous / "source/managed-input.json"
    verified = verify_job(job, sha256(config_path))
    require(verified["status"] == "Failed" and not verified["training_completed"],
            "Previous allocation is not terminal and unsuccessful")
    ledger = load(previous / "allocation-consumed.json")
    require(ledger["allocated_gpu_leases"] == 1 and ledger["optimizer_steps"] == 0
            and ledger["compute_stopped"] is True, "Previous allocation was not consumed and stopped")
    host_path = previous / "downloaded/output/host-receipt.json"
    host = load(host_path)
    log_path = previous / "downloaded/output/workload.log"
    log = log_path.read_text()
    require(host["managed_input"] == identity(config_path)
            and host["workload_return_code"] == 1 and host["training_completed"] is False
            and host["workload_stopped_confirmed"] is True and host["processes_after_stop"]["pids"] == []
            and "isolation = observe_isolation()" in log
            and 'raise ValueError("Container networking or privilege isolation differs")' in log,
            "Previous result is not the retained pre-optimizer isolation failure")
    diagnosis = load(diagnostic / "reconciliation.json")
    require(diagnosis["passed"] is True and diagnosis["cloudwatch_equals_downloaded_output"] is True
            and diagnosis["source_bound_to_actual_job"] is True and diagnosis["compute_stopped"] is True
            and diagnosis["temporary_compute_roles_and_schedule_cleanup_complete"] is True
            and diagnosis["optimizer_steps"] == 0 and diagnosis["model_loaded"] is False,
            "Separate diagnostic evidence is incomplete")
    from .training_worker.managed_runner import validate_isolation
    observation = deepcopy(load(diagnostic / "downloaded/output/diagnostic.json")["child"])
    observation.update(mechanism="sagemaker-traffic-isolation-v2", platform="Linux", architecture="x86_64",
                       host_network_namespace_compared=False,
                       control_plane_configuration_requires_separate_readback=True)
    validate_isolation(observation)
    return {
        "preflight_path": root / "organism/evidence/first-lineage-training-preparation-20261016-02/preflight/report.json",
        "reservation_before_usd": 312,
        "previous_attempt_evidence": {
            "original_recipe": original_path, "terminal_job": previous / "job-terminal.json",
            "consumed_allocation": previous / "allocation-consumed.json", "host": host_path, "workload_failure": log_path,
            "diagnostic_reconciliation": diagnostic / "reconciliation.json",
            "actual_diagnostic": diagnostic / "downloaded/output/diagnostic.json",
        },
    }
