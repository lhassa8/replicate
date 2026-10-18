"""Bind a complete development comparison to an independently checked export."""
import hashlib
import json
import os
from pathlib import Path

from .audit_cpu_export import numerical_readback
from .safe_archive import safe_relative

ARMS = ("stock-control", "trained-merged")
HISTORICAL_RESPONSE_AUDITORS = {
    "e70bbd24e8e17921f5e1efefaa11b4356d14f01babe6aa54a30af2b91b5a7871":
        "organism/evidence/refinement-export-preparation-20261016-01/before/audit_export_responses.py",
    "1955d771fe2865162ca2cb0fbe5e2cae91470a09012cd1c05659f421936415c4":
        "organism/evidence/first-lineage-calibration-preparation-20261016-01/before/audit_export_responses.py",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Expected a regular bound input")
    with path.open("rb") as source:
        value = hashlib.file_digest(source, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": value}


def runtime_inventory(root, inputs, *, expected_links=None):
    """Verify pinned library bytes and confine aliases to this runtime directory."""
    root = Path(root).resolve()
    directory = root / "tools/llama.cpp/b9850/llama-b9850"
    links = {}
    for name, expected in inputs.items():
        path = root / safe_relative(name)
        require(path.parent == directory and path.resolve().parent == directory.resolve(),
                "Runtime file or library alias escapes its declared directory")
        require(identity(path.resolve()) == expected, "Pinned inference runtime changed")
        if path.is_symlink():
            links[name] = os.readlink(path)
    if expected_links is not None:
        require(links == expected_links, "Pinned runtime library aliases changed")
    return links


def check_workload(workload, admission):
    require(workload["passed"] is True and workload["failure"] is None
            and workload["export_pair_verified"] is True,
            "A completed verified export pair is required")
    require(workload["training_admission"] == admission
            and type(workload["optimizer_steps_this_stage"]) is int and workload["optimizer_steps_this_stage"] == 0
            and workload["quality_selection_performed"] is False and workload["qualification_passed"] is False,
            "Export belongs to different training or performed new optimizer updates or quality selection")
    # The current publisher leaves this broader flag false after numerical
    # export. Preserve it in the binding; it is not the matched-pair success gate.
    require(type(workload["deployment_export_complete"]) is bool, "Invalid deployment status")


def check_retained_model(expected, observed, parameters):
    require({key: observed[key] for key in ("bytes", "sha256")} == expected
            and observed["parameters"] == parameters,
            "Retained model differs from the model checked during numerical export")


def check_response_auditor(recorded_source, current_source):
    """Recognize the preserved auditor without executing evidence-supplied code."""
    if recorded_source == current_source:
        return
    require(isinstance(recorded_source, dict)
            and recorded_source.get("sha256") in HISTORICAL_RESPONSE_AUDITORS,
            "Recorded response auditor is not a reviewed version")
    source = Path(__file__).resolve().parents[1] / HISTORICAL_RESPONSE_AUDITORS[recorded_source["sha256"]]
    require(identity(source) == recorded_source, "Preserved response auditor changed")


def checked_precision_evidence(directory, exported):
    """Reconstruct versioned precision exports before any downstream admission."""
    from . import audit_export_responses
    from .audit_export_responses import audit, precision_specification

    spec = precision_specification(exported)
    if spec is None:
        return {}
    directory = Path(directory)
    audit_path = directory / "independent-response-audit.json"
    job = directory / "inputs/export-job.json"
    prepared = directory / "inputs/export-calibration"
    source = Path(__file__).resolve().parents[1] / "tools/export/llama.cpp-4f31eedb0ccf546b7e8d6bb243b170f12522f54d"
    recorded = json.loads(audit_path.read_text())
    current_source = identity(Path(audit_export_responses.__file__))
    check_response_auditor(recorded.get("source"), current_source)
    reconstructed = audit(directory / "returned", calibration=prepared, gguf_source=source, export_job=job)
    require(reconstructed["reconstruction_passed"] is True
            and reconstructed["numerical_gates_passed"] is True
            and reconstructed["qualification_passed"] is False,
            "Independent precision export reconstruction did not pass")
    require(reconstructed.get("source") == current_source, "Current response auditor binding differs")
    require({k: v for k, v in recorded.items() if k not in ("audited_at", "source")}
            == {k: v for k, v in reconstructed.items() if k not in ("audited_at", "source")},
            "Saved independent export audit differs from reconstruction")
    require(reconstructed["quantization_precision"]["specification"] == spec,
            "Reconstructed precision differs from the exported model")
    return {"quantization_precision": spec, "response_audit": identity(audit_path),
            "export_job": identity(job),
            "export_report": identity(directory / "returned/workload/export/report.json")}


def checked_pair(directory, admission):
    directory = Path(directory).resolve()
    audit_path = directory / "result-audit.json"
    audited = json.loads(audit_path.read_text())
    returned = directory / "returned"
    workload_path = returned / "workload/workload.json"
    workload = json.loads(workload_path.read_text())
    check_workload(workload, admission)
    require(audited["archive"] == identity(directory / "outputs/results.tar.gz"),
            "Audited result archive changed")
    for name, expected in audited["files"].items():
        path = returned / safe_relative(name)
        require(path.resolve().is_relative_to(returned), "Audited export path escapes its directory")
        require(identity(path) == expected, "An audited export file changed: " + name)
    numerical = numerical_readback(returned)
    require(audited["numerical_readback"]["numerical_gates_passed"] is True
            and audited["numerical_readback"]["reported_export_passed"] is True
            and numerical["numerical_gates_passed"] is True and numerical["reported_export_passed"] is True,
            "Frozen numerical export comparisons did not pass")
    models = {}
    exported = json.loads((returned / "workload/export/report.json").read_text())
    require(exported["admission"] == admission, "Numerical export belongs to a different training run")
    precision = checked_precision_evidence(directory, exported)
    for arm, label in zip(ARMS, ("stock", "trained")):
        name = "workload/export/" + label + "-q4.gguf"
        observed = exported["models"][label + "-q4"]
        check_retained_model(audited["files"][name], observed, admission["base_parameters"])
        models[arm] = {"path": str(returned / name), **audited["files"][name]}
    require(models[ARMS[0]]["sha256"] != models[ARMS[1]]["sha256"], "The two model arms have identical weights")
    require(2_000_000_000 <= admission["base_parameters"] <= 4_000_000_000,
            "Exported model is outside the required stored parameter range")
    return {**precision, "models": models, "total_stored_parameters": admission["base_parameters"],
            "export_directory": str(directory), "audit": identity(audit_path),
            "workload": identity(workload_path), "numerical_gates_passed": True,
            "publisher_deployment_export_complete": workload["deployment_export_complete"],
            "qualification_passed": False}


def check_model_binding(plan, arm, record):
    require(plan.get("model_representation") == "merged-q4" and tuple(plan["arms"]) == ARMS
            and set(plan["models"]) == set(ARMS), "Unexpected exported comparison arms")
    model = plan["models"][arm]
    require(record.get("model_input") == model and record.get("model_representation") == "merged-q4"
            and record["adapter_scale"] is None and record["loaded_adapters"] == []
            and record["final_adapters"] == [], "Exported arm model or adapter binding differs")
    start = record["server_start"]
    command = start["command"]
    require(command[0] == plan["binary_path"] and command.count("--model") == 1
            and command[command.index("--model") + 1] == model["path"]
            and start["properties"]["model_path"] == model["path"]
            and start["trained_for_organism"] is (arm == ARMS[1]),
            "Server did not load the declared model arm")
    require(not any(argument.startswith("--lora") for argument in command),
            "Exported model arm has an unexpected runtime adapter")
