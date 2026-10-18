"""Readback and harness-driven reader checks, not qualification execution."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time

from .audit_equipment_diagnostics import inspect_job
from .diagnostic_agent import EquipmentTool
from .diagnostic_oracle import encoded, reconstruct, require
from .equipment_diagnostic_agent import SPECS
from .equipment_interfaces import InterfaceDirectory, InterfaceSession
from .qualification_inputs import entry, fingerprint, generate, materialize, GENERATOR, REFERENCE_TIME, write


def read_inputs(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    require(manifest["schema"] == "organism-qualification-input-manifest-v1"
            and manifest["generator"] == GENERATOR
            and manifest["status"] == "draft_inputs_not_sealed" and manifest["failure"] is None
            and manifest["model_calls"] == manifest["baseline_runs"] == 0
            and all(manifest[key] is False for key in (
                "qualification_sealed", "ready_to_seal", "training_admitted",
                "native_execution_verified", "qualification_passed")), "Draft input scope or status differs")
    raw_seed = (directory / "seed.json").read_bytes()
    raw_design, raw_contract = ((directory / name).read_bytes()
                                for name in ("design.draft.json", "contract.draft.json"))
    require(entry(raw_seed) == manifest["seed_receipt"] and entry(raw_design) == manifest["design_receipt"]
            and entry(raw_contract) == manifest["contract_receipt"], "Input seed, design or contract changed")
    names = {"qualification_inputs.py", "diagnostic_oracle.py", "equipment_development.py", "evaluation_design.py"}
    require(set(manifest["source_receipts"]) == names, "Authoring source inventory differs")
    for name, receipt in manifest["source_receipts"].items():
        require(entry((directory / "sources" / name).read_bytes()) == receipt
                == entry(Path(__file__).with_name(name).read_bytes()), "Authoring source changed")
    seed_record = json.loads(raw_seed)
    require(seed_record["generator"] == GENERATOR, "Seed generator version differs")
    seed = seed_record["seed"]
    expected = generate(json.loads(raw_design), json.loads(raw_contract), seed)
    require(len(manifest["cases"]) == len(expected) == 120, "Draft case inventory differs")
    cases = []
    for index, row in enumerate(manifest["cases"]):
        require(isinstance(row["case_id"], str) and re.fullmatch(r"qual-[0-9a-f]{32}", row["case_id"]),
                "Invalid draft case identity")
        raw = (directory / "cases" / (row["case_id"] + ".json")).read_bytes()
        require(entry(raw) == row["receipt"], "Draft case bytes changed")
        case = json.loads(raw)
        require(case == expected[index] and case["case_id"] == row["case_id"]
                and case["slot"]["slot_id"] == row["slot_id"], "Draft case differs from its saved seed and allocation")
        for field, key in (("document", "document_sha256"), ("oracle_plan", "oracle_sha256"),
                           ("starting_state", "starting_state_sha256"), ("time_binding", "time_binding_sha256")):
            require(fingerprint(case[field]) == row[key], "Draft case content binding changed")
        cases.append(case)
    return manifest, cases


def expectation_matches(case, result):
    expected = case["expected"]
    return (result["expected_status"] == expected["status"] and result["expected_unit"] == expected["unit"]
            and (result["expected_value"] is None if expected["value"] is None else
                 result["expected_value"] is not None and abs(result["expected_value"] - expected["value"]) < 1e-7))


async def call(session, name, arguments):
    spec = next(spec for spec in SPECS if spec["name"] == name)
    tool = EquipmentTool(session, spec)
    output = [event async for event in tool.stream(
        {"toolUseId": "input-check-" + str(len(session.events)), "input": arguments}, {})]
    require(len(output) == 1, "Unexpected conformance tool response count")
    return json.loads(output[0].tool_result["content"][0]["text"])


async def exercise(case, workspace, epoch):
    """A harness supplies every choice; a model never sees qualification data."""
    document = materialize(case, epoch)
    raw = encoded(document)
    result = reconstruct(raw, case["oracle_plan"], now=epoch)
    require(expectation_matches(case, result), "Time translation changed the expected diagnostic")
    identity = case["case_id"]
    path = workspace / (identity + ".json")
    path.write_bytes(raw)
    session = InterfaceSession(InterfaceDirectory(workspace, clock=lambda: epoch), identity)
    await call(session, "inspect_equipment", {"device_id": identity})
    fault_receipt = None
    if case["fault"] is not None:
        missing = workspace / (identity + ".withheld")
        path.rename(missing)
        try:
            rejected = await call(session, "configure_reader", result["expected_reader"])
        finally:
            missing.rename(path)
        require(rejected == {"status": "invalid_observation_or_arguments", "error_type": "FileNotFoundError"}
                and session.configuration is None, "Configuration interruption did not preserve a clean reader")
        fault_receipt = {"expected_error": rejected, "configuration_unchanged": True,
                         "observation_restored_unchanged": path.read_bytes() == raw,
                         "method": "harness temporarily renamed the owned fixture around one configuration call",
                         "native_injection_verified": False}
    await call(session, "configure_reader", result["expected_reader"])
    reading = await call(session, "read_equipment", {"device_id": identity})
    expected = case["expected"]
    await call(session, "submit_diagnosis", {"device_id": identity, "status": expected["status"],
                "value": expected["value"], "unit": expected["unit"], "evidence_id": reading["evidence_id"]})
    record = {"device_id": identity, "task_kind": "equipment_interface_v1", "status": "finished",
              "submission": session.submission, "trace": {"tool_events": session.events}}
    job = {"record": record, "digest": hashlib.sha256(encoded(record)).hexdigest()}
    oracle_case = {"case_id": identity, "document": document,
                  **{key: result[key] for key in ("expected_reader", "expected_status", "expected_value", "expected_unit")}}
    checked = inspect_job(oracle_case, raw, job)
    require(checked["semantic_passed"] is True and path.read_bytes() == raw,
            "Actual reader evidence did not reproduce the expected fixture diagnosis")
    require(len(session.events) == (5 if fault_receipt else 4), "Unexpected input-check tool sequence")
    return {"case_id": identity, "epoch": epoch, "observation": document, "job": job,
            "reconstruction": result, "trace_check": checked, "fault_check": fault_receipt,
            "maximum_tool_result_bytes": max(len(encoded(event["result"])) for event in session.events),
            "model_calls": 0, "harness_selected_reader_and_answer": True,
            "native_execution_verified": False, "qualification_passed": False}


def audit_capture(inputs, output):
    _, cases = read_inputs(inputs)
    output = Path(output)
    report = json.loads((output / "report.json").read_bytes())
    require(report["schema"] == "organism-qualification-input-check-v1"
            and report["model_calls"] == report["baseline_runs"] == 0
            and all(report[key] is False for key in (
                "qualification_passed", "qualification_sealed", "native_execution_verified"))
            and report["harness_selected_reader_and_answer"] is True
            and report["failure"] is None, "Input-check scope changed")
    require(entry((Path(inputs) / "manifest.json").read_bytes()) == report["input_manifest"],
            "Input-check manifest binding changed")
    sources = {"check_qualification_inputs.py", "diagnostic_oracle.py", "qualification_inputs.py",
               "audit_equipment_diagnostics.py", "audit_application.py", "equipment_interfaces.py",
               "diagnostic_agent.py", "equipment_diagnostic_agent.py", "equipment.py", "capsule.py"}
    require(set(report["source_receipts"]) == sources, "Input-check source inventory differs")
    for name, receipt in report["source_receipts"].items():
        require(entry((output / "sources" / name).read_bytes()) == receipt
                == entry(Path(__file__).with_name(name).read_bytes()), "Input-check source changed")
    require(len(report["cases"]) == len(cases), "Input-check case inventory differs")
    for case, row in zip(cases, report["cases"]):
        identity = case["case_id"]
        require(row["case_id"] == identity, "Input-check case order differs")
        raw = (output / "cases" / (identity + ".json")).read_bytes()
        require(entry(raw) == row["receipt"], "Saved input-check record changed")
        saved = json.loads(raw)
        require(saved["case_id"] == identity and saved["model_calls"] == 0
                and saved["qualification_passed"] is False, "Saved input-check scope changed")
        observed = encoded(saved["observation"])
        require(saved["observation"] == materialize(case, saved["epoch"]), "Materialized observation differs")
        result = reconstruct(observed, case["oracle_plan"], now=saved["epoch"])
        require(expectation_matches(case, result) and saved["reconstruction"] == result,
                "Saved independent reconstruction differs")
        reads = [event["result"] for event in saved["job"]["record"]["trace"]["tool_events"]
                 if event["tool"] == "read_equipment" and event["result"].get("status") == "observed"]
        require(len(reads) == 1 and reads[0]["read_unix_seconds"] == saved["epoch"],
                "Saved reader clock differs from its materialized input clock")
        oracle_case = {"case_id": identity, "document": saved["observation"],
                      **{key: result[key] for key in ("expected_reader", "expected_status", "expected_value", "expected_unit")}}
        checked = inspect_job(oracle_case, observed, saved["job"])
        require(checked == saved["trace_check"] and checked["semantic_passed"] is True,
                "Saved actual reader trace differs")
        require(checked["expected_status"] == result["expected_status"]
                and checked["expected_value"] == result["expected_value"],
                "Trace expectation differs from the independent input oracle")
        events = saved["job"]["record"]["trace"]["tool_events"]
        expected_count = 5 if case["fault"] else 4
        require(len(events) == expected_count, "Interrupted reader call is missing or duplicated")
        if case["fault"]:
            require(events[1]["tool"] == "configure_reader" and events[1]["arguments"] == result["expected_reader"]
                    and events[1]["result"] == {"status": "invalid_observation_or_arguments", "error_type": "FileNotFoundError"}
                    and saved["fault_check"]["observation_restored_unchanged"] is True
                    and saved["fault_check"]["configuration_unchanged"] is True,
                    "Configuration fault was not exercised as declared")
        else:
            require(saved["fault_check"] is None, "Unexpected injected fault")
        require(saved["maximum_tool_result_bytes"] == max(len(encoded(event["result"])) for event in events)
                <= 8192, "Input-check result size differs")
    return {"audit_passed": True, "draft_cases": len(cases), "model_calls": 0, "baseline_runs": 0,
            "scope": "Input and saved reader-evidence reconstruction; no candidate qualification execution",
            "native_execution_verified": False, "qualification_sealed": False, "qualification_passed": False}


def check(inputs, output):
    _, cases = read_inputs(inputs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    sources = ("check_qualification_inputs.py", "diagnostic_oracle.py", "qualification_inputs.py",
               "audit_equipment_diagnostics.py", "audit_application.py", "equipment_interfaces.py",
               "diagnostic_agent.py", "equipment_diagnostic_agent.py", "equipment.py", "capsule.py")
    (output / "sources").mkdir()
    source_receipts = {}
    for name in sources:
        raw = Path(__file__).with_name(name).read_bytes()
        (output / "sources" / name).write_bytes(raw)
        source_receipts[name] = entry(raw)
    report = {"schema": "organism-qualification-input-check-v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "input_manifest": entry((Path(inputs) / "manifest.json").read_bytes()), "source_receipts": source_receipts,
              "cases": [], "failure": None, "model_calls": 0, "baseline_runs": 0,
              "harness_selected_reader_and_answer": True,
              "native_execution_verified": False, "qualification_sealed": False, "qualification_passed": False}
    write(output / "request.json", report)
    began = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="qualification-input-check-") as temporary:
            for case in cases:
                saved = asyncio.run(exercise(case, Path(temporary), REFERENCE_TIME + 7000000))
                path = output / "cases" / (case["case_id"] + ".json")
                write(path, saved)
                report["cases"].append({"case_id": case["case_id"], "receipt": entry(path.read_bytes()),
                                        "tool_calls": len(saved["job"]["record"]["trace"]["tool_events"]),
                                        "maximum_tool_result_bytes": saved["maximum_tool_result_bytes"]})
                write(output / "progress.json", report)
        require(all(entry(Path(__file__).with_name(name).read_bytes()) == record
                    for name, record in source_receipts.items()), "Input-check source changed during execution")
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    report["elapsed_seconds"] = round(time.monotonic() - began, 3)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    write(output / "report.json", report)
    audit = audit_capture(inputs, output) if report["failure"] is None else {
        "audit_passed": False, "failure": report["failure"], "qualification_passed": False}
    write(output / "audit.json", audit)
    return report, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, audit = check(args.inputs, args.output)
    print(json.dumps({"cases": len(report["cases"]), "failure": report["failure"],
                      "elapsed_seconds": report["elapsed_seconds"], "audit_passed": audit["audit_passed"],
                      "qualification_passed": False}))
    if report["failure"] or not audit["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
