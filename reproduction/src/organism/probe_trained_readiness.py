"""Exercise the exact carried checker on three cold Mac starts of an admitted release."""

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import uuid

from .audit_readiness import inspect as audit_smoke
from .export_task_inputs import identity, require, runtime_inventory
from .inference_server import RuntimeConfig
from .native_preflight.trained_inputs import prepare, validate_catalog
from .package_application import collect
from .package_self_test import package
from .probe_first_lineage_tasks import audit as audit_temperature
from .probe_readiness import run as run_smoke
from .probe_trained_tasks import write


VALUES_C = (9, 39, 41)


def challenges(release_id):
    return [
        {"value_c": value, "challenge_id": format(value + 10, "08x")
         + hashlib.sha256(f"readiness-v5/{release_id}/{index}".encode()).hexdigest()[:24]}
        for index, value in enumerate(VALUES_C, 1)
    ]


def check_packaged_application(root, artifact, body):
    """Read the signed archive itself, not its unsigned builder receipt."""
    expected, _ = collect(root / "organism")
    archive = artifact / "capsule" / body["agent"]
    require(identity(archive) == body["files"][body["agent"]], "Signed application archive changed")
    seen = set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            require(member.name in expected and member.name not in seen and member.isfile()
                    and member.size == len(expected[member.name]), "Unexpected packaged application source")
            with stream.extractfile(member) as source:
                require(source.read(member.size + 1) == expected[member.name],
                        "Carried application differs from the application under test")
            seen.add(member.name)
    require(seen == set(expected), "Carried application source inventory is incomplete")


def aggregate(plan, rows):
    require(plan["cases"] == challenges(plan["runtime"]["release_id"]),
            "The predeclared public startup challenges changed")
    require(len(rows) == 3 and [row["challenge_id"] for row in rows]
            == [case["challenge_id"] for case in plan["cases"]], "Startup trials are incomplete or repeated")
    require(len({row["process_token"] for row in rows}) == 3,
            "Startup trials reused an application incarnation")
    return {
        "schema": "organism-trained-cold-smoke-audit-v1",
        "cold_starts": 3, "passed": all(row["passed"] and row["audit_failure"] is None for row in rows),
        "cases": rows, "native_execution_verified": False, "native_cold_starts_verified": 0,
        "qualification_passed": False,
    }


def audit(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    require(report["plan"] == identity(directory / "plan.json") and report["failure"] is None
            and "completed_at" in report, "Cold smoke run is incomplete or its plan changed")
    require(identity(directory / "candidate/receipt.json") == plan["candidate_binding"]
            and identity(directory / "component/receipt.json") == plan["component_receipt"],
            "Cold smoke candidate or checker component changed")
    rows = []
    for index, case in enumerate(plan["cases"], 1):
        child = directory / f"cold-{index:02d}"
        challenge = json.loads((child / "challenge.json").read_text())
        require(challenge["challenge_id"] == case["challenge_id"], "Cold smoke challenge changed")
        result = audit_smoke(child, expected_runtime=plan["runtime"])
        require(result == json.loads((child / "independent-audit.json").read_text()),
                "Cold smoke audit differs from independent reconstruction")
        rows.append({**case, "process_token": challenge["process_token"],
                     "passed": result["independent_diagnostic_passed"], "audit_failure": None})
    require(rows == report["cases"], "Cold smoke summary differs from the saved traces")
    return aggregate(plan, rows)


async def run(output, artifact, admission, export, recipe, temperature, frozen, cases_sha256):
    root = Path(__file__).resolve().parents[1]
    checked = audit_temperature(temperature, frozen, cases_sha256)
    require(checked["temperature_model_gate_passed"], "Complete the twenty-case temperature gate first")
    temperature_plan = json.loads((temperature / "batch-01/plan.json").read_text())
    require(temperature_plan["training_admission"] == identity(admission)
            and temperature_plan["explicit_training_recipe"] == identity(recipe),
            "Startup and temperature preflights use different training")
    output.mkdir(parents=True, exist_ok=False)
    binding = prepare(root, output / "candidate", artifact=artifact, export_directory=export,
                      admission_path=admission, training_recipe=recipe)
    require(binding["export_pair"] == temperature_plan["export_pair"],
            "Startup and temperature preflights use different exports")
    catalog = validate_catalog(json.loads((output / "candidate/trained-catalog.json").read_text()))
    body = json.loads(base64.b64decode(catalog["manifest_base64"]))["body"]
    check_packaged_application(root, artifact, body)
    component = output / "component"
    component.mkdir()
    packed = package(root / "organism/self_test/readiness.py", component / "self-test.tar.gz")
    require(identity(component / "self-test.tar.gz") == body["files"][body["self_test"]]
            and (component / "self-test.tar.gz").read_bytes()
            == (artifact / "capsule" / body["self_test"]).read_bytes(),
            "Startup checker is not the exact signed carried component")
    write(component / "receipt.json", packed)
    runtime_inventory(root, temperature_plan["runtime_inputs"], expected_links=temperature_plan["runtime_links"])
    binary = Path(temperature_plan["binary_path"])
    model_path = artifact / "capsule" / body["model"]["path"]
    runtime = {
        "model_path": str(model_path), "model_sha256": body["files"][body["model"]["path"]]["sha256"],
        "model_bytes": body["files"][body["model"]["path"]]["bytes"],
        "runtime_path": str(binary), "runtime_sha256": identity(binary)["sha256"],
        "release_id": catalog["release_id"], "trained_for_organism": True,
    }
    plan = {
        "schema": "organism-trained-cold-smoke-plan-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime, "cases": challenges(catalog["release_id"]),
        "candidate_binding": identity(output / "candidate/receipt.json"),
        "temperature_audit": identity(temperature / "audit.json"),
        "component_receipt": identity(component / "receipt.json"), "artifact": str(artifact),
        "native_execution_verified": False, "qualification_passed": False,
    }
    write(output / "plan.json", plan)
    report = {
        "schema": "organism-trained-cold-smoke-run-v1", "plan": identity(output / "plan.json"),
        "started_at": datetime.now(timezone.utc).isoformat(), "cases": [], "failure": None,
        "native_execution_verified": False, "qualification_passed": False,
    }
    write(output / "report.json", report)
    try:
        for index, case in enumerate(plan["cases"], 1):
            token = uuid.uuid4().hex
            config = RuntimeConfig(
                model_path, runtime["model_sha256"], runtime["model_bytes"], binary,
                runtime["runtime_sha256"], token,
                hashlib.sha256(f"mac-smoke-device/{catalog['release_id']}/{index}".encode()).hexdigest(),
                uuid.uuid4().hex, catalog["release_id"], True, False)
            directory = output / f"cold-{index:02d}"
            result = await run_smoke(root, directory, component, runtime_config=config,
                                     challenge_id=case["challenge_id"])
            row = {**case, "process_token": token, "passed": False, "audit_failure": None}
            try:
                audited = audit_smoke(directory, expected_runtime=runtime)
                write(directory / "independent-audit.json", audited)
                row["passed"] = audited["independent_diagnostic_passed"]
            except Exception as exc:
                row["audit_failure"] = type(exc).__name__ + ": " + str(exc)
            report["cases"].append(row)
            write(output / "report.json", report)
            print(json.dumps({"cold_start": index, **row}), flush=True)
            require(result.get("shutdown", {}).get("owned_server_stopped") is True
                    and result["shutdown"].get("server_return_code") == 0
                    and result.get("socket_removed") is True and not result.get("shutdown_failure"),
                    "The preceding smoke model did not stop cleanly")
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        write(output / "report.json", report)
        write(output / "audit.json", audit(output))
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report.setdefault("completed_at", datetime.now(timezone.utc).isoformat())
        write(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "artifact", "admission", "export-directory", "training-recipe",
                 "temperature-evaluation", "frozen-cases"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cases-sha256", required=True)
    args = parser.parse_args()
    asyncio.run(run(
        args.output.resolve(), args.artifact.resolve(), args.admission.resolve(),
        args.export_directory.resolve(), args.training_recipe.resolve(),
        args.temperature_evaluation.resolve(), args.frozen_cases.resolve(), args.cases_sha256,
    ))
