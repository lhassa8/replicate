"""Run ten development tasks with a verified merged Q4 pair and bound tools."""
import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil

from .audit_trained_tasks import audit
from .equipment_development import case, VARIATIONS
from .equipment_diagnostic_agent import SYSTEM_PROMPT
from .temperature_measurement import PROFILE
from .export_task_inputs import ARMS, checked_pair, identity, require, runtime_inventory
from .package_application import APPLICATION_MODULES
from .probe_trained_tasks import (
    LIMITS, PREVIOUS, PREVIOUS_REPORT_SHA256, registered_specs, run_arm, write,
)


def make_plan(root, output, admission_path, export_directory, *, training_recipe_path=None):
    admission = json.loads(admission_path.read_text())
    pair = checked_pair(export_directory, admission)
    previous = root / PREVIOUS / "report.json"
    require(identity(previous)["sha256"] == PREVIOUS_REPORT_SHA256, "Prior runtime binding changed")
    old = json.loads(previous.read_text())
    original_recipe = root / "organism/training-recipe.current-v4.admission-v2.draft.json"
    if admission != old["training_admission"]:
        require(training_recipe_path is not None
                and identity(original_recipe) == old["training_admission"]["recipe"]
                and identity(training_recipe_path) == admission["recipe"],
                "A new trained comparison requires its exact recipe and the original base binding")
        require(json.loads(Path(training_recipe_path).read_text())["base"]
                == json.loads(original_recipe.read_text())["base"]
                and admission["base_parameters"] == old["training_admission"]["base_parameters"],
                "The exported comparison changed the original base")
    if training_recipe_path is not None:
        require(identity(training_recipe_path) == admission["recipe"],
                "Explicit training recipe differs from the admitted run")
        shutil.copyfile(training_recipe_path, output / "training-recipe.json")
        shutil.copyfile(original_recipe, output / "original-training-recipe.json")
        shutil.copyfile(previous, output / "previous-runtime-report.json")
    runtime_inputs = {name: expected for name, expected in old["tool_inputs"]["files"].items()
                      if name.startswith("tools/llama.cpp/") and not name.endswith("llama-quantize")}
    runtime_links = runtime_inventory(root, runtime_inputs)
    binary = root / "tools/llama.cpp/b9850/llama-b9850/llama-server"
    require(str(binary.relative_to(root)) in runtime_inputs, "Inference binary is not bound")
    shutil.copyfile(admission_path, output / "training-admission.json")
    plan = {
        "schema": "organism-exported-development-tasks-plan-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Ten unchanged public development validation cases with the actual application and Strands tools",
        "cases": [case(profile, variation, now=0) for profile in ("g", "h") for variation in VARIATIONS],
        "system_prompt": SYSTEM_PROMPT, "tool_specs": registered_specs(), "tool_profile": PROFILE,
        "arms": list(ARMS),
        "model_representation": "merged-q4", "models": pair["models"], "export_pair": pair,
        "training_admission": identity(output / "training-admission.json"),
        "previous_runtime_binding": identity(previous),
        "explicit_training_recipe": None if training_recipe_path is None else identity(training_recipe_path),
        "binary_path": str(binary), "runtime_inputs": runtime_inputs, "runtime_links": runtime_links,
        "limits": LIMITS, "source_files": {}, "total_stored_parameters": pair["total_stored_parameters"],
        "versions": {name: version(name) for name in ("strands-agents", "httpx", "cryptography", "jsonschema")},
        "baseline_status": "Existing unchanged development fixed policy; not a blind Q5 comparison",
        "matching": "Same cases, prompts, tools and settings; stock then trained; only fixture clocks change",
        "development_validation_only": True, "qualification_passed": False,
    }
    require(plan["versions"]["strands-agents"] == "1.57.1", "Unexpected Strands version")
    require(all(item["split"] == "development_validation" for item in plan["cases"]),
            "Comparison includes non-validation cases")
    names = {name + ".py" for name in APPLICATION_MODULES} | {
        "probe_trained_tasks.py", "audit_trained_tasks.py", "probe_equipment_interfaces.py",
        "equipment_development.py", "equipment_baseline.py", "package_application.py",
        "probe_exported_tasks.py", "export_task_inputs.py", "audit_cpu_export.py", "export_math.py",
    }
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name in sorted(names):
        source = root / "organism" / name
        shutil.copyfile(source, snapshot / name)
        plan["source_files"][name] = identity(source)
    write(output / "plan.json", plan)
    return plan


async def run(output, admission_path, export_directory, *, training_recipe_path=None):
    require(platform.system() == "Darwin", "This is the local Mac development comparison")
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    require(shutil.disk_usage(output).free > 2 * 1024**3, "Insufficient evidence space")
    plan = make_plan(root, output, admission_path.resolve(), export_directory.resolve(),
                     training_recipe_path=training_recipe_path)
    report = {"schema": "organism-exported-development-tasks-v1", "plan": identity(output / "plan.json"),
              "started_at": datetime.now(timezone.utc).isoformat(), "arms": {}, "failure": None,
              "qualification_passed": False}
    try:
        for arm in ARMS:
            record = await run_arm(root, output, arm, plan, model_override=plan["models"][arm])
            report["arms"][arm] = record
            require(record["failure"] is None, "Arm failed: " + str(record["failure"]))
            require(record["shutdown"]["owned_server_stopped"] is True, "Owned server is still running")
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        report["sources_unchanged"] = all(identity(root / "organism" / name) == expected
                                          for name, expected in plan["source_files"].items())
        try:
            runtime_inventory(root, plan["runtime_inputs"], expected_links=plan["runtime_links"])
            report["inputs_unchanged"] = all(
                identity(Path(entry["path"])) == {key: entry[key] for key in ("bytes", "sha256")}
                for entry in plan["models"].values())
            if training_recipe_path is not None:
                report["inputs_unchanged"] = (
                    report["inputs_unchanged"]
                    and identity(training_recipe_path) == plan["explicit_training_recipe"])
        except (OSError, ValueError) as exc:
            report["inputs_unchanged"] = False
            report["failure"] = report["failure"] or "Input readback failed: " + str(exc)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        write(output / "report.json", report)
    try:
        checked = audit(output, exported=True)
    except Exception as exc:
        checked = {"evidence_reconstructed": False, "failure": type(exc).__name__ + ": " + str(exc),
                   "qualification_passed": False}
    write(output / "audit.json", checked)
    print(json.dumps({"report": str(output / "report.json"), "failure": report["failure"],
                      "evidence_reconstructed": checked["evidence_reconstructed"]}), flush=True)
    return report, checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--export-directory", type=Path, required=True)
    parser.add_argument("--training-recipe", type=Path, help="Required to compare a newly admitted training run")
    args = parser.parse_args()
    report, checked = asyncio.run(run(args.output, args.admission, args.export_directory,
                                     training_recipe_path=args.training_recipe))
    if report["failure"] or not checked["evidence_reconstructed"]:
        raise SystemExit(1)
