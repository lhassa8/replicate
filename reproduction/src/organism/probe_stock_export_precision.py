"""Diagnose a rejected stock Q4 export with matched local cache settings.

This uses three existing training-reference prompts, never qualification
inputs or equipment tools. It cannot admit a release or change export limits.
The caller must bound the complete owned process group while this runs.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil

from .audit_cpu_export import numerical_readback
from .export_math import compare_native_probabilities
from .export_serving import serving_rows
from .export_task_inputs import identity, require, runtime_inventory
from .gguf_export import Toolchain
from .probe_trained_tasks import PREVIOUS, PREVIOUS_REPORT_SHA256
from .safe_archive import safe_relative


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def inputs(root, export):
    audit_path = export / "result-audit.json"
    audit = json.loads(audit_path.read_text())
    require(audit["archive"] == identity(export / "outputs/results.tar.gz"), "Result archive changed")
    returned = export / "returned"
    for name, expected in audit["files"].items():
        path = returned / safe_relative(name)
        require(path.resolve().is_relative_to(returned) and identity(path) == expected,
                "Returned result changed")
    numerical = numerical_readback(returned)
    require(numerical == audit["numerical_readback"], "Numerical readback changed")
    reported = json.loads((returned / "workload/export/report.json").read_text())
    require(reported["passed"] is False
            and reported["numerical_failures"] == ["stock-q4: Q4 mean divergence"]
            and numerical["merge"]["passed"] is True
            and numerical["export"]["stock-q4"]["mean_kl_divergence"] > 0.10,
            "This diagnostic requires the isolated stock Q4 fidelity failure")
    model = returned / "workload/export/stock-q4.gguf"
    model_identity = identity(model)
    require(model_identity == {key: reported["models"]["stock-q4"][key] for key in ("bytes", "sha256")},
            "Returned stock model differs from the numerical export")
    previous = root / PREVIOUS / "report.json"
    require(identity(previous)["sha256"] == PREVIOUS_REPORT_SHA256, "Runtime binding changed")
    old = json.loads(previous.read_text())
    require(reported["admission"] == old["training_admission"], "Different training/export lineage")
    runtime = {name: expected for name, expected in old["tool_inputs"]["files"].items()
               if name.startswith("tools/llama.cpp/") and not name.endswith("llama-quantize")}
    links = runtime_inventory(root, runtime)
    binary = "tools/llama.cpp/b9850/llama-b9850/llama-server"
    tools = Toolchain(root, {"server": binary, "files": runtime,
                            "library_paths": ["tools/llama.cpp/b9850/llama-b9850"]})
    return reported, model, {
        "export_directory": str(export), "result_audit": identity(audit_path),
        "stock_model": {"path": str(model), **model_identity},
        "runtime_files": runtime, "runtime_links": links,
        "reference": identity(returned / "workload/merge/base-reference-logits.npy"),
    }, tools


async def run(output, export):
    import numpy as np

    require(platform.system() == "Darwin", "This diagnostic is explicitly a local Mac comparison")
    root = Path(__file__).resolve().parents[1]
    export, output = Path(export).resolve(), Path(output).resolve()
    require(not output.exists(), "Diagnostic output must be new")
    require(shutil.disk_usage(output.parent).free > 1024**3, "Insufficient evidence space")
    reported, model, bound, tools = inputs(root, export)
    output.mkdir()
    source_names = ("probe_stock_export_precision.py", "export_serving.py", "export_math.py",
                    "inference_server.py", "runtime_environment.py", "export_task_inputs.py")
    source_hashes = {name: identity(root / "organism" / name) for name in source_names}
    snapshot = output / "source-snapshot"
    snapshot.mkdir()
    for name in source_names:
        shutil.copyfile(root / "organism" / name, snapshot / name)
    plan = {
        "schema": "organism-stock-export-precision-plan-v1",
        "created_at": datetime.now(timezone.utc).isoformat(), **bound,
        "prompts": reported["prompts"], "cases": reported["development_reference_cases"],
        "source_files": source_hashes,
        "arms": ["default-cache", "f32-cache-no-flash"],
        "maximum_arm_seconds": 900, "maximum_request_seconds": 300,
        "scope": "Same rejected stock Q4 weights and three training reference prompts on the same Mac CPU",
        "qualification_passed": False,
    }
    require(len(plan["prompts"]) == 3
            and all(case["source_split"] == "train" for case in plan["cases"]),
            "Diagnostic includes an unexpected reference input")
    write(output / "plan.json", plan)
    reference = np.load(export / "returned/workload/merge/base-reference-logits.npy", allow_pickle=False)
    report = {"schema": "organism-stock-export-precision-v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "plan": identity(output / "plan.json"), "arms": {}, "failure": None,
              "native_linux_execution": False, "release_admitted": False, "qualification_passed": False}
    try:
        for label, precision in (("default-cache", False), ("f32-cache-no-flash", True)):
            values, served = await serving_rows(
                tools, output, model, label, plan["prompts"], reference.shape[1],
                reference_precision=precision, synthetic=False, trained=False,
                total_seconds=900, request_seconds=300)
            np.save(output / (label + "-log-probabilities.npy"), values, allow_pickle=False)
            report["arms"][label] = {
                "comparison": compare_native_probabilities(reference, values),
                "serving_report": identity(output / label / "report.json"),
                "shutdown": served["shutdown"], "precision": served["precision"],
            }
            print(json.dumps({"arm": label, "comparison": report["arms"][label]["comparison"]}), flush=True)
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        try:
            runtime_inventory(root, bound["runtime_files"], expected_links=bound["runtime_links"])
            require(identity(model) == {key: bound["stock_model"][key] for key in ("bytes", "sha256")},
                    "Stock weights changed during the probe")
            require(all(identity(root / "organism" / name) == expected
                        for name, expected in source_hashes.items()), "Probe source changed")
            report["inputs_unchanged"] = True
        except Exception as exc:
            report["inputs_unchanged"] = False
            report["failure"] = report["failure"] or str(exc)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args.output, args.export_directory))
    if result["failure"]:
        raise SystemExit(result["failure"])
