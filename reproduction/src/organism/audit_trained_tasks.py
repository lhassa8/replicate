"""Reconstruct the public development task comparison without running an agent."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .audit_equipment_diagnostics import inspect_job, require, resolve


ARMS = ("stock-control", "trained-adapter")


def identity(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def with_clock(template, now):
    """Only published timestamps vary between otherwise matched fixtures."""
    item = deepcopy(template)
    for source in item["document"]["sources"]:
        for field in source["fields"]:
            if field["kind"] != "timestamp":
                continue
            selector = field["selector"]
            value = resolve(source, selector)
            require(type(value) in (int, float), "Template timestamp is not numeric")
            if source["format"] == "register_map":
                bank, address = selector.split(":")
                source["payload"][bank][str(int(address))] = now + value
            else:
                parts = [part.replace("~1", "/").replace("~0", "~") for part in selector[1:].split("/")]
                current = source["payload"]
                for part in parts[:-1]:
                    current = current[int(part)] if isinstance(current, list) else current[part]
                current[int(parts[-1]) if isinstance(current, list) else parts[-1]] = now + value
    return item


def evaluate(stored, raw, *, baseline=False):
    """Retain incomplete traces as audit failures, never fabricate completions."""
    job = stored["baseline_job"] if baseline else stored["job"]
    try:
        return {"audit_passed": True, **inspect_job(
            stored["case"], raw, job, live=not baseline,
            clock_window=None if baseline else stored["clock_window"])}
    except Exception as exc:
        trace = job["record"]["trace"]
        return {"audit_passed": False, "semantic_passed": False,
                "audit_failure": type(exc).__name__ + ": " + str(exc),
                "model_calls": len(trace.get("model_calls", [])),
                "tool_calls": len(trace.get("tool_events", [])), "qualification_passed": False}


def check_adapters(actual, path, scale):
    required = {"id": 0, "path": str(path), "scale": scale}
    require(len(actual) == 1 and all(actual[0].get(key) == value for key, value in required.items())
            and set(actual[0]) <= set(required) | {"task_name", "prompt_prefix"}
            and all(actual[0].get(key, "") == "" for key in ("task_name", "prompt_prefix")),
            "Loaded adapter identity, scale or task metadata differs")


def check_inventory(plan, summaries):
    expected = [item["case_id"] for item in plan["cases"]]
    require(len(expected) == 10 and len(set(expected)) == 10
            and [item["case_id"] for item in summaries] == expected,
            "Development case inventory is incomplete, reordered or duplicated")


def check_export_recipes(directory, plan, admission):
    """Bind a refinement comparison to its saved original and candidate recipes."""
    if plan.get("explicit_training_recipe") is None:
        return
    previous_path = directory / "previous-runtime-report.json"
    original_path = directory / "original-training-recipe.json"
    recipe_path = directory / "training-recipe.json"
    require(identity(previous_path) == plan["previous_runtime_binding"]
            and identity(previous_path)["sha256"]
            == "22a8809161fee3a68e6990774ca19586995b10d31851c812b2adca9d39857610",
            "Original training comparison binding changed")
    previous = json.loads(previous_path.read_text())["training_admission"]
    require(identity(original_path) == previous["recipe"]
            and identity(recipe_path) == plan["explicit_training_recipe"] == admission["recipe"],
            "Comparison recipe snapshot changed")
    require(json.loads(original_path.read_text())["base"]
            == json.loads(recipe_path.read_text())["base"]
            and previous["base_parameters"] == admission["base_parameters"],
            "Comparison recipes use different original bases")


def audit(directory, *, exported=False):
    directory = Path(directory).resolve()
    plan_path = directory / "plan.json"
    plan = json.loads(plan_path.read_text())
    report = json.loads((directory / "report.json").read_text())
    require(identity(plan_path) == report["plan"], "Pre-inference plan changed")
    require(report["failure"] is None and report["sources_unchanged"] is True
            and report["inputs_unchanged"] is True and report["qualification_passed"] is False,
            "Run failed or its source/input identities changed")
    for name, expected in plan["source_files"].items():
        require(identity(directory / "source-snapshot" / name) == expected, "Source snapshot changed")
    arms = ARMS
    if exported:
        from .export_task_inputs import ARMS as EXPORTED_ARMS, checked_pair, check_model_binding
        arms = EXPORTED_ARMS
        require(plan["schema"] == "organism-exported-development-tasks-plan-v1",
                "Expected an exported development comparison plan")
        admission = json.loads((directory / "training-admission.json").read_text())
        require(identity(directory / "training-admission.json") == plan["training_admission"],
                "Training admission changed")
        check_export_recipes(directory, plan, admission)
        pair = checked_pair(plan["export_pair"]["export_directory"], admission)
        require(pair == plan["export_pair"] and pair["models"] == plan["models"],
                "Exported model evidence differs from the pre-inference plan")
    require(set(report["arms"]) == set(arms), "Matched comparison arm is missing")
    results = {}
    for arm, scale in zip(arms, (0, 1)):
        record = json.loads((directory / arm / "report.json").read_text())
        require(record == report["arms"][arm] and record["failure"] is None, "Arm report differs or failed")
        require(record["shutdown"]["owned_server_stopped"] is True
                and record["shutdown"]["server_return_code"] == 0
                and record["socket_removed"] is True, "Application shutdown is incomplete")
        if exported:
            check_model_binding(plan, arm, record)
        else:
            check_adapters(record["loaded_adapters"], plan["adapter_path"], scale)
        require(record["final_adapters"] == record["loaded_adapters"], "Adapter changed during tasks")
        check_inventory(plan, record["cases"])
        cases = []
        for template, summary in zip(plan["cases"], record["cases"]):
            case_id = template["case_id"]
            stored = json.loads((directory / arm / (case_id + ".json")).read_text())
            profile = plan.get("tool_profile", "organism-raw-equipment-tools-v1")
            require(all(stored[key]["record"]["trace"].get(
                "tool_profile", "organism-raw-equipment-tools-v1") == profile
                for key in ("job", "baseline_job")),
                "Model and fixed policy must use the planned tool profile")
            raw = (directory / arm / "work-evidence/equipment-interfaces" / (case_id + ".json")).read_bytes()
            require(stored["case"] == with_clock(template, stored["fixture_epoch"]),
                    "Case differs from the predeclared input beyond its clock")
            require(stored["retry_preserved_identity"] is True and stored["input_unchanged"] is True,
                    "Request retry or read-only evidence is missing")
            model, fixed = evaluate(stored, raw), evaluate(stored, raw, baseline=True)
            require(model == stored["evaluation"] and fixed == stored["baseline_evaluation"],
                    "Saved evaluation differs from reconstructed evidence")
            require(model["semantic_passed"] == summary["model_passed"]
                    and fixed["semantic_passed"] == summary["baseline_passed"],
                    "Summary differs from reconstructed outcomes")
            trace = stored["job"]["record"]["trace"]
            require(trace["system_prompt"] == plan["system_prompt"], "Agent prompt changed")
            for call, request in zip(trace["model_calls"], trace["http_requests"]):
                require(call["tools"] == plan["tool_specs"], "Agent tools differ from the fixed plan")
                require("lora" not in request["body"], "Request overrides the arm's loaded adapter")
                require(request["body"]["model"] == record["model_alias"], "Request model alias differs")
            cases.append({"case_id": case_id, "model_passed": model["semantic_passed"],
                          "baseline_passed": fixed["semantic_passed"],
                          "trace_audit_passed": model["audit_passed"] and fixed["audit_passed"],
                          "model_calls": model["model_calls"], "tool_calls": model["tool_calls"],
                          "seconds": trace.get("seconds"), "failure": stored["job"]["record"]["failure"],
                          "audit_failure": model.get("audit_failure")})
        results[arm] = {
            "cases": cases, "model_successes": sum(item["model_passed"] for item in cases),
            "fixed_policy_successes": sum(item["baseline_passed"] for item in cases),
            "all_trace_audits_passed": all(item["trace_audit_passed"] for item in cases),
            "model_calls": sum(item["model_calls"] for item in cases),
            "tool_calls": sum(item["tool_calls"] for item in cases),
            "peak_sampled_server_rss_bytes": record["peak_sampled_server_rss_bytes"],
        }
    return {"schema": ("organism-exported-development-tasks-audit-v1" if exported
                       else "organism-trained-development-tasks-audit-v1"), "evidence_reconstructed": True,
            "plan": identity(plan_path), "run_report": identity(directory / "report.json"),
            "arms": results, "development_validation_only": True, "teacher_forced_prefixes": False,
            "real_strands_tools_executed": True, "native_linux_execution": False,
            "frozen_merged_export_comparison": exported, "qualification_passed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exported", action="store_true")
    args = parser.parse_args()
    result = audit(args.directory, exported=args.exported)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
