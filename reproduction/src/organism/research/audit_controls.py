"""Reconstruct the separate cohosted comparison without widening original gates."""

import argparse
import base64
from datetime import datetime
import json
from pathlib import Path
import statistics
import types

from organism.audit_trained_tasks import evaluate, with_clock
from organism.capsule import verify_manifest
from organism.enrollment import Enrollment
from organism.native_preflight import audit_lineage_transport as original
from organism.native_preflight.audit_startup import Evidence, require
from organism.research.control_policy import verify_fixed


class ComparedEvidence(Evidence):
    """Map original auditor paths onto this experiment's explicit public layout."""

    def __init__(self, root, individual):
        super().__init__(root)
        self.work_prefix = "seed-evidence/data/work_root/" + individual + "/"

    def path(self, name):
        if name.startswith(self.work_prefix):
            name = "work/" + name[len(self.work_prefix):]
        elif name.startswith("seed-evidence/control/state_directory/"):
            name = "receiver-state/" + name[len("seed-evidence/control/state_directory/"):]
        return super().path(name)


def fixed_decision(job, *, parent, child, catalog, child_history):
    # Demonstrate that the unchanged model predicate still rejects this profile.
    try:
        original.decision(job, parent=parent, child=child, catalog=catalog,
                          child_history=child_history)
    except ValueError:
        pass
    else:
        raise ValueError("The original model predicate unexpectedly admitted fixed policy")
    choice = verify_fixed(job, peer_id=child["binding"]["device_id"],
                          envelope={"release_id": catalog["release_id"],
                                    "body": {"files": catalog["files"], "platforms": catalog["platforms"]}})
    observation = job["record"]["trace"]["tool_events"][0]["result"]
    anchor = original.anchored(child_history, observation["transport_receipt"])
    public = {key: observation[key] for key in
              ("device_id", "platform", "owner_mode", "available", "revision")}
    public.update(policy_id=child["policy_id"], observation_valid_seconds=30)
    require(anchor["phase"] == "completed" and anchor["operation"] == "observe"
            and anchor["peer_id"] == parent["binding"]["device_id"]
            and anchor["detail"]["response_sha256"] == original.sha(original.canonical(public)),
            "Fixed choice lacks its actual authenticated receiver observation")
    require(choice["action"] == "establish", "Fixed policy did not establish this child")
    return {"model_calls": 0, "tool_calls": 2, "choice": choice,
            "decision_sha256": job["digest"], "receiver_inspection_verified": True,
            "original_model_predicate_rejected": True}


def fixed_link(*args, **kwargs):
    # A separate function with only the decision checker replaced. The original
    # model-only auditor and every caller of it remain unchanged.
    namespace = dict(original.link.__globals__, decision=fixed_decision)
    checker = types.FunctionType(original.link.__code__, namespace, "fixed_profile_link")
    result = checker(*args, **kwargs)
    result.pop("model_choice_and_peer_history_reconstructed")
    result["fixed_choice_and_peer_history_reconstructed"] = True
    return result


def arm_audit(directory, cases):
    report = json.loads((directory / "report.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    contexts = {slot: json.loads((directory / slot / "receiver-context.json").read_text())
                for slot in ("seed", "child", "grandchild")}
    policy = Enrollment((directory / "enrollment.json").read_bytes(),
                        base64.b64decode(contexts["seed"]["owner_public"]))
    verify_manifest((directory / "manifest.json").read_bytes(), policy.release_keys, require_trained=False)
    require(report["release_id"] == manifest["release_id"], "Reported release differs")
    result = {"arm": report["arm"], "reported_failure": report["failure"],
              "tasks": [], "births": [], "validation_failure": None}
    fixed = report["arm"] == "fixed-policy"
    public, evidence = {}, {}
    try:
        for slot in ("seed", "child", "grandchild"):
            node = report["nodes"].get(slot)
            require(node and node.get("readiness_passed"), "No useful individual: " + slot)
            binding = node["binding"]
            snapshot = json.loads((directory / slot / "receiver-snapshot.json").read_text())
            public[slot] = {"binding": binding, "policy_id": policy.policy_id,
                            "birth": snapshot["current_birth"],
                            "readiness_records": {"intent": {"record": {"boot_id": snapshot["boot_id"]}}}}
            evidence[slot] = ComparedEvidence(directory / slot, binding["individual_id"])
            receipt = json.loads((directory / slot / "work" /
                                  ("application-" + binding["start_token"] + ".json")).read_text())
            require(receipt["native"] is True and receipt["shutdown"]["owned_server_stopped"],
                    "Native application or owned server shutdown lacks evidence")
            require(all(receipt[k] == binding[k] for k in binding), "Application binding differs")
            if fixed:
                require(receipt["inference"]["model_started"] is False
                        and receipt["inference"]["pid"] is None,
                        "Deterministic application started inference")
            else:
                require(receipt["inference"]["native_execution"] is True
                        and receipt["inference"]["alias"] == "organism-" + binding["start_token"],
                        "Model is not the individual's native incarnation")
            require(int(node["process"]["MemoryMax"]) == 8 * 1024**3
                    and int(node["stop"]["MainPID"]) == 0,
                    "Declared application memory ceiling or stopped process differs")
            for index, case in enumerate(cases):
                name = "readiness" if index == 0 else f"task-{index:02d}"
                stored = json.loads((directory / slot / (name + ".json")).read_text())
                require(stored["case"] == with_clock(case, stored["fixture_epoch"]),
                        "Native diagnostic input differs")
                raw = original.canonical(stored["case"]["document"])
                checked = evaluate(stored, raw, baseline=fixed)
                require(checked == stored["evaluation"], "Native diagnostic score does not reconstruct")
                require((directory / slot / "work/equipment-interfaces" /
                         (case["case_id"] + ".json")).read_bytes() == raw, "Retained native input changed")
                if not fixed:
                    require(all(r["body"]["model"] == "organism-" + binding["start_token"]
                                for r in stored["job"]["record"]["trace"]["http_requests"]),
                            "Diagnostic used another model incarnation")
                passed = bool(checked["audit_passed"] and checked["semantic_passed"]
                              and checked["expected_status"] == case["expected_status"]
                              and checked["expected_value"] == case["expected_value"]
                              and stored["input_unchanged"] and stored["seconds"] <= 180)
                require(passed == stored["passed"], "Native saved pass differs")
                result["tasks"].append({"slot": slot, "case_id": case["case_id"], "passed": passed,
                                        "seconds": stored["seconds"],
                                        "model_calls": checked["model_calls"], "tool_calls": checked["tool_calls"]})
        catalog = {"release_id": manifest["release_id"], "files": manifest["body"]["files"],
                   "platforms": manifest["body"]["platforms"]}
        for parent, child in (("seed", "child"), ("child", "grandchild")):
            enabled = next(e["at"] for e in report["events"] if e["operation"] == "enabled" and e["slot"] == child)
            checked = (fixed_link if fixed else original.link)(
                evidence[parent], public[parent], evidence[child], public[child], catalog,
                child_enabled_at=enabled)
            result["births"].append(checked)
        stopped = next(e["at"] for e in report["events"]
                       if e["operation"] == "retired_and_processes_absent" and e["slot"] == "seed")
        enabled = next(e["at"] for e in report["events"] if e["operation"] == "enabled" and e["slot"] == "grandchild")
        require(datetime.fromisoformat(stopped) < datetime.fromisoformat(enabled),
                "Grandchild became available before seed process and receiver removal")
    except (ValueError, KeyError, OSError, StopIteration) as exc:
        result["validation_failure"] = type(exc).__name__ + ": " + str(exc)
    result["successful_tasks"] = sum(row["passed"] for row in result["tasks"])
    result["planned_tasks"] = 3 * len(cases)
    result["recorded_diagnostic_cases"] = sum(
        (directory / slot / ("readiness.json" if index == 0 else f"task-{index:02d}.json")).is_file()
        for slot in ("seed", "child", "grandchild") for index in range(len(cases)))
    result["unrecorded_planned_cases"] = (
        result["planned_tasks"] - result["recorded_diagnostic_cases"])
    result["reconstructed_diagnostic_cases"] = len(result["tasks"])
    result["reconstructed_failed_tasks"] = sum(not row["passed"] for row in result["tasks"])
    result["complete_lineage_reconstructed"] = (
        not result["reported_failure"] and not result["validation_failure"]
        and len(result["births"]) == 2 and result["successful_tasks"] == result["planned_tasks"])
    if result["tasks"]:
        result["median_case_seconds"] = statistics.median(r["seconds"] for r in result["tasks"])
        result["diagnostic_model_calls"] = sum(r["model_calls"] for r in result["tasks"])
    samples = json.loads((directory / "resource-samples.json").read_text())
    peaks = [int(row["MemoryPeak"]) for row in samples if str(row.get("MemoryPeak", "")).isdigit()]
    result["largest_reported_application_memory_peak_bytes"] = max(peaks, default=None)
    result["separate_machine_independence_claimed"] = False
    return result


def audit(source, cases_path, output):
    cases = json.loads(cases_path.read_text())["cases"]
    host = json.loads((source / "host-report.json").read_text())
    arms = []
    for name in ("fixed-policy", "stock-control", "trained-merged"):
        directory = source / name
        if not (directory / "report.json").exists():
            arms.append({"arm": name, "not_executed": True, "complete_lineage_reconstructed": False})
        else:
            arms.append(arm_audit(directory, cases))
    result = {"host_failure": host["failure"], "inputs_verified_by_worker": host["inputs_verified"],
              "arms": arms, "cohosted_native_comparison": True, "qualification_passed": False}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.cases, args.output), indent=2))
