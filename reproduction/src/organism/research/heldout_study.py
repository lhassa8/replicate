"""Freeze, execute, and reconstruct the post-development diagnostic experiment."""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import tempfile
import tarfile
import time
import uuid

from organism.application import AgentApplication, local_request
from organism.audit_trained_tasks import evaluate, with_clock
from organism.capsule import canonical
from organism.equipment_baseline import solve
from organism.equipment_diagnostic_agent import SYSTEM_PROMPT, SPECS
from organism.equipment_interfaces import InterfaceDirectory, InterfaceSession, validate_document
from organism.inference_server import RuntimeConfig, digest
from organism.probe_equipment_interfaces import baseline_job
from organism.probe_trained_tasks import LIMITS, monitor
from organism.research.heldout_cases import build_cases


ROOT = Path(__file__).resolve().parents[2]
PRIOR = "organism/evidence/first-lineage-temperature-evaluation-20261016-05/batch-01/plan.json"
ARMS = ("stock-control", "trained-merged")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identity(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def require(value, message):
    if not value:
        raise ValueError(message)


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    old = json.loads((ROOT / PRIOR).read_text())
    require(SYSTEM_PROMPT == old["system_prompt"], "The final diagnostic prompt changed")
    capsule = ROOT / "organism/evidence/first-lineage-native-execution-20261017-06/trained-capsule/capsule"
    manifest = json.loads((capsule / ".manifest.json").read_text())
    agent = capsule / manifest["body"]["agent"]
    require(identity(agent) == manifest["body"]["files"][manifest["body"]["agent"]],
            "The completed native application archive changed")
    with tarfile.open(agent, "r:gz") as archive:
        native_sources = {Path(m.name).name: archive.extractfile(m).read()
                          for m in archive.getmembers() if m.isfile() and m.name.endswith(".py")}
    native_changes = {}
    for name, expected in old["source_files"].items():
        current = identity(ROOT / "organism" / name)
        if current != expected:
            require(name in native_sources and (ROOT / "organism" / name).read_bytes() == native_sources[name],
                    "Source differs from both the final comparison and native release: " + name)
            native_changes[name] = {"development_comparison": expected, "native_release": current}
    for entry in old["models"].values():
        require(identity(entry["path"]) == {key: entry[key] for key in ("bytes", "sha256")},
                "The completed compact model changed")
    for name, expected in old["runtime_inputs"].items():
        require(identity(ROOT / name) == expected, "The local inference runtime changed")
    source_files = {}
    paths = list((ROOT / "organism").glob("*.py"))
    paths += list((ROOT / "organism/native_preflight").glob("*.py"))
    paths += list((ROOT / "organism/research").glob("*.py"))
    for source in sorted(paths):
        relative = source.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_files[str(relative)] = identity(source)
    freeze = {
        "schema": "replicate-paper-agent-freeze-v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "prior_plan": {"path": PRIOR, **identity(ROOT / PRIOR)},
        "native_release_id": manifest["release_id"],
        "native_agent_archive": identity(agent),
        "native_changes_since_development_comparison": native_changes,
        "models": old["models"], "binary_path": old["binary_path"],
        "runtime_inputs": old["runtime_inputs"], "source_files": source_files,
        "system_prompt": SYSTEM_PROMPT, "tool_specs": SPECS, "limits": LIMITS,
        "versions": {name: version(name) for name in ("strands-agents", "httpx", "cryptography", "jsonschema")},
        "host": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
        "training_or_policy_changes_permitted": False,
        "scope": "One-shot author-designed synthetic holdout; separate from Q1-Q10 and native lineages",
        "native_linux_execution": False, "external_replication": False,
    }
    write(output / "agent-freeze.json", freeze)
    # No outcomes have been computed when this input set is generated and saved.
    cases = build_cases()
    require(len(cases) == 40 and len({row["case_id"] for row in cases}) == 40, "Holdout inventory differs")
    for item in cases:
        validate_document(item["document"], item["case_id"])
    write(output / "cases.json", {"schema": "replicate-paper-holdout-v1", "cases": cases})
    write(output / "protocol.json", {
        "schema": "replicate-paper-holdout-protocol-v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "agent_freeze": identity(output / "agent-freeze.json"),
        "cases": identity(output / "cases.json"),
        "blocks": [{"block": i + 1, "case_ids": [c["case_id"] for c in cases[i * 10:(i + 1) * 10]],
                    "arm_order": list(ARMS if i % 2 == 0 else reversed(ARMS))} for i in range(4)],
        "baseline": "Existing unchanged equipment_baseline.solve; 40 separate executions, zero model calls",
        "primary_endpoint": "Correct reader, exact declared condition, diagnosis and value within 180 seconds",
        "missing_or_failed_run": "Count as failure; retain record; no selective rerun",
        "statistical_unit": "Four designed descriptor families; case counts are descriptive, not iid reliability estimates",
    })
    return freeze


def fixed_case(directory, template):
    epoch = time.time()
    item = with_clock(template, epoch)
    raw = canonical(item["document"])
    (directory / (item["case_id"] + ".json")).write_bytes(raw)
    session = InterfaceSession(InterfaceDirectory(directory), item["case_id"], canonical_temperature=True)
    began, failure = time.perf_counter(), None
    try:
        solve(session, item["case_id"])
    except Exception as exc:
        failure = type(exc).__name__ + ": " + str(exc)
    elapsed = time.perf_counter() - began
    job = baseline_job(session)
    if failure:
        job["record"]["status"] = "failed"
        job["record"]["failure"] = failure
        job["digest"] = __import__("hashlib").sha256(canonical(job["record"])).hexdigest()
    stored = {
        "case": item, "fixture_epoch": epoch, "clock_window": [epoch - 1, time.time() + 1],
        "baseline_job": job, "baseline_seconds": elapsed, "failure": failure,
    }
    stored["evaluation"] = evaluate(stored, raw, baseline=True)
    return stored


async def model_block(output, block, arm, templates, freeze):
    directory = output / f"block-{block:02d}" / arm
    directory.mkdir(parents=True, exist_ok=False)
    model = freeze["models"][arm]
    config = RuntimeConfig(
        Path(model["path"]), model["sha256"], model["bytes"], Path(freeze["binary_path"]),
        digest(freeze["binary_path"]), uuid.uuid4().hex, "d" * 64, uuid.uuid4().hex,
        "e" * 64, arm == "trained-merged", False,
    )
    record = {"arm": arm, "block": block, "cases": [], "failure": None, "rss_samples": [],
              "started_at": datetime.now(timezone.utc).isoformat(), "qualification_passed": False}
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rep-holdout-", dir="/private/tmp") as temporary:
        work = Path(temporary)
        app = AgentApplication(config, work)
        watcher = None
        try:
            async with asyncio.timeout(LIMITS["arm_seconds"]):
                watcher = asyncio.create_task(monitor(app, directory, work, record))
                record["initial_health"] = await app.start()
                record["model_alias"] = app.inference.alias
                record["inference"] = app.receipt["inference"]
                record["loaded_adapters"] = await app.inference.json_get("/lora-adapters")
                require(record["loaded_adapters"] == [], "Unexpected additional adapter")
                for template in templates:
                    require(not app.inference.failure, app.inference.failure)
                    epoch = time.time()
                    item = with_clock(template, epoch)
                    raw = canonical(item["document"])
                    source = work / "equipment-interfaces" / (item["case_id"] + ".json")
                    source.write_bytes(raw)
                    request_id, failure = uuid.uuid4().hex, None
                    request = {"operation": "diagnose", "request_id": request_id, "device_id": item["case_id"]}
                    started = time.monotonic()
                    try:
                        first = await local_request(app.socket, request)
                        retry = await local_request(app.socket, request)
                        require(first["request_id"] == retry["request_id"] == request_id,
                                "Request retry changed identity")
                        async with asyncio.timeout(LIMITS["case_seconds"]):
                            while True:
                                state = await local_request(app.socket, {"operation": "status", "request_id": request_id})
                                if state["status"] != "running":
                                    break
                                await asyncio.sleep(0.25)
                    except Exception as exc:
                        failure = type(exc).__name__ + ": " + str(exc)
                    elapsed = time.monotonic() - started
                    job = app.store.get(request_id)
                    stored = {
                        "case": item, "fixture_epoch": epoch,
                        "clock_window": [epoch - 1, time.time() + 1],
                        "request_id": request_id, "job": job, "seconds": elapsed,
                        "input_unchanged": source.read_bytes() == raw, "failure": failure,
                    }
                    stored["evaluation"] = evaluate(stored, raw)
                    checked = stored["evaluation"]
                    stored["passed"] = bool(
                        checked["semantic_passed"] and checked["audit_passed"] and not failure
                        and stored["input_unchanged"] and elapsed <= LIMITS["case_seconds"]
                        and checked["expected_status"] == template["expected_status"]
                        and checked["expected_value"] == template["expected_value"])
                    write(directory / (item["case_id"] + ".json"), stored)
                    summary = {"case_id": item["case_id"], "family": item["profile"],
                               "condition": item["variation"], "passed": stored["passed"],
                               "seconds": elapsed, "model_calls": checked["model_calls"],
                               "tool_calls": checked["tool_calls"], "failure": failure or job["record"]["failure"]}
                    record["cases"].append(summary)
                    write(directory / "progress.json", record)
                    print(json.dumps({"arm": arm, "block": block, **summary}), flush=True)
                    if failure:
                        raise RuntimeError("Request infrastructure failed; preserve remaining cases as unexecuted")
                record["final_health"] = app.health()
        except BaseException as exc:
            record["failure"] = type(exc).__name__ + ": " + str(exc)
        finally:
            if watcher:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            await app.close()
            record["shutdown"] = app.receipt.get("shutdown")
            shutil.copytree(work, directory / "work-evidence", ignore=shutil.ignore_patterns("*.sock"))
            record["peak_sampled_server_rss_bytes"] = max((r["rss_bytes"] for r in record["rss_samples"]), default=None)
            record["elapsed_seconds"] = time.monotonic() - began
            record["completed_at"] = datetime.now(timezone.utc).isoformat()
            write(directory / "report.json", record)
    return record


def verify_freeze(output, *, current_sources=False):
    freeze = json.loads((output / "agent-freeze.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    require(identity(output / "agent-freeze.json") == protocol["agent_freeze"]
            and identity(output / "cases.json") == protocol["cases"], "Frozen experiment inputs changed")
    for name, expected in freeze["source_files"].items():
        require(identity(output / "source" / name) == expected, "Snapshot changed: " + name)
        if current_sources:
            require(identity(ROOT / name) == expected, "Runtime source changed: " + name)
    return freeze, protocol


async def run(output):
    freeze, protocol = verify_freeze(output, current_sources=True)
    cases = json.loads((output / "cases.json").read_text())["cases"]
    fixed = output / "fixed-policy"
    fixed.mkdir()
    with tempfile.TemporaryDirectory(prefix="rep-fixed-", dir="/private/tmp") as temporary:
        for template in cases:
            stored = fixed_case(Path(temporary), template)
            write(fixed / (template["case_id"] + ".json"), stored)
    for block in protocol["blocks"]:
        templates = [row for row in cases if row["case_id"] in block["case_ids"]]
        for arm in block["arm_order"]:
            await model_block(output, block["block"], arm, templates, freeze)
    verify_freeze(output, current_sources=True)
    for entry in freeze["models"].values():
        require(identity(entry["path"]) == {key: entry[key] for key in ("bytes", "sha256")}, "Model changed during run")
    write(output / "execution-complete.json", {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "models_and_sources_unchanged": True, "qualification_passed": False,
    })
    audit(output)


def audit(output):
    freeze, protocol = verify_freeze(output)
    cases = json.loads((output / "cases.json").read_text())["cases"]
    results = {arm: [] for arm in ("fixed-policy", *ARMS)}
    for template in cases:
        for arm in results:
            block = next(b["block"] for b in protocol["blocks"] if template["case_id"] in b["case_ids"])
            directory = output / ("fixed-policy" if arm == "fixed-policy" else f"block-{block:02d}/{arm}")
            path = directory / (template["case_id"] + ".json")
            if not path.exists():
                results[arm].append({"case_id": template["case_id"], "family": template["profile"],
                                     "condition": template["variation"], "passed": False, "missing": True})
                continue
            stored = json.loads(path.read_text())
            require(stored["case"] == with_clock(template, stored["fixture_epoch"]), "Case changed after freeze")
            raw = canonical(stored["case"]["document"])
            fixed = arm == "fixed-policy"
            checked = evaluate(stored, raw, baseline=fixed)
            require(checked == stored["evaluation"], "Saved score does not reconstruct")
            if not fixed:
                require((directory / "work-evidence/equipment-interfaces" / path.name).read_bytes() == raw,
                        "Retained equipment bytes changed")
                report = json.loads((directory / "report.json").read_text())
                require(report["shutdown"]["owned_server_stopped"] is True, "Model shutdown is unverified")
                require(all(r["body"]["model"] == report["model_alias"]
                            for r in stored["job"]["record"]["trace"]["http_requests"]),
                        "Diagnostic used a different model incarnation")
            seconds = stored["baseline_seconds"] if fixed else stored["seconds"]
            passed = bool(checked["semantic_passed"] and checked["audit_passed"] and not stored["failure"]
                          and seconds <= LIMITS["case_seconds"]
                          and checked["expected_status"] == template["expected_status"]
                          and checked["expected_value"] == template["expected_value"])
            if not fixed:
                require(passed == stored["passed"], "Stored pass differs from reconstructed result")
            results[arm].append({
                "case_id": template["case_id"], "family": template["profile"],
                "condition": template["variation"], "passed": passed,
                "trace_audit_passed": checked["audit_passed"], "seconds": seconds,
                "model_calls": checked["model_calls"], "tool_calls": checked["tool_calls"],
                "failure": stored["failure"],
            })
    summary = {
        "schema": "replicate-paper-heldout-results-v1",
        "reconstructed_at": datetime.now(timezone.utc).isoformat(),
        "protocol": identity(output / "protocol.json"), "cases_per_arm": len(cases),
        "families": 4, "one_shot_synthetic_holdout": True,
        "native_linux_execution": False, "external_replication": False,
        "arms": {arm: {"successes": sum(r["passed"] for r in rows), "cases": rows,
                       "by_family": {f: sum(r["passed"] for r in rows if r["family"] == f)
                                     for f in sorted({r["family"] for r in rows})}}
                 for arm, rows in results.items()},
        "qualification_passed": False,
    }
    write(output / "results.json", summary)
    print(json.dumps({arm: result["successes"] for arm, result in summary["arms"].items()}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "audit"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        asyncio.run(run(args.output.resolve()))
    elif args.action == "prepare":
        prepare(args.output.resolve())
    else:
        audit(args.output.resolve())
