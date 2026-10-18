"""Two live Mac stock-model tasks with explicitly fixture-sourced reader notes."""

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import tempfile
import time
import uuid

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

from .application_store import ApplicationStore
from .audit_equipment_diagnostics import inspect_job
from .audit_inherited_diagnostics import inspect_inherited_job
from .capsule import canonical
from .enrollment import Enrollment
from .equipment_baseline import solve
from .equipment_development import case
from .equipment_diagnostic_agent import diagnose
from .equipment_interfaces import InterfaceDirectory, InterfaceSession
from .experience import require, VerifiedExperience
from .experience_issue import checker_identity, issue
from .experience_store import ExperienceStore
from .inference_server import InferenceServer, RuntimeConfig, digest
from .package_application import APPLICATION_MODULES
from .probe_experience import job as fixture_job, lab, RELEASE


SELECTION = (("a", "fault", "nominal"), ("e", "nominal", "fault"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate(stored, policy, checker):
    try:
        result = inspect_inherited_job(stored["case"], canonical(stored["case"]["document"]), stored["job"],
            policy=policy, release_id=RELEASE, allowed_checkers={checker},
            records={identity: base64.b64decode(raw) for identity, raw in stored["available_records"].items()},
            require_live_source=False, live=True, clock_window=stored["clock_window"])
        return {"audit_passed": True, **result}
    except Exception as exc:
        trace = stored["job"]["record"]["trace"]
        return {"audit_passed": False, "semantic_passed": False,
                "audit_failure": type(exc).__name__ + ": " + str(exc),
                "model_calls": len(trace.get("model_calls", [])), "tool_calls": len(trace.get("tool_events", [])),
                "inherited_reader_uses": [], "qualification_passed": False}


def audit_saved(output):
    report = json.loads((output / "report.json").read_bytes())
    context = json.loads((output / "lab.json").read_bytes())
    policy = Enrollment(base64.b64decode(context["enrollment"]), base64.b64decode(context["owner_public"]))
    require(len(report["cases"]) == len(SELECTION), "Inherited development selection is incomplete")
    for name, expected in report["source_hashes"].items():
        require(digest(output / "source-snapshot" / name) == expected, "Retained inherited-probe source changed")
    outcomes = []
    for summary in report["cases"]:
        stored = json.loads((output / (summary["case_id"] + ".json")).read_bytes())
        check = evaluate(stored, policy, context["checker_sha256"])
        require(check == stored["evaluation"] and check["semantic_passed"] == summary["model_passed"],
                "Inherited diagnostic readback differs")
        for identity, encoded in stored["available_records"].items():
            signed = base64.b64decode(encoded)
            note = VerifiedExperience(signed, policy, release_id=RELEASE,
                allowed_checkers={context["checker_sha256"]},
                now=json.loads(signed)["body"]["created_unix_seconds"], require_live_source=False)
            source = json.loads((output / ("source-" + identity + ".json")).read_bytes())
            raw = canonical(source["case"]["document"])
            require(inspect_job(source["case"], raw, source["job"])["semantic_passed"],
                    "Source fixture diagnosis no longer validates")
            require(note.body["source"]["kind"] == "development_fixture"
                    and note.body["source"]["job_sha256"] == source["job"]["digest"]
                    and note.body["source"]["observation_sha256"] == hashlib.sha256(raw).hexdigest()
                    and note.body["source"]["validation_sha256"] == hashlib.sha256(canonical(source["validation"])).hexdigest(),
                    "Retained note does not bind its source fixture")
        outcomes.append(check)
    return {"audit_passed": all(item["audit_passed"] for item in outcomes), "cases": len(outcomes),
            "model_successes": sum(item["semantic_passed"] for item in outcomes),
            "cases_using_inherited_readers": sum(bool(item["inherited_reader_uses"]) for item in outcomes),
            "fixture_source_notes": True, "native_execution_verified": False, "qualification_passed": False}


async def run(root, output):
    require(platform.system() == "Darwin", "This is a Mac development probe")
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((root / "organism/experiment-contract.draft.json").read_bytes())
    model = contract["model"]
    policy, owner, keys, identities = lab()
    checker = checker_identity()[0]
    write(output / "lab.json", {"enrollment": base64.b64encode(policy.signed).decode(),
        "owner_public": base64.b64encode(owner).decode(), "device_ids": identities,
        "checker_sha256": checker, "release_id": RELEASE, "all_identities_on_one_mac": True})
    report = {"scope": "Live stock target diagnostics using explicitly fixture-sourced procedure notes",
        "started_at": datetime.now(timezone.utc).isoformat(), "cases": [], "failure": None,
        "model_sha256": model["local_export_sha256"], "model_parameters": 3659737600,
        "source_model_calls": 0, "target_model_calls": 0, "qualification_passed": False,
        "trained_for_organism": False, "native_execution_verified": False, "q7_comparison": False,
        "automatic_peer_exchange_exercised": False, "application_rpc_exercised": False,
        "source_hashes": {}, "private_identity_keys_retained": False}
    sources = {name + ".py" for name in APPLICATION_MODULES} | {
        Path(__file__).name, "package_application.py", "audit_inherited_diagnostics.py",
        "audit_equipment_diagnostics.py", "audit_application.py", "equipment_development.py",
        "equipment_baseline.py", "experience_issue.py", "diagnostic_oracle.py", "probe_experience.py"}
    (output / "source-snapshot").mkdir()
    for name in sorted(sources):
        raw = (root / "organism" / name).read_bytes()
        (output / "source-snapshot" / name).write_bytes(raw)
        report["source_hashes"][name] = hashlib.sha256(raw).hexdigest()
    write(output / "request.json", report)
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="inherit-live-", dir="/private/tmp") as temporary:
        work = Path(temporary)
        config = RuntimeConfig(root / "models/granite-3b/granite-base-Q4_K_M.gguf",
            model["local_export_sha256"], model["local_export_bytes"],
            root / "tools/llama.cpp/b9850/llama-b9850/llama-server", contract["runtime"]["development_binary_sha256"],
            uuid.uuid4().hex, identities[2], uuid.uuid4().hex, RELEASE, False, False)
        server = InferenceServer(config, work)
        jobs = ApplicationStore(work, config.start_token)
        experiences = ExperienceStore(work / "experience", policy, keys[2], release_id=RELEASE,
                                       allowed_checkers={checker}, require_live_source=False)
        (work / "source").mkdir()
        (work / "equipment").mkdir()
        observations = InterfaceDirectory(work / "equipment")
        records = {}
        try:
            report["inference"] = await server.start()
            report["server_pid"] = server.process.pid
            print(json.dumps({"application_component": "serving", "server_pid": server.process.pid}), flush=True)
            for profile, source_condition, target_condition in SELECTION:
                source_started, source_now = time.monotonic(), int(time.time())
                source_case = case(profile, source_condition, source_now)
                source_raw = canonical(source_case["document"])
                (work / "source" / (source_case["case_id"] + ".json")).write_bytes(source_raw)
                source_session = InterfaceSession(InterfaceDirectory(work / "source", clock=lambda: source_now),
                                                   source_case["case_id"])
                solve(source_session, source_case["case_id"])
                source_job = fixture_job(source_session)
                plan = {"schema": "organism-diagnostic-oracle-v1",
                    **{key: value for key, value in source_case["expected_reader"].items() if key != "device_id"},
                    "operation": {"kind": "scalar"}}
                signed, validation = issue(source_raw, source_job, plan, policy=policy, producer_key=keys[0],
                    verifier_key=keys[1], producer_individual_id="a" * 32, release_id=RELEASE,
                    created=source_now, expires=source_now + 3600, source_clock=source_now, live=False)
                identity = experiences.add(signed)
                records[identity] = base64.b64encode(signed).decode()
                write(output / ("source-" + identity + ".json"), {"case": source_case, "job": source_job,
                    "validation": validation, "reviewed_fixture_plan": plan,
                    "fixture_creation_seconds": round(time.monotonic() - source_started, 6)})
                window_start = time.time()
                item = case(profile, target_condition, window_start)
                path = work / "equipment" / (item["case_id"] + ".json")
                raw = canonical(item["document"])
                path.write_bytes(raw)
                request = uuid.uuid4().hex
                jobs.reserve(request, item["case_id"], task_kind="equipment_interface_v1")
                target_started = time.monotonic()
                job = await diagnose(server, observations, jobs, request, item["case_id"], experience_store=experiences)
                elapsed = time.monotonic() - target_started
                stored = {"case": item, "job": job, "available_records": dict(records),
                          "clock_window": [window_start - 1, time.time() + 1],
                          "target_elapsed_seconds": round(elapsed, 6)}
                stored["evaluation"] = evaluate(stored, policy, checker)
                require(path.read_bytes() == raw, "Read-only diagnostic changed its equipment fixture")
                write(output / (item["case_id"] + ".json"), stored)
                result = stored["evaluation"]
                summary = {"case_id": item["case_id"], "model_passed": result["semantic_passed"],
                    "audit_passed": result["audit_passed"], "model_calls": result["model_calls"],
                    "tool_calls": result["tool_calls"], "inherited_reader_uses": result["inherited_reader_uses"],
                    "seconds": round(elapsed, 6), "job_status": job["record"]["status"],
                    "failure": job["record"]["failure"], "audit_failure": result.get("audit_failure")}
                report["cases"].append(summary)
                report["target_model_calls"] += summary["model_calls"]
                write(output / "progress.json", report)
                print(json.dumps(summary), flush=True)
        except Exception as exc:
            report["failure"] = type(exc).__name__ + ": " + str(exc)
        finally:
            report["shutdown"] = await server.stop()
            report["server_returncode"] = server.process.returncode if server.process else None
            shutil.copytree(work, output / "work-evidence")
            report["api_key_found_in_model_logs"] = any(
                server.api_key.encode() in path.read_bytes() for path in (output / "work-evidence").glob("model-*.log"))
    report["elapsed_seconds"] = round(time.monotonic() - began, 3)
    report["model_child_peak_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    report["sources_unchanged_during_run"] = all(
        digest(root / "organism" / name) == expected for name, expected in report["source_hashes"].items())
    if not report["sources_unchanged_during_run"]:
        report["failure"] = report["failure"] or "Probe source changed during execution"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    write(output / "report.json", report)
    try:
        audit = audit_saved(output)
    except Exception as exc:
        audit = {"audit_passed": False, "failure": type(exc).__name__ + ": " + str(exc), "qualification_passed": False}
    write(output / "audit.json", audit)
    print(json.dumps({"report": str(output / "report.json"), "audit": audit, "failure": report["failure"]}), flush=True)
    return report, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, audit = asyncio.run(run(Path(__file__).resolve().parents[1], args.output))
    if report["failure"] or not audit["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
