"""Two-hour owner observation with six repeated five-category task cycles."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import uuid

from ..audit_equipment_diagnostics import encoded
from ..audit_trained_tasks import with_clock
from ..first_lineage_cases import load_frozen
from ..training_inputs import require
from . import first_lineage_tasks as tasks
from .first_lineage_profile import CASES_SHA256


DURATION = 7200
CYCLE_SECONDS = 1200
CYCLES = 6
VARIATIONS = ("nominal", "fault", "stale", "invalid", "missing")
MAXIMUM_HEALTH_GAP = 195


def scheduled_case(cases, index, epoch):
    require(type(index) is int and 0 <= index < CYCLES * len(VARIATIONS),
            "Endurance task is outside its declared six cycles")
    require(type(epoch) in (int, float) and math.isfinite(epoch) and epoch > 0,
            "Endurance fixture clock is invalid")
    variation = VARIATIONS[index % len(VARIATIONS)]
    cycle = index // len(VARIATIONS)
    choices = [item for item in cases if item["variation"] == variation]
    require(len(choices) == 4, "Endurance requires all frozen temperature variants")
    template = deepcopy(choices[cycle % 4])
    original_id = template["case_id"]
    token = hashlib.sha256(("first-lineage-endurance-v1/" + str(index) + "/" + original_id).encode()).hexdigest()[:20]
    device_id = "equipment-" + token
    template["case_id"] = template["document"]["device_id"] = device_id
    template["expected_reader"]["device_id"] = device_id
    return with_clock(template, epoch)


def run(work, output, frozen, *, binding, application_pid, uid, start_epoch):
    require(platform.system() == "Linux" and os.geteuid() == 0,
            "Endurance execution requires the independent native Linux owner")
    require(type(start_epoch) in (int, float) and math.isfinite(start_epoch)
            and 0 < start_epoch - time.time() <= 300,
            "Endurance requires a common future start within five minutes")
    require(type(uid) is int and uid > 0 and type(application_pid) is int and application_pid > 1,
            "Endurance requires a separately observed application")
    cases = load_frozen(frozen, CASES_SHA256)
    work, output = Path(work).absolute(), Path(output).absolute()
    require(work.resolve() == work and output.resolve() == output
            and not output.is_relative_to(work) and not work.is_relative_to(output)
            and work.is_dir() and work.stat().st_uid == uid,
            "Owner endurance evidence must be separate from application storage")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    output.chmod(0o700)
    (output / "temperature-cases.frozen.json").write_bytes(Path(frozen).read_bytes())
    tasks.write(output / "binding.json", binding)
    socket_path = work / "agent.sock"
    arguments = {"application_pid": application_pid, "uid": uid}
    health, outcomes = [], []
    report = {
        "schema": "organism-native-lineage-endurance-v1", "binding": binding,
        "application_pid": application_pid, "application_uid": uid,
        "frozen_cases_sha256": CASES_SHA256, "start_epoch": start_epoch,
        "duration_seconds": DURATION, "cycle_seconds": CYCLE_SECONDS,
        "cycles": CYCLES, "categories_per_cycle": list(VARIATIONS),
        "case_seconds": tasks.CASE_SECONDS, "health": health, "cases": outcomes,
        "passed": False, "failure": None, "native_execution_verified": False,
        "seed_disconnection_requires_external_aws_proof": True,
        "qualification_passed": False,
    }
    anchor = time.monotonic() + start_epoch - time.time()
    end = anchor + DURATION

    def check_health():
        value = tasks.request(socket_path, {"operation": "health"},
                              deadline=time.monotonic() + 10, **arguments)
        tasks.check_health(value, binding["start_token"])
        health.append({"elapsed_seconds": time.monotonic() - anchor,
                       "observed_at": datetime.now(timezone.utc).isoformat(), "value": value})

    def wait_until(target):
        while time.monotonic() < target:
            check_health()
            time.sleep(min(5, max(0, target - time.monotonic())))

    try:
        application = json.loads(tasks.bounded_file(
            work / ("application-" + binding["start_token"] + ".json"),
            uid=uid, maximum=tasks.MAX_RPC_BYTES))
        require(all(application.get(name) == value for name, value in binding.items())
                and application.get("native") is True and application.get("trained_for_organism") is True,
                "Endurance application receipt changed")
        tasks.write(output / "application-receipt-before.json", application)
        wait_until(anchor)
        report["observed_start_epoch"] = time.time()
        require(abs(report["observed_start_epoch"] - start_epoch) <= 5,
                "Endurance missed the common start")
        for cycle in range(CYCLES):
            wait_until(anchor + cycle * CYCLE_SECONDS)
            require(time.monotonic() <= anchor + cycle * CYCLE_SECONDS + 10,
                    "Endurance cycle started late")
            for offset in range(len(VARIATIONS)):
                index = cycle * len(VARIATIONS) + offset
                check_health()
                epoch = time.time()
                item = scheduled_case(cases, index, epoch)
                path, raw = tasks.publish_document(work, item, uid=uid)
                request_id = uuid.uuid4().hex
                case_deadline = min(end, anchor + (cycle + 1) * CYCLE_SECONDS,
                                    time.monotonic() + tasks.CASE_SECONDS)
                attempt = {"index": index, "case": item, "fixture_epoch": epoch,
                           "request_id": request_id, "observation_sha256": hashlib.sha256(raw).hexdigest(),
                           "health_before": health[-1]["value"]}
                tasks.write(output / (item["case_id"] + "-attempt.json"), attempt)
                message = {"operation": "diagnose", "request_id": request_id, "device_id": item["case_id"]}
                replies = [tasks.request(socket_path, message, deadline=case_deadline, **arguments)
                           for _ in range(2)]
                for value in replies:
                    tasks.check_public(value, request_id, item["case_id"])
                while True:
                    state = tasks.request(socket_path, {"operation": "status", "request_id": request_id},
                                          deadline=case_deadline, **arguments)
                    tasks.check_public(state, request_id, item["case_id"])
                    if state["status"] != "running":
                        break
                    time.sleep(min(0.25, max(0, case_deadline - time.monotonic())))
                job = tasks.read_job(work, request_id, start_token=binding["start_token"], uid=uid)
                tasks.check_model_incarnation(job, binding["start_token"])
                require(state["trace_sha256"] == job["digest"] and state["status"] == job["record"]["status"]
                        and tasks.bounded_file(path, uid=uid, maximum=tasks.MAX_RPC_BYTES) == raw
                        and time.monotonic() <= case_deadline,
                        "Endurance task changed its journal, input or time allowance")
                stored = {**attempt, "clock_window": [epoch - 1, time.time() + 1], "job": job,
                          "first_replies": replies, "final_reply": state,
                          "input_unchanged": True, "retry_preserved_identity": True}
                stored["evaluation"] = tasks.evaluate_case(stored, raw)
                tasks.write(output / (item["case_id"] + ".json"), stored)
                outcomes.append({"index": index, "case_id": item["case_id"], **stored["evaluation"]})
                require(stored["evaluation"]["semantic_passed"]
                        and stored["evaluation"]["frozen_condition_preserved"],
                        "An endurance diagnosis was incorrect")
        wait_until(end)
        check_health()
        report["elapsed_seconds"] = time.monotonic() - anchor
        report["observed_end_epoch"] = time.time()
        require(DURATION <= report["elapsed_seconds"] <= DURATION + 20
                and len(outcomes) == CYCLES * len(VARIATIONS)
                and len(health) <= 1600
                and all(0 <= second["elapsed_seconds"] - first["elapsed_seconds"] <= MAXIMUM_HEALTH_GAP
                        for first, second in zip(health, health[1:])),
                "Endurance duration, inventory or health coverage differs")
        report["passed"] = True
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    tasks.write(output / "report.json", report)
    return report


def audit(output, *, binding, start_epoch):
    output = Path(output)
    cases = load_frozen(output / "temperature-cases.frozen.json", CASES_SHA256)
    report = json.loads((output / "report.json").read_bytes())
    require(report["schema"] == "organism-native-lineage-endurance-v1"
            and report["binding"] == binding == json.loads((output / "binding.json").read_bytes())
            and report["start_epoch"] == start_epoch
            and report["duration_seconds"] == DURATION and report["cycle_seconds"] == CYCLE_SECONDS
            and report["cycles"] == CYCLES and report["categories_per_cycle"] == list(VARIATIONS)
            and report["case_seconds"] == tasks.CASE_SECONDS
            and report["frozen_cases_sha256"] == CASES_SHA256
            and report["native_execution_verified"] is False and report["qualification_passed"] is False
            and report["seed_disconnection_requires_external_aws_proof"] is True,
            "Endurance report changed its declared scope")
    require(report["failure"] is None and report["passed"] is True
            and abs(report["observed_start_epoch"] - start_epoch) <= 5
            and DURATION <= report["elapsed_seconds"] <= DURATION + 20
            and abs(report["observed_end_epoch"] - start_epoch - report["elapsed_seconds"]) <= 5,
            "Endurance time or success record differs")
    application = json.loads((output / "application-receipt-before.json").read_bytes())
    require(all(application.get(name) == value for name, value in binding.items())
            and application.get("native") is True and application.get("trained_for_organism") is True,
            "Endurance application identity changed")
    health = report["health"]
    require(2 <= len(health) <= 1600 and health[0]["elapsed_seconds"] <= 5
            and health[-1]["elapsed_seconds"] >= DURATION
            and all(0 <= b["elapsed_seconds"] - a["elapsed_seconds"] <= MAXIMUM_HEALTH_GAP
                    for a, b in zip(health, health[1:])), "Endurance health coverage contains a gap")
    for entry in health:
        tasks.check_health(entry["value"], binding["start_token"])
    reconstructed, request_ids = [], set()
    for index in range(CYCLES * len(VARIATIONS)):
        expected_id = scheduled_case(cases, index, 1)["case_id"]
        stored = json.loads((output / (expected_id + ".json")).read_bytes())
        epoch = stored["fixture_epoch"]
        require(stored["index"] == index and stored["case"] == scheduled_case(cases, index, epoch),
                "Endurance fixture changed beyond its predetermined ID and fresh timestamp")
        cycle = index // len(VARIATIONS)
        require(start_epoch + cycle * CYCLE_SECONDS - 5 <= epoch
                <= start_epoch + cycle * CYCLE_SECONDS + 4 * tasks.CASE_SECONDS + 15,
                "Endurance task did not run in its declared cycle")
        request_id = stored["request_id"]
        require(request_id not in request_ids and stored["job"]["record"]["request_id"] == request_id,
                "Endurance request was repeated or changed")
        request_ids.add(request_id)
        raw = encoded(stored["case"]["document"])
        require(stored["observation_sha256"] == hashlib.sha256(raw).hexdigest()
                and stored["input_unchanged"] is True and stored["retry_preserved_identity"] is True
                and stored["clock_window"][0] == epoch - 1
                and epoch <= stored["clock_window"][1] <= epoch + tasks.CASE_SECONDS + 1,
                "Endurance input or task clock evidence differs")
        tasks.check_health(stored["health_before"], binding["start_token"])
        require(len(stored["first_replies"]) == 2, "Endurance retry evidence is missing")
        for value in [*stored["first_replies"], stored["final_reply"]]:
            tasks.check_public(value, request_id, expected_id)
        require(stored["final_reply"]["trace_sha256"] == stored["job"]["digest"]
                and stored["final_reply"]["status"] == stored["job"]["record"]["status"],
                "Endurance socket and journal differ")
        tasks.check_model_incarnation(stored["job"], binding["start_token"])
        attempt = json.loads((output / (expected_id + "-attempt.json")).read_bytes())
        require(attempt == {key: stored[key] for key in (
            "index", "case", "fixture_epoch", "request_id", "observation_sha256", "health_before")},
            "Endurance pre-request evidence changed")
        checked = tasks.evaluate_case(stored, raw)
        require(checked == stored["evaluation"] and checked["semantic_passed"]
                and checked["frozen_condition_preserved"], "Endurance diagnosis failed reconstruction")
        reconstructed.append({"index": index, "case_id": expected_id, **checked})
    require(reconstructed == report["cases"], "Endurance summary differs from its actual task records")
    return {"tasks": len(reconstructed), "duration_seconds": report["elapsed_seconds"],
            "semantic_and_time_audit_passed": True, "native_execution_verified": False,
            "requires_external_seed_process_resource_and_cloud_evidence": True,
            "qualification_passed": False}
