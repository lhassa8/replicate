"""Owner observer for frozen tasks in an already running native application.

This module has no application launcher, model client, or fixed-policy solver.
It publishes only equipment documents, requests diagnostics over the local
socket, and reads the application's journal without opening it for writing.
Host admission, resource evidence, retirement and cleanup remain independent.
"""

from datetime import datetime, timezone
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sqlite3
import stat
import struct
import time
import uuid

from ..audit_equipment_diagnostics import encoded, require
from ..audit_trained_tasks import evaluate, with_clock
from ..first_lineage_cases import load_frozen


MAX_RPC_BYTES = 65536
MAX_TRACE_BYTES = 512 * 1024
CASE_SECONDS = 180
SUITE_SECONDS = 3900


def bounded_file(path, *, uid, maximum):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        entry = os.fstat(stream.fileno())
        require(stat.S_ISREG(entry.st_mode) and entry.st_uid == uid
                and entry.st_nlink == 1 and not entry.st_mode & 0o077
                and 0 < entry.st_size <= maximum, "Observer input is not a bounded private file")
        raw = stream.read(maximum + 1)
    require(len(raw) == entry.st_size, "Observer input changed while reading")
    return raw


def request(socket_path, value, *, application_pid, uid, deadline):
    remaining = deadline - time.monotonic()
    require(remaining > 0, "Task observer exhausted its execution allowance")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(min(5, remaining))
        connection.connect(str(socket_path))
        peer = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        require(peer[:2] == (application_pid, uid), "Local socket belongs to another application")
        connection.sendall(encoded(value) + b"\n")
        raw = b""
        while not raw.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            require(remaining > 0, "Task reply exceeded the execution allowance")
            connection.settimeout(min(5, remaining))
            block = connection.recv(min(4096, MAX_RPC_BYTES + 1 - len(raw)))
            require(block, "Local task reply ended before its frame")
            raw += block
            require(len(raw) <= MAX_RPC_BYTES, "Local task reply exceeded its byte allowance")
        value = json.loads(raw)
        require(isinstance(value, dict), "Local task reply is not an object")
        return value


def read_job(work, request_id, *, start_token, uid):
    """Read a settled job without ApplicationStore's initialization or recovery."""
    require(re.fullmatch("[0-9a-f]{32}", request_id) is not None, "Invalid diagnostic identity")
    path = Path(work) / "application.sqlite3"
    entry = path.lstat()
    require(stat.S_ISREG(entry.st_mode) and entry.st_uid == uid
            and entry.st_nlink == 1 and not entry.st_mode & 0o077,
            "Application journal is not a private regular file")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)) as db:
        db.execute("PRAGMA query_only=ON")
        row = db.execute(
            "SELECT request_id,device_id,token,status,digest,body FROM jobs WHERE request_id=?",
            (request_id,)).fetchone()
    after = path.lstat()
    require((entry.st_dev, entry.st_ino) == (after.st_dev, after.st_ino),
            "Application journal was replaced during readback")
    require(row is not None and row[0] == request_id and row[2] == start_token,
            "Diagnostic belongs to another request or application incarnation")
    raw = row[5].encode()
    require(len(raw) <= MAX_TRACE_BYTES and hashlib.sha256(raw).hexdigest() == row[4],
            "Application journal trace identity differs")
    record = json.loads(raw)
    require(record["request_id"] == row[0] and record["device_id"] == row[1]
            and record["status"] == row[3], "Application journal columns differ from its body")
    return {"digest": row[4], "record": record}


def publish_document(work, item, *, uid):
    directory = Path(work) / "equipment-interfaces"
    raw = encoded(item["document"])
    require(len(raw) <= MAX_RPC_BYTES, "Equipment observation exceeded its byte allowance")
    name = item["case_id"] + ".json"
    require(re.fullmatch("equipment-[0-9a-f]{20}.json", name) is not None, "Unexpected frozen device name")
    path = directory / name
    # Root publishes a fresh fixture for the unprivileged application. An
    # existing observation, symlink or prior trial must never be overwritten.
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        entry = os.fstat(directory_fd)
        require(stat.S_ISDIR(entry.st_mode) and entry.st_uid == uid
                and not entry.st_mode & 0o077, "Equipment directory is not private application storage")
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory_fd)
        with os.fdopen(fd, "wb") as stream:
            os.fchown(stream.fileno(), uid, entry.st_gid)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path, raw


def check_health(value, start_token):
    expected = {"status": "serving", "inference_ready": True, "native": True,
                "trained_for_organism": True, "start_token": start_token,
                "discovery_running": True, "reproduction_running": True}
    require(all(type(value.get(key)) is type(expected_value) and value[key] == expected_value
                for key, expected_value in expected.items()), "Native application health or identity changed")


def check_public(value, request_id, device_id):
    require(value.get("request_id") == request_id and value.get("device_id") == device_id
            and value.get("status") in ("running", "finished", "failed", "interrupted"),
            "Diagnostic response changed its identity or state")


def check_model_incarnation(job, start_token):
    requests = job["record"]["trace"]["http_requests"]
    require(requests and all(call["body"]["model"] == "organism-" + start_token for call in requests),
            "Native task used a different model incarnation")


def write(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(raw)


def evaluate_case(stored, raw):
    checked = evaluate(stored, raw)
    checked["frozen_condition_preserved"] = (
        checked.get("expected_status") == stored["case"]["expected_status"]
        and checked.get("expected_value") == stored["case"]["expected_value"])
    return checked


def run(work, output, frozen, cases_sha256, *, binding, application_pid, uid):
    """Observe twenty tasks; never start, restart, stop or retire an individual."""
    require(platform.system() == "Linux" and os.geteuid() == 0,
            "Native task observation requires the independent Linux owner")
    require(type(uid) is int and uid > 0 and type(application_pid) is int and application_pid > 1,
            "A separately observed unprivileged application is required")
    require(set(binding) == {"device_id", "individual_id", "release_id", "start_token"}
            and all(isinstance(value, str) and re.fullmatch("[0-9a-f]{" + str(
                32 if key in ("individual_id", "start_token") else 64) + "}", value)
                    for key, value in binding.items()), "Native application binding is incomplete")
    work, output = Path(work).absolute(), Path(output).absolute()
    require(work.resolve() == work and output.resolve() == output
            and work.is_dir() and work.stat().st_uid == uid,
            "Expected the observed application's work directory")
    require(not output.is_relative_to(work) and not work.is_relative_to(output),
            "Owner answers and traces must remain outside application storage")
    cases = load_frozen(frozen, cases_sha256)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    output.chmod(0o700)
    write(output / "binding.json", binding)
    (output / "temperature-cases.frozen.json").write_bytes(Path(frozen).read_bytes())
    report = {"schema": "organism-native-first-lineage-tasks-v1", "binding": binding,
              "frozen_cases_sha256": cases_sha256, "application_pid": application_pid,
              "application_uid": uid, "case_seconds": CASE_SECONDS, "suite_seconds": SUITE_SECONDS,
              "started_at": datetime.now(timezone.utc).isoformat(), "cases": [], "failure": None,
              "native_execution_verified": False, "qualification_passed": False}
    deadline = time.monotonic() + SUITE_SECONDS
    arguments = {"application_pid": application_pid, "uid": uid}
    socket_path = work / "agent.sock"
    try:
        receipt = json.loads(bounded_file(work / ("application-" + binding["start_token"] + ".json"),
                                          uid=uid, maximum=MAX_RPC_BYTES))
        require(all(receipt.get(key) == value for key, value in binding.items())
                and receipt.get("native") is True and receipt.get("trained_for_organism") is True,
                "Application receipt differs from the admitted individual")
        write(output / "application-receipt-before.json", receipt)
        for template in cases:
            health_before = request(socket_path, {"operation": "health"}, deadline=deadline, **arguments)
            check_health(health_before, binding["start_token"])
            epoch = time.time()
            require(math.isfinite(epoch) and epoch > 0, "Invalid fixture observation time")
            item = with_clock(template, epoch)
            path, raw = publish_document(work, item, uid=uid)
            request_id = uuid.uuid4().hex
            case_deadline = min(deadline, time.monotonic() + CASE_SECONDS)
            message = {"operation": "diagnose", "request_id": request_id, "device_id": item["case_id"]}
            # This file exists even if the first reply or subsequent polling fails.
            attempt = {"case": item, "fixture_epoch": epoch, "request_id": request_id,
                       "observation_sha256": hashlib.sha256(raw).hexdigest(),
                       "health_before": health_before}
            write(output / (item["case_id"] + "-attempt.json"), attempt)
            replies = [request(socket_path, message, deadline=case_deadline, **arguments) for _ in range(2)]
            for value in replies:
                check_public(value, request_id, item["case_id"])
            while True:
                state = request(socket_path, {"operation": "status", "request_id": request_id},
                                deadline=case_deadline, **arguments)
                check_public(state, request_id, item["case_id"])
                if state["status"] != "running":
                    break
                time.sleep(min(0.25, max(0, case_deadline - time.monotonic())))
            job = read_job(work, request_id, start_token=binding["start_token"], uid=uid)
            check_model_incarnation(job, binding["start_token"])
            require(state["trace_sha256"] == job["digest"] and state["status"] == job["record"]["status"],
                    "Socket outcome differs from the settled journal")
            require(bounded_file(path, uid=uid, maximum=MAX_RPC_BYTES) == raw,
                    "Read-only diagnostic changed its equipment observation")
            require(time.monotonic() <= case_deadline, "Task evidence exceeded its execution allowance")
            stored = {**attempt, "clock_window": [epoch - 1, time.time() + 1], "job": job,
                      "first_replies": replies, "final_reply": state, "input_unchanged": True,
                      "retry_preserved_identity": True}
            checked = evaluate_case(stored, raw)
            stored["evaluation"] = checked
            write(output / (item["case_id"] + ".json"), stored)
            report["cases"].append({"case_id": item["case_id"], **checked})
        report["final_health"] = request(socket_path, {"operation": "health"}, deadline=deadline, **arguments)
        check_health(report["final_health"], binding["start_token"])
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["tasks_passed"] = bool(report["failure"] is None and len(report["cases"]) == 20
                                and all(item["semantic_passed"] and item["frozen_condition_preserved"]
                                        for item in report["cases"]))
    report["requires_independent_host_admission_and_cleanup"] = True
    write(output / "report.json", report)
    return report


def audit(output, *, cases_sha256, binding):
    """Reconstruct task results; native process/resource proof is a separate gate."""
    output = Path(output)
    cases = load_frozen(output / "temperature-cases.frozen.json", cases_sha256)
    report = json.loads((output / "report.json").read_bytes())
    require(report["schema"] == "organism-native-first-lineage-tasks-v1"
            and report["binding"] == binding
            and json.loads((output / "binding.json").read_bytes()) == binding
            and report["frozen_cases_sha256"] == cases_sha256
            and report["case_seconds"] == CASE_SECONDS and report["suite_seconds"] == SUITE_SECONDS
            and report["qualification_passed"] is False
            and report["native_execution_verified"] is False,
            "Native task report changed its scope or binding")
    receipt = json.loads((output / "application-receipt-before.json").read_bytes())
    require(all(receipt.get(key) == value for key, value in binding.items())
            and receipt.get("native") is True and receipt.get("trained_for_organism") is True,
            "Saved application receipt differs")
    reconstructed, request_ids = [], set()
    require([item["case_id"] for item in report["cases"]]
            == [item["case_id"] for item in cases], "Frozen task inventory is incomplete or reordered")
    for template in cases:
        stored = json.loads((output / (template["case_id"] + ".json")).read_bytes())
        epoch = stored["fixture_epoch"]
        require(type(epoch) in (float, int) and math.isfinite(epoch) and epoch > 0,
                "Invalid saved fixture clock")
        require(stored["case"] == with_clock(template, epoch),
                "Native input or answer changed beyond the fresh fixture timestamp")
        raw = encoded(stored["case"]["document"])
        request_id = stored["request_id"]
        require(re.fullmatch("[0-9a-f]{32}", request_id) is not None and request_id not in request_ids
                and stored["job"]["record"]["request_id"] == request_id,
                "Repeated or mismatched native task request")
        request_ids.add(request_id)
        require(stored["observation_sha256"] == hashlib.sha256(raw).hexdigest()
                and stored["input_unchanged"] is True and stored["retry_preserved_identity"] is True
                and epoch - 1 == stored["clock_window"][0]
                and epoch <= stored["clock_window"][1] <= epoch + CASE_SECONDS + 1,
                "Native observation evidence or time allowance differs")
        check_health(stored["health_before"], binding["start_token"])
        require(len(stored["first_replies"]) == 2, "Missing native retry observation")
        for value in [*stored["first_replies"], stored["final_reply"]]:
            check_public(value, request_id, template["case_id"])
        require(stored["final_reply"]["trace_sha256"] == stored["job"]["digest"]
                and stored["final_reply"]["status"] == stored["job"]["record"]["status"],
                "Saved socket reply differs from the journal")
        check_model_incarnation(stored["job"], binding["start_token"])
        attempt = json.loads((output / (template["case_id"] + "-attempt.json")).read_bytes())
        require(attempt == {key: stored[key] for key in (
            "case", "fixture_epoch", "request_id", "observation_sha256", "health_before")},
            "Saved pre-request evidence differs")
        checked = evaluate_case(stored, raw)
        require(checked == stored["evaluation"], "Native task evaluation differs from readback")
        reconstructed.append({"case_id": template["case_id"], **checked})
    require(reconstructed == report["cases"], "Native task summary differs from readback")
    check_health(report["final_health"], binding["start_token"])
    passed = report["failure"] is None and all(
        item["semantic_passed"] and item["frozen_condition_preserved"] for item in reconstructed)
    require(report["tasks_passed"] is passed, "Native task pass claim differs from its evidence")
    return {"task_evidence_reconstructed": True, "tasks_passed": passed,
            "successes": sum(item["semantic_passed"] and item["frozen_condition_preserved"]
                             for item in reconstructed),
            "native_execution_verified": False, "qualification_passed": False}
