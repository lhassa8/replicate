"""Independent, read-only reconstruction of native seed startup artifacts.

This checks captured evidence against an owner-selected local kit. It does not
attest the machine or replace the external AWS launch/cleanup ledger.
"""

import argparse
import base64
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import tarfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ..audit_application import canonical, inspect_diagnostic, sha


MIB = 1024**2
NS_PATH = "/run/netns/organism-native-startup"
DRIVER = "organism-native-startup.service"
BASE = "/var/lib/organism-startup"
KINDS = {"intent", "running", "exit", "check_intent", "check_running", "check_result", "check_admission"}


class AuditError(ValueError):
    pass


def require(value, message):
    if not value:
        raise AuditError(message)


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate evidence field")
            result[key] = value
        return result

    def invalid(_):
        raise AuditError("Nonfinite evidence number")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def moment(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "Evidence timestamp has no timezone")
    return result.timestamp()


def identity(value, length=64):
    require(isinstance(value, str) and re.fullmatch("[0-9a-f]{" + str(length) + "}", value),
            "Invalid evidence identity")
    return value


class Evidence:
    def __init__(self, root):
        self.root = Path(root).absolute()
        require(self.root.is_dir() and not self.root.is_symlink(), "Evidence root is not a directory")
        self.read_hashes = {}

    def path(self, name):
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Evidence path escapes its root")
        current = self.root
        for part in relative.parts[:-1]:
            current /= part
            entry = current.lstat()
            require(stat.S_ISDIR(entry.st_mode), "Evidence parent is not a real directory")
        return self.root / relative

    def read(self, name, maximum=MIB):
        path = self.path(name)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            entry = os.fstat(source.fileno())
            require(stat.S_ISREG(entry.st_mode) and entry.st_nlink == 1 and entry.st_size <= maximum,
                    "Evidence file is linked, irregular or too large")
            raw = source.read(maximum + 1)
        require(len(raw) == entry.st_size and len(raw) <= maximum, "Evidence size changed while reading")
        digest = sha(raw)
        require(name not in self.read_hashes or self.read_hashes[name] == digest, "Evidence changed during audit")
        self.read_hashes[name] = digest
        return raw

    def json(self, name, maximum=MIB):
        return decode(self.read(name, maximum))

    def rows(self, name, query, maximum=8 * MIB):
        self.read(name, maximum)
        with sqlite3.connect(self.path(name).as_uri() + "?mode=ro&immutable=1", uri=True) as db:
            db.execute("PRAGMA query_only=ON")
            rows = db.execute(query).fetchall()
        self.read(name, maximum)
        return rows


def trusted_release(kit, architecture):
    require(architecture in ("arm64", "x86_64"), "Unsupported requested architecture")
    receipt = kit.json("build-receipt.json")
    raw = kit.read("catalog.json")
    require(sha(raw) == receipt["catalog_sha256"], "Reviewed catalog hash differs")
    require(sha(kit.read("kit.tar.gz", 2 * MIB)) == receipt["kit_sha256"], "Reviewed source kit hash differs")
    catalog = decode(raw)
    manifest_raw = base64.b64decode(catalog["manifest_base64"], validate=True)
    manifest = decode(manifest_raw)
    body = manifest["body"]
    public = base64.b64decode(catalog["release_keys"][manifest["signer_id"]], validate=True)
    require(sha(public) == manifest["signer_id"], "Release signer identity differs")
    Ed25519PublicKey.from_public_bytes(public).verify(
        base64.b64decode(manifest["signature"], validate=True), canonical(body))
    require(sha(canonical(body)) == manifest["release_id"] == catalog["release_id"] == receipt["release_id"],
            "Signed release identity differs")
    require(body["schema"] == "organism-capsule-v1" and body["model"]["trained_for_organism"] is False
            and catalog["trained_for_organism"] is False and catalog["qualification_passed"] is False
            and catalog["schema"] == "organism-native-startup-v1", "Not the declared stock startup kit")
    require(body["files"] == catalog["files"] and body["platforms"] == catalog["platforms"]
            and set(body["platforms"]) == {"linux-arm64", "linux-x86_64"} and len(body["files"]) == 11,
            "Complete two-platform payload declaration differs")
    inventory = kit.json("upload-inventory.json")["files"]

    def member(component, wanted):
        entries = [item for item in inventory if item["key_suffix"] == "inputs/capsule/" + component]
        require(len(entries) == 1, "Reviewed component inventory is ambiguous")
        item = entries[0]
        require(item["sha256"] == body["files"][component]["sha256"]
                and item["bytes"] == body["files"][component]["bytes"], "Reviewed component differs from manifest")
        path = Path(item["path"])
        reader = Evidence(path.parent)
        raw = reader.read(path.name, 64 * MIB)
        require(len(raw) == item["bytes"] and sha(raw) == item["sha256"], "Local reviewed component changed")
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            matches = []
            for index, entry in enumerate(archive):
                require(index < 100, "Native component has excessive archive members")
                if entry.name == wanted:
                    require(entry.isfile() and 0 < entry.size <= 64 * MIB, "Native component member is invalid")
                    matches.append(archive.extractfile(entry).read(entry.size + 1))
            require(len(matches) == 1, "Native component member is missing or duplicated")
        return sha(matches[0])

    native = body["platforms"]["linux-" + architecture]
    checker = member(body["self_test"], "readiness.py")
    runtime = member(native["inference"], "llama-b9850/llama-server")
    model = body["files"][body["model"]["path"]]
    return catalog, {"release_id": manifest["release_id"], "manifest_sha256": sha(manifest_raw),
                     "checker_sha256": checker, "runtime_sha256": runtime,
                     "model_sha256": model["sha256"], "model_bytes": model["bytes"]}


def owner_policy(evidence, catalog, architecture):
    config = evidence.json("seed-evidence/receiver.json", 16384)
    enrollment = evidence.json("seed-evidence/enrollment.json")
    book = evidence.json("seed-evidence/discovery.json")
    public = base64.b64decode(config["trusted_owner_public_key"], validate=True)
    owner = sha(public)
    for signed in (enrollment, book):
        require(signed["owner_id"] == owner, "Enrollment and discovery owners differ")
        Ed25519PublicKey.from_public_bytes(public).verify(
            base64.b64decode(signed["signature"], validate=True), canonical(signed["body"]))
    policy = enrollment["body"]
    policy_id = sha(canonical(enrollment))
    device = identity(config["device_id"])
    require(policy["schema"] == "organism-enrollment-v1" and policy["require_trained"] is False
            and policy["release_keys"] == catalog["release_keys"] and len(policy["devices"]) == 2
            and policy["max_births_per_device"] == 1 and policy["max_process_starts"] == 1
            and policy["birth_seconds"] == 1200, "Owner seed enrollment differs from its declared bounds")
    require(device in policy["devices"], "Seed is not in its enrollment")
    for enrolled, value in policy["devices"].items():
        require(sha(base64.b64decode(value["public_key"], validate=True)) == enrolled
                and value["platform"] == "linux-" + architecture, "Enrolled platform or device identity differs")
    endpoints = book["body"]["endpoints"]
    require(book["body"]["policy_id"] == policy_id and set(endpoints) == set(policy["devices"])
            and book["body"]["lab_cidrs"] == ["127.0.0.0/24"]
            and endpoints[device] == {"address": "127.0.0.1", "port": 5443}
            and endpoints[next(key for key in endpoints if key != device)] == {"address": "127.0.0.1", "port": 5444},
            "Owner seed endpoints differ")
    require(config["network_namespace"] == NS_PATH and config["bind_address"] == "127.0.0.1"
            and config["bind_port"] == 5443 and config["lab_cidrs"] == ["127.0.0.0/24"]
            and config["service_user"] == config["service_group"] == "organism",
            "Owner receiver configuration differs")
    return device, policy_id, next(key for key in policy["devices"] if key != device)


def records(evidence, seed, device, policy_id, release):
    require(evidence.read("seed-evidence/state/supervision.enrolled", 128) == b"organism-supervision-v1\n",
            "Supervision marker differs")
    rows = evidence.rows("seed-evidence/state/supervision.sqlite3",
                         "SELECT token,kind,digest,substr(body,1,65537) FROM records LIMIT 8")
    require(6 <= len(rows) <= 7, "Startup supervision record set is incomplete or exceeds one incarnation")
    result, shared = {}, None
    for token, kind, digest, text in rows:
        require(kind in KINDS and kind not in result and len(text.encode()) <= 65536
                and sha(text.encode()) == digest, "Supervision record is corrupt or duplicated")
        body = decode(text)
        require(body["schema"] == "organism-supervision-record-v1" and body["kind"] == kind
                and body["process_token"] == token and body["device_id"] == device
                and body["policy_id"] == policy_id and body["operation_id"] == seed["operation_id"]
                and body["manifest_sha256"] == release["manifest_sha256"], "Supervision record identity differs")
        identity(token, 32)
        identity(body["individual_id"], 32)
        current = {key: body[key] for key in ("individual_id", "process_token", "operation_id", "boot_id")}
        require(shared is None or shared == current, "Supervision records mix process incarnations")
        shared = current
        result[kind] = {"digest": digest, "record": body}
    require(set(result) >= KINDS - {"check_running"}, "Required supervision record is missing")
    for kind in KINDS - {"exit"}:
        require(seed["readiness_records"][kind] == result.get(kind), "Seed receipt differs from supervision database")
    require(seed["exit_record"] == result["exit"], "Seed exit differs from supervision database")
    return result, shared


def transport(evidence, seed, catalog, bootstrap, *, state_prefix="seed-evidence/state/"):
    require(evidence.read(state_prefix + "peer-audit.enrolled", 128) == b"organism-peer-audit-v1\n",
            "Transport audit marker differs")
    name = state_prefix + "peer-audit.sqlite3"
    rows = evidence.rows(name, "SELECT sequence,previous,digest,substr(body,1,2049) FROM records ORDER BY sequence LIMIT 65537",
                         maximum=192 * MIB)
    head = evidence.rows(name, "SELECT sequence,digest FROM head LIMIT 2", maximum=192 * MIB)
    require(0 < len(rows) <= 65536 and len(head) == 1, "Transport history is empty or exceeds its bound")
    previous, payloads, selected, failures = "0" * 64, {}, None, []
    ready = seed["ready_receipt"]
    anchor = ready["transport_receipt"]
    for index, (sequence, ancestor, digest, text) in enumerate(rows, 1):
        require(sequence == index and ancestor == previous and len(text.encode()) <= 2048
                and sha(bytes.fromhex(previous) + text.encode()) == digest, "Transport history chain differs")
        previous = digest
        row = decode(text)
        if row["phase"] in ("failed", "rejected", "response_lost"):
            failures.append({"sequence": sequence, "phase": row["phase"], "detail": row["detail"]})
        if row["phase"] == "completed" and row["operation_id"] == seed["operation_id"]:
            require(row["peer_id"] == bootstrap, "Seed transfer or receipt used an unexpected peer identity")
            detail = row["detail"]
            if row["operation"] == "put":
                payloads[detail["payload"]] = {"bytes": detail["bytes"], "sha256": detail["sha256"]}
            if sequence == anchor["sequence"]:
                require(digest == anchor["digest"] and row["operation"] == "status"
                        and detail["status"] == "ready" and detail["release_id"] == catalog["release_id"],
                        "Ready receipt transport anchor differs")
                selected = detail["response_sha256"]
    require(head[0] == (len(rows), previous), "Transport history head differs")
    require(payloads == catalog["files"], "Transport lacks the complete signed capsule")
    require(selected == sha(canonical({key: value for key, value in ready.items() if key != "transport_receipt"})),
            "Ready receipt bytes differ from receiver transport history")
    return {"records": len(rows), "payload_components": len(payloads), "ready_receipt_anchor_verified": True,
            "retained_failed_attempts": failures}


def namespace(binding, observed, pid, *, expected_path=NS_PATH):
    require(binding["path"] == expected_path
            and all(type(binding[key]) is int and binding[key] >= 0
                    for key in ("device", "inode", "host_device", "host_inode"))
            and (binding["device"], binding["inode"]) != (binding["host_device"], binding["host_inode"]),
            "Namespace binding is invalid or identifies the host")
    require(type(pid) is int and pid > 1 and observed == {"pid": pid, "device": binding["device"], "inode": binding["inode"]},
            "Process namespace differs from its bound namespace")


def running(value, intent, boot, *, namespace_path=NS_PATH):
    properties, process = value["properties"], value["process"]
    require(value["state"] == "running" and value["native_observation"] is True and value["cgroup_populated"] is True
            and value["unit"] == intent["unit"] and value["unit_sha256"] == intent["unit_sha256"]
            and value["boot_id"] == boot == process["boot_id"], "Application running observation differs")
    require(properties["Id"] == intent["unit"] and properties["ActiveState"] == "active"
            and properties["SubState"] == "running" and properties["MainPID"] == str(process["pid"])
            and properties["LoadState"] == "loaded" and properties["NeedDaemonReload"] == "no"
            and not properties["DropInPaths"] and properties["User"] == properties["Group"] == "organism"
            and properties["NetworkNamespacePath"] == namespace_path, "Loaded application properties differ")
    limits = value["kernel_limits"]
    require(limits["memory.max"] == str(6144 * MIB) and limits["memory.swap.max"] == "0"
            and limits["pids.max"] == "128", "Observed application kernel limits differ")
    cpu = limits["cpu.max"].split()
    require(len(cpu) == 2 and all(v.isdigit() for v in cpu) and int(cpu[1]) > 0
            and int(cpu[0]) == 4 * int(cpu[1]), "Observed CPU quota differs")
    namespace(intent["network_namespace"], value["network_namespace"], process["pid"], expected_path=namespace_path)
    return process


def replay_readiness(evidence, seed, values, shared, release, *, trained=False, namespace_path=NS_PATH,
                     state_prefix="seed-evidence/state/", work_prefix="seed-evidence/work/"):
    observations = {kind: value["record"]["observation"] for kind, value in values.items()}
    intent, checked = observations["intent"], observations["check_intent"]
    token, individual = shared["process_token"], shared["individual_id"]
    for item in (intent, checked):
        require(sha(item["unit_text"].encode()) == item["unit_sha256"], "Recorded unit text hash differs")
    expected_unit = "organism-" + values["intent"]["record"]["device_id"][:16] + "-" + individual + "-" + token + ".service"
    require(intent["unit"] == expected_unit and checked["unit"] == expected_unit[:-8] + "-check.service",
            "Application or checker unit belongs to another incarnation")
    process = running(observations["running"], intent, shared["boot_id"], namespace_path=namespace_path)
    for sample in (seed["application_observation"], observations["check_admission"]["application_observation"]):
        require(running(sample, intent, shared["boot_id"], namespace_path=namespace_path) == process,
                "Application changed during readiness")
    require(checked["application_process"] == process and checked["network_namespace"] == intent["network_namespace"],
            "Checker was bound to another application or namespace")
    require(seed["namespace"] == intent["network_namespace"], "Seed driver namespace differs")
    admission, completed = observations["check_admission"], observations["check_result"]
    require(admission["check_result_hash"] == values["check_result"]["digest"]
            and admission["native_observation"] is True and admission["qualification_passed"] is False
            and completed["passed"] is True and completed["native_observation"] is True and completed["failure"] is None,
            "Receiver did not admit a successful native checker")
    require(seed["ready_receipt"]["ready_evidence"] == values["check_admission"]["digest"],
            "Peer readiness receipt differs from receiver admission")
    challenge = checked["challenge"]
    capture = state_prefix + token + "-check/"
    raw_challenge = evidence.read(capture + "challenge.json", 16384)
    stdout = evidence.read(capture + "stdout", 32768)
    stderr = evidence.read(capture + "stderr")
    require(raw_challenge == canonical(challenge) and not stderr
            and sha(stdout) == completed["stdout_sha256"] and sha(stderr) == completed["stderr_sha256"],
            "Owner checker captures differ")
    result = decode(stdout)
    require(result == completed["result"] and result["passed"] is True and result["native"] is True
            and result["failure"] is None and result["qualification_passed"] is False
            and result["receiver_readiness_acknowledged"] is False, "Checker result or scope differs")
    for key in ("challenge_id", "device_id", "individual_id", "process_token", "release_id"):
        require(result[key] == challenge[key], "Checker result identity differs from challenge")
    require(challenge["device_id"] == values["intent"]["record"]["device_id"]
            and challenge["individual_id"] == individual and challenge["process_token"] == token
            and challenge["release_id"] == release["release_id"] and challenge["native"] is True,
            "Checker challenge differs from the admitted seed")
    require(result["challenge_sha256"] == sha(raw_challenge)
            and result["checker_sha256"] == checked["checker_sha256"] == release["checker_sha256"],
            "Checker implementation differs from the reviewed capsule")
    for name in ("model_sha256", "model_bytes", "runtime_sha256"):
        require(challenge[name] == release[name] == intent["readiness"][name],
                "Readiness used an unexpected model or runtime")
    start, end, ready = moment(result["started_at"]), moment(result["completed_at"]), moment(seed["ready_observed_at"])
    require(challenge["issued_unix_seconds"] <= start <= end <= ready < challenge["expires_unix_seconds"]
            <= challenge["issued_unix_seconds"] + 240
            and number(result["elapsed_seconds"]) and abs(result["elapsed_seconds"] - (end - start)) < 1
            and checked["issued_clock"] <= seed["ready_observed_clock"] < checked["deadline_clock"],
            "Readiness was observed outside its challenge lifetime")
    worker = completed["observation"]
    props = worker["properties"]
    require(worker["state"] == "complete" and worker["native_observation"] is True
            and worker["cgroup_populated"] is False and worker["unit"] == checked["unit"]
            and worker["unit_sha256"] == checked["unit_sha256"]
            and worker["boot_id"] == shared["boot_id"]
            and props["MainPID"] == "0" and props["ExecMainCode"] == "1" and props["ExecMainStatus"] == "0"
            and props["Result"] == "success" and props["NetworkNamespacePath"] == namespace_path
            and props["User"] == props["Group"] == "organism"
            and props["Type"] == "oneshot" and props["RemainAfterExit"] == "yes"
            and props["MemoryMax"] == str(256 * MIB) and props["MemorySwapMax"] == "0"
            and props["TasksMax"] == "16" and props["CPUQuotaPerSecUSec"] in ("1s", "1000000"),
            "Checker process did not complete as declared")
    identity(props["InvocationID"], 32)
    require(all(props[key].isdigit() for key in ("ExecMainPID", "ExecMainStartTimestampMonotonic",
                                                 "ExecMainExitTimestampMonotonic")), "Checker execution timestamps are missing")
    execution_start, execution_end = (int(props[key]) for key in
                                      ("ExecMainStartTimestampMonotonic", "ExecMainExitTimestampMonotonic"))
    require(int(props["ExecMainPID"]) > 1 and int(props["ExecMainPID"]) != process["pid"]
            and 0 < execution_start <= execution_end <= execution_start + 245_000_000,
            "Checker native execution interval differs")
    if "check_running" in observations:
        earlier = observations["check_running"]
        require(earlier["pid"] == int(props["ExecMainPID"]) and earlier["boot_id"] == shared["boot_id"]
                and earlier["invocation_id"] == props["InvocationID"], "Checker restarted before completion")
        namespace(intent["network_namespace"], earlier["network_namespace"], earlier["pid"],
                  expected_path=namespace_path)
    work = work_prefix + individual + "/"
    target = work + "readiness/" + token + "/"
    require(evidence.json(target + "result.json") == result, "Application-owned result differs from owner capture")
    names = {"challenge.json", "application-before.json", "application-after.json", "observation.json", "reply.json", "job.json"}
    require(set(result["evidence_files"]) == names, "Successful checker lacks its complete evidence set")
    for name, digest in result["evidence_files"].items():
        require(sha(evidence.read(target + name, 2 * MIB)) == digest, "Checker evidence file hash differs")
    require(evidence.read(target + "challenge.json", 16384) == raw_challenge, "Checker challenge copy differs")
    raw = evidence.read(target + "observation.json", 8192)
    source = decode(raw)
    expected_value = int(identity(challenge["challenge_id"], 32)[:8], 16) % 61 - 10
    require(source["device_id"] == "readiness-" + challenge["challenge_id"] and source["unit"] == "C"
            and source["value"] == expected_value and source["normal_range_celsius"] == [10, 40]
            and source["maximum_observation_age_seconds"] == 240
            and evidence.read(work + "equipment/" + source["device_id"] + ".json", 8192) == raw,
            "Diagnostic input differs from the public challenge")
    job = evidence.json(target + "job.json", 2 * MIB)
    reply = evidence.json(target + "reply.json", 8192)
    require(reply["status"] == "finished" and reply["trace_sha256"] == job["digest"]
            and reply["request_id"] == challenge["challenge_id"], "Application reply differs from diagnostic")
    trace = job["record"]["trace"]
    require(job["record"]["device_id"] == source["device_id"] and job["record"]["status"] == "finished"
            and job["record"]["failure"] is None and job["record"]["independently_verified"] is False,
            "Diagnostic job contains a failure or an inconsistent identity")
    require(1 <= len(trace["model_calls"]) <= 4 and len(trace["model_streams"]) == len(trace["model_calls"])
            and 3 <= len(trace["tool_events"]) <= 8, "Diagnostic has no complete literal model trace")
    for request in trace["http_requests"]:
        require(request["body"]["model"] == "organism-" + token and request["body"]["stream"] is True,
                "Diagnostic used another inference incarnation")
    case = {"job": job, "request_id": challenge["challenge_id"], "observation": source, "evaluation": {"passed": True}}
    diagnostic = inspect_diagnostic(case, raw, {"case_id": source["device_id"], "passed": True},
                                    {"completed_at": result["completed_at"]})
    require(diagnostic["task_passed"] and diagnostic["literal_streams_present"]
            and diagnostic["malformed_streamed_tool_calls"] == 0, "Independent diagnostic replay failed")
    require(result["verification"] == {
        "model_calls": len(trace["model_calls"]), "tool_calls": len(trace["tool_events"]),
        "trace_sha256": job["digest"], "observation_sha256": sha(raw),
        "status": diagnostic["expected_status"], "temperature_celsius": diagnostic["expected_celsius"],
        "literal_tool_arguments_verified": True,
    }, "Checker verification differs from independent diagnostic replay")
    for health in (result["initial_health"], result["final_health"]):
        require(health["status"] == "serving" and health["inference_ready"] is True
                and health["start_token"] == token and health["native"] is True
                and health["trained_for_organism"] is trained, "Checker health belongs to another application")
    app_receipts = [evidence.json(target + name) for name in ("application-before.json", "application-after.json")]
    for app in app_receipts:
        require(app["device_id"] == challenge["device_id"] and app["individual_id"] == individual
                and app["release_id"] == release["release_id"] and app["start_token"] == token
                and app["native"] is True and app["trained_for_organism"] is trained
                and "shutdown" not in app and not app.get("failure")
                and app["useful_readiness_verified"] is False and app["autonomous_reproduction_verified"] is False
                and app["qualification_passed"] is False, "Application receipt identity or lifetime differs")
        inference = app["inference"]
        command = inference["command"]
        require(inference["native_execution"] is True and inference["alias"] == "organism-" + token
                and inference["pid"] == result["inference_pid"] and command[0] == challenge["runtime_path"]
                and inference["trained_for_organism"] is trained,
                "Application inference command differs")
        for flag, wanted in {"--model": challenge["model_path"], "--alias": "organism-" + token,
                             "--host": "127.0.0.1", "--ctx-size": "4096", "--parallel": "1",
                             "--gpu-layers": "0", "--device": "none", "--threads": "4", "--threads-batch": "4"}.items():
            require(command.count(flag) == 1 and command[command.index(flag) + 1] == wanted,
                    "Inference command differs from its native CPU profile")
        require(all(command.count(flag) == 1 for flag in ("--offline", "--no-kv-offload", "--no-op-offload")),
                "Inference command omits the local CPU/offline controls")
        props = inference["properties"]
        require(props["model_alias"] == inference["alias"] and props["total_slots"] == 1
                and props["default_generation_settings"]["n_ctx"] == 4096
                and props["ui"] is False and props["cors_proxy_enabled"] is False,
                "Observed inference properties differ from the declared model service")
    require(app_receipts[0]["inference"] == app_receipts[1]["inference"], "Inference restarted during readiness")
    namespace(intent["network_namespace"], seed["inference_namespace"], result["inference_pid"],
              expected_path=namespace_path)
    return {"diagnostic": diagnostic, "unit": expected_unit, "checker_unit": checked["unit"],
            "process_token": token, "inference_pid": result["inference_pid"]}


def absence(value, unit, *, host=False):
    props = value["properties"]
    require(value["unit"] == unit and props["Id"] == unit and props["MainPID"] == "0"
            and props["ActiveState"] in ("inactive", "failed") and props["Job"] in ("", "0"),
            "Stopped unit still has a process, job or different identity")
    if host:
        require(value["absence_verified"] is True and value["entire_cgroup_empty"] is True
                and (value["show_return_code"] == 0 or value["show_return_code"] == 4 and props["LoadState"] == "not-found"),
                "Host cleanup did not establish complete cgroup absence")
    else:
        require(value["state"] == "absent" and value["cgroup_populated"] is False
                and value["native_observation"] is True, "Receiver cleanup lacks native process absence")


def cleanup_and_memory(evidence, host, seed, values, replay):
    cleanup = host["cleanup"]
    require(cleanup["passed"] is True and cleanup["namespace_removed"] is True
            and cleanup["remaining_namespace_pids"] == [], "Native namespace cleanup is incomplete")
    absence(cleanup["driver"], DRIVER, host=True)
    units = {item["unit"]: item for item in cleanup["application_and_checker"]}
    require(len(cleanup["application_and_checker"]) == 2
            and set(units) == {replay["unit"], replay["checker_unit"]}, "Host cleanup omitted an owned service")
    for unit, item in units.items():
        absence(item, unit, host=True)
    exited = values["exit"]["record"]["observation"]
    absence(exited, replay["unit"])
    absence(exited["readiness_observation"], replay["checker_unit"])
    retirement = seed["retirement"]
    # The receiving loop can finish the stop before the driver's final poll.
    # An idle poll then has no new evidence hash; require the immutable exit
    # record above and the final retired state instead of inventing a new exit.
    retired = (retirement["state"] == "absent" and retirement["process_exit_verified"] is True
               and retirement["evidence_hash"] == values["exit"]["digest"])
    if retirement["state"] == "idle":
        retired = retirement["owner_mode"] == "retired" and retirement["restart_required"] is False
    require(seed["final_owner_mode"] == "retired" and seed["final_process_token"] is None and retired,
            "Owner retirement evidence differs")
    samples = evidence.json("memory-samples.json", 8 * MIB)
    require(isinstance(samples, list) and 2 <= len(samples) <= 1565, "Missing or excessive resource observations")
    previous, seen, peak, available = None, set(), {}, []
    for sample in samples:
        at = sample["monotonic_seconds"]
        require(number(at) and (previous is None or previous < at), "Memory observation clock is invalid")
        previous = at
        require(type(sample["MemTotal_kib"]) is int and 7 * MIB <= sample["MemTotal_kib"] <= 8 * MIB
                and type(sample["MemAvailable_kib"]) is int and 0 <= sample["MemAvailable_kib"] <= sample["MemTotal_kib"]
                and sample["SwapTotal_kib"] == sample["SwapFree_kib"] == 0, "Observed host memory profile differs")
        available.append(sample["MemAvailable_kib"])
        for unit, fields in sample["cgroups"].items():
            require(unit in {DRIVER, replay["unit"], replay["checker_unit"]}, "Resource sample includes an unrelated unit")
            if fields == {"observation": "one or more cgroup counters unavailable during sample"}:
                continue
            require(all(str(fields[key]).isdigit() for key in ("memory.current", "memory.peak", "memory.swap.current")),
                    "Invalid cgroup memory counter")
            require(int(fields["memory.swap.current"]) == 0, "Native service used swap")
            require(int(fields["memory.peak"]) >= int(fields["memory.current"]),
                    "Cgroup current memory exceeds its recorded peak")
            events = dict(line.split() for line in fields["memory.events"].splitlines())
            require(all(events.get(key) == "0" for key in ("oom", "oom_kill")), "Native service had an OOM event")
            seen.add(unit)
            peak[unit] = max(peak.get(unit, 0), int(fields["memory.peak"]))
    require({DRIVER, replay["unit"]} <= seen, "No usable driver/application memory observation")
    return {"sample_count": len(samples), "sampled_host_minimum_available_kib": min(available),
            "observed_cgroup_peak_bytes": peak, "whole_device_peak_measured": False,
            "limitation": "One-second samples and observed cgroup peaks do not measure every host transient."}


def inspect(directory, kit_directory, architecture):
    evidence, kit = Evidence(directory), Evidence(kit_directory)
    catalog, release = trusted_release(kit, architecture)
    require(evidence.read("input-catalog.json") == kit.read("catalog.json"), "Run used a different reviewed catalog")
    host, seed = evidence.json("host-receipt.json"), evidence.json("seed-receipt.json")
    for item in (host, seed):
        require(item["passed"] is True and item["qualification_passed"] is False
                and item["autonomous_reproduction_verified"] is False and item["architecture"] == architecture
                and item["release_id"] == release["release_id"] and not item.get("failure")
                and not item.get("cleanup_failure"), "Native seed run reports failure or excessive scope")
    require(host["service_return_code"] == 0 and host["seed_passed"] is True
            and not host.get("evidence_copy_failure") and seed["ready"] is True
            and seed["cleanup_verified"] is True and seed["receiver_loop_stopped"] is True
            and seed["seed_provisioned_by_owner"] is True, "Seed lifecycle did not complete")
    device, policy_id, bootstrap = owner_policy(evidence, catalog, architecture)
    peer = seed["ready_receipt"]
    require(peer["device_id"] == device and peer["operation_id"] == seed["operation_id"]
            and peer["release_id"] == release["release_id"] and peer["status"] == "ready",
            "Ready receipt belongs to another seed")
    values, shared = records(evidence, seed, device, policy_id, release)
    peer_audit = transport(evidence, seed, catalog, bootstrap)
    replay = replay_readiness(evidence, seed, values, shared, release)
    health = seed["application_health"]
    require(health["status"] == "serving" and health["inference_ready"] is True
            and health["native"] is True and health["trained_for_organism"] is False
            and health["start_token"] == shared["process_token"]
            and health["discovery_running"] is True and health["reproduction_running"] is True,
            "Current seed application health differs")
    memory = cleanup_and_memory(evidence, host, seed, values, replay)
    require(moment(host["started_at"]) <= moment(seed["ready_observed_at"]) <= moment(seed["completed_at"])
            <= moment(host["completed_at"]), "Host and seed lifetime ordering differs")
    return {"scope": "Independent reconstruction of captured native seed artifacts; external run provenance still required",
            "artifact_consistency_passed": True, "architecture": architecture, "release_id": release["release_id"],
            "device_id": device, "process_token": shared["process_token"],
            "diagnostic": replay["diagnostic"], "memory": memory, "transport": peer_audit,
            "owner_retirement_and_cleanup_consistent": True,
            "read_file_sha256": evidence.read_hashes, "reviewed_kit_sha256": sha(kit.read("kit.tar.gz", 2 * MIB)),
            "native_execution_verified": False, "external_run_provenance_required": True,
            "autonomous_reproduction_verified": False, "trained_for_organism": False, "qualification_passed": False,
            "auditor_sha256": sha(Path(__file__).read_bytes()),
            "semantic_auditor_sha256": sha(Path(inspect_diagnostic.__code__.co_filename).read_bytes()),
            "limitations": ["Captured host observations are trusted lab evidence, not hardware attestation.",
                           "Verify the external launch ledger, native instance identity and downloaded archive hash separately.",
                           "Publisher-signature and native library-loading evidence requires its separate review.",
                           "This owner-provisioned public startup smoke test is not a lineage or sealed quality trial."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--architecture", required=True, choices=("arm64", "x86_64"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = inspect(args.directory, args.kit, args.architecture)
    except Exception as exc:
        result = {"scope": "Independent native startup artifact audit failed; original inputs are preserved",
                  "artifact_consistency_passed": False, "failure": type(exc).__name__ + ": " + str(exc)[:512],
                  "native_execution_verified": False, "external_run_provenance_required": True,
                  "autonomous_reproduction_verified": False, "qualification_passed": False,
                  "auditor_sha256": sha(Path(__file__).read_bytes()),
                  "semantic_auditor_sha256": sha(Path(inspect_diagnostic.__code__.co_filename).read_bytes())}
    target = args.output or args.directory / "independent-startup-audit.json"
    with target.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({"artifact_consistency_passed": result["artifact_consistency_passed"], "native_execution_verified": False,
                      "external_run_provenance_required": True}))
    return 0 if result["artifact_consistency_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
