"""Owner-side execution receipts for the capsule's public smoke checker.

The loader knows the checker protocol and incarnation identity, not diagnostic
answers. Task checking remains in the signed capsule. This is a cooperative lab
process check, not attestation against a compromised application Unix account.
"""

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time
import uuid

from .capsule import canonical
from . import device_resources, runtime_storage
from .runtime_environment import library_environment
from .network_namespace import namespace_path, verify_process_namespace
from .systemd_supervision import (
    ObservationUncertain, SHOW_FIELDS, SubmissionUncertain, SupervisionError,
    description, literal_path, parse_show, process_start_ticks, quoted, safe_path, unit_name,
)


CHECK_SECONDS = 240
CHECK_MEMORY = 256 * 1024**2
CHECK_TASKS = 16
MAX_OUTPUT = 32768
MAX_CAPTURE = 1024**2
CHECK_FIELDS = SHOW_FIELDS + (
    "Type", "RemainAfterExit", "ExecMainPID", "ExecMainCode", "ExecMainStatus",
    "ExecMainStartTimestampMonotonic", "ExecMainExitTimestampMonotonic", "TimeoutStartUSec",
    "StandardInput", "StandardOutput", "StandardError",
)
RESULT_FIELDS = {
    "schema", "challenge_sha256", "challenge_id", "device_id", "individual_id",
    "process_token", "release_id", "native", "passed", "failure", "started_at",
    "checker_sha256", "receiver_readiness_acknowledged", "qualification_passed",
    "elapsed_seconds", "completed_at", "evidence_files",
}
SUCCESS_FIELDS = {"verification", "initial_health", "final_health", "inference_pid"}


def check_name(birth):
    return unit_name(birth).removesuffix(".service") + "-check.service"


def check_description(birth):
    return description(birth) + " purpose=readiness"


def require(condition, message):
    if not condition:
        raise ObservationUncertain(message)


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate checker result field")
            result[key] = value
        return result

    def invalid(_):
        raise ObservationUncertain("Nonfinite checker result")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def private_read(path, maximum, owner_uid):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        entry = os.fstat(source.fileno())
        require(stat.S_ISREG(entry.st_mode) and entry.st_uid == owner_uid and entry.st_nlink == 1
                and not entry.st_mode & 0o077 and entry.st_size <= maximum,
                "Invalid owner-private checker capture")
        raw = source.read(maximum + 1)
    require(len(raw) <= maximum, "Checker capture exceeds its bound")
    return raw


def publish(path, raw, mode=0o600):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "wb") as target:
        target.write(raw)
        target.flush()
        os.fchmod(target.fileno(), mode)
        os.fsync(target.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def render_check_unit(birth, app, target, *, state_directory, policy_id, guard_python, guard_script):
    resource_slice = device_resources.validate_slice(app.get("resource_slice"))
    declaration = app["readiness"]
    guard = (str(safe_path(guard_python)), "-I", str(safe_path(guard_script)),
             "--state", str(safe_path(state_directory)), "--policy-id", policy_id,
             "--device-id", birth["device_id"], "--operation-id", birth["operation_id"],
             "--process-token", birth["process_token"], "--purpose", "readiness")
    storage = app.get("runtime_storage")
    command = ("/usr/bin/env", "-i", "PATH=/usr/bin:/bin",
               *(key + "=" + value for key, value in runtime_storage.environment(storage).items()),
               "LC_ALL=C.UTF-8",
               "PYTHONDONTWRITEBYTECODE=1",
               *(key + "=" + value for key, value in library_environment(app["library_paths"]).items()),
               app["interpreter"], "-I", "-S", declaration["checker_path"],
               "--work-directory", app["working_directory"], "--challenge-stdin")
    # Requisite does not launch its target. PartOf propagates owner stop. An
    # unexpected application exit is handled by reconciliation and the checker.
    lines = [
        "[Unit]", "Description=" + check_description(birth),
        "Requisite=" + unit_name(birth), "After=" + unit_name(birth), "PartOf=" + unit_name(birth),
        "[Service]", "Type=oneshot", "RemainAfterExit=yes", "Restart=no",
        "Slice=" + (resource_slice or "system.slice"),
        "ExecCondition=+:" + " ".join(quoted(arg) for arg in guard),
        "ExecStart=:" + " ".join(quoted(arg) for arg in command),
        "User=" + app["user"], "Group=" + app["group"],
        "WorkingDirectory=" + literal_path(app["working_directory"]),
        "UMask=0077", "NoNewPrivileges=yes", "CapabilityBoundingSet=", "PrivateTmp=yes", "PrivateDevices=yes",
        "ProtectSystem=strict", "ProtectHome=yes", "ProtectControlGroups=yes", "ProtectKernelTunables=yes",
        "ProtectKernelModules=yes", "RestrictSUIDSGID=yes", "RestrictRealtime=yes", "LockPersonality=yes",
        "Delegate=no", "RestrictAddressFamilies=AF_UNIX", "IPAddressDeny=any",
        "ReadWritePaths=" + quoted(app["working_directory"]),
        "ReadOnlyPaths=" + " ".join(quoted(path) for path in app["readonly_paths"]),
        "MemoryAccounting=yes", "MemoryMax=" + str(CHECK_MEMORY), "MemorySwapMax=0",
        "CPUQuota=100%", "TasksMax=" + str(CHECK_TASKS), "LimitFSIZE=" + str(MAX_CAPTURE),
        "KillMode=control-group", "SendSIGKILL=yes", "TimeoutStartSec=245", "TimeoutStopSec=10",
        # These directives parse a literal file: prefix, not a quoted word
        # list. Retain internal spaces and escape only systemd specifiers.
        "StandardInput=file:" + str(safe_path(target / "challenge.json")).replace("%", "%%"),
        "StandardOutput=file:" + str(safe_path(target / "stdout")).replace("%", "%%"),
        "StandardError=file:" + str(safe_path(target / "stderr")).replace("%", "%%"), "",
    ]
    if app.get("network_namespace") is not None:
        lines.insert(-1, "NetworkNamespacePath=" + literal_path(namespace_path(app["network_namespace"]["path"])))
        lines.insert(-1, "RestrictNamespaces=yes")
    if storage is not None:
        lines = [line for line in lines if not line.startswith("PrivateTmp=")]
        lines[-1:-1] = runtime_storage.directives(storage, quoted)
    return "\n".join(lines)


def validate_result(raw, intent, *, observed_unix_seconds=None):
    """Validate the protocol, never reproduce the carried diagnostic rules."""
    result = decode(raw)
    require(isinstance(result, dict) and RESULT_FIELDS <= set(result)
            and set(result) <= RESULT_FIELDS | SUCCESS_FIELDS, "Checker result schema differs")
    challenge = intent["challenge"]
    expected = {"schema": "organism-useful-smoke-v1",
                "challenge_sha256": hashlib.sha256(canonical(challenge)).hexdigest(),
                "checker_sha256": intent["checker_sha256"],
                **{key: challenge[key] for key in
                   ("challenge_id", "device_id", "individual_id", "process_token", "release_id")}}
    require(all(result[key] == value for key, value in expected.items()), "Checker result identity differs")
    require(type(result["native"]) is bool and result["native"] == challenge["native"]
            and type(result["passed"]) is bool and result["receiver_readiness_acknowledged"] is False
            and result["qualification_passed"] is False, "Checker claims exceed the public smoke protocol")
    require(type(result["elapsed_seconds"]) in (int, float) and math.isfinite(result["elapsed_seconds"])
            and 0 <= result["elapsed_seconds"] <= CHECK_SECONDS, "Checker duration exceeds its bound")
    started, completed = (datetime.fromisoformat(result[key]) for key in ("started_at", "completed_at"))
    require(started.tzinfo is not None and completed.tzinfo is not None
            and challenge["issued_unix_seconds"] <= started.timestamp() <= completed.timestamp()
            <= challenge["expires_unix_seconds"], "Checker result is outside its challenge lifetime")
    require(abs(result["elapsed_seconds"] - (completed - started).total_seconds()) < 1,
            "Checker elapsed duration differs from its timestamps")
    if observed_unix_seconds is not None:
        require(completed.timestamp() <= observed_unix_seconds + 1, "Checker completion is in the future")
    files = result["evidence_files"]
    names = {"challenge.json", "application-before.json", "application-after.json", "job.json", "reply.json",
             "observation.json"}
    require(isinstance(files, dict) and set(files) <= names
            and all(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) for value in files.values()),
            "Checker evidence references differ")
    if result["passed"]:
        require(set(files) == names and SUCCESS_FIELDS <= set(result) and result["failure"] is None
                and files["challenge.json"] == expected["challenge_sha256"],
                "Successful checker result lacks its evidence references")
    else:
        require(isinstance(result["failure"], str) and 0 < len(result["failure"]) <= 1024,
                "Failed checker result lacks its reason")
    return result


class ReadinessBackend:
    """One asynchronous, persisted checker attempt per application incarnation."""

    def __init__(self, backend, *, wall_clock=time.time):
        self.backend, self.state, self.host, self.records = backend, backend.state, backend.host, backend.records
        self.wall_clock = wall_clock

    def target(self, birth):
        # The existing state directory is already owner-private and persistent.
        return self.state.directory / (birth["process_token"] + "-check")

    def _file(self, birth):
        return self.host.unit_root / check_name(birth)

    def _intent(self, birth):
        record = self.records.get(birth, "check_intent")
        return record["record"]["observation"] if record else None

    def _owned_unit(self, birth, intent):
        path = self._file(birth)
        entry = path.lstat()
        require(stat.S_ISREG(entry.st_mode) and entry.st_uid == self.host.owner_uid
                and entry.st_nlink == 1 and not entry.st_mode & 0o022 and entry.st_size <= 32768,
                "Checker unit is not owner-controlled")
        text = path.read_text()
        require(text == intent["unit_text"] and hashlib.sha256(text.encode()).hexdigest() == intent["unit_sha256"],
                "Checker unit differs from its persisted intent")
        return path

    def start(self, birth):
        """Caller holds Supervisor's lock; intent precedes every side effect."""
        require(self._intent(birth) is None, "This incarnation already has a checker attempt")
        stored = self.records.get(birth, "intent")
        require(stored is not None, "Application start intent is missing")
        app = stored["record"]["observation"]
        if not app.get("readiness"):
            return False
        self.backend.resource_observation(app.get("resource_slice"))
        self.backend.network_observation(app.get("network_budget"), app.get("network_namespace"))
        self.backend.storage_observation(app.get("storage_profile"), app.get("storage_binding"), app.get("storage_layout"))
        declaration = app["readiness"]
        checker = Path(declaration["checker_path"])
        entry = checker.lstat()
        require(stat.S_ISREG(entry.st_mode) and entry.st_uid == self.host.owner_uid
                and entry.st_nlink == 1 and not entry.st_mode & 0o222 and entry.st_size <= 65536,
                "Installed checker is not an immutable owner file")
        require(hashlib.sha256(checker.read_bytes()).hexdigest() == declaration["checker_sha256"],
                "Installed checker differs from the prepared source")
        require(hashlib.sha256(self.backend.guard_script.read_bytes()).hexdigest() == app["guard_sha256"],
                "Owner guard changed after application launch")
        now, clock = self.wall_clock(), self.state.clock()
        target = self.target(birth)
        challenge = {"schema": "organism-readiness-challenge-v1", "challenge_id": uuid.uuid4().hex,
                     **{key: birth[key] for key in ("device_id", "individual_id", "process_token", "release_id")},
                     **{key: value for key, value in declaration.items() if not key.startswith("checker_")},
                     "native": not self.host.is_fixture, "issued_unix_seconds": now,
                     "expires_unix_seconds": now + CHECK_SECONDS}
        text = render_check_unit(birth, app, target, state_directory=self.state.directory,
                                 policy_id=self.state.policy.policy_id,
                                 guard_python=self.backend.guard_python, guard_script=self.backend.guard_script)
        intent = {"unit": check_name(birth), "unit_text": text,
                  "unit_sha256": hashlib.sha256(text.encode()).hexdigest(), "challenge": challenge,
                  "checker_sha256": declaration["checker_sha256"], "checker_path": str(checker),
                  "application_process": birth["process_observation"], "issued_clock": clock,
                  "deadline_clock": clock + CHECK_SECONDS, "user": app["user"], "group": app["group"],
                  "network_namespace": app.get("network_namespace"),
                  "network_budget": app.get("network_budget"),
                  "storage_profile": app.get("storage_profile"), "storage_binding": app.get("storage_binding"),
                  "storage_layout": app.get("storage_layout"),
                  "runtime_storage": app.get("runtime_storage"),
                  "resource_slice": app.get("resource_slice")}
        self.records.put(birth, "check_intent", intent)
        target.mkdir(mode=0o700)
        publish(target / "challenge.json", canonical(challenge))
        publish(target / "stdout", b"")
        publish(target / "stderr", b"")
        publish(self._file(birth), text.encode(), 0o644)
        try:
            if self.host.run(["systemctl", "daemon-reload"]).returncode:
                raise SubmissionUncertain("Checker reload did not confirm success")
            if self.host.run(["systemctl", "start", check_name(birth), "--no-block"]).returncode:
                raise SubmissionUncertain("Checker submission did not confirm success")
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SubmissionUncertain("Checker submission observation was interrupted") from exc
        return True

    def inspect(self, birth):
        intent = self._intent(birth)
        if intent is None:
            return {"state": "not_attempted", "cgroup_populated": False,
                    "native_observation": not self.host.is_fixture}
        name = check_name(birth)
        try:
            response = self.host.run(["systemctl", "show", name, "--no-pager",
                                      "--property=" + ",".join(CHECK_FIELDS)])
            value = parse_show(response.stdout)
            require(all(key in value for key in CHECK_FIELDS), "Checker manager observation is incomplete")
            require(value["Id"] == name and value["MainPID"].isdigit(), "Checker service identity differs")
            resource_slice = device_resources.validate_slice(intent.get("resource_slice"))
            populated = self.backend._group_population(name, value["ControlGroup"], resource_slice=resource_slice)
            base = {"unit": name, "boot_id": self.host.boot_id, "properties": value,
                    "cgroup_populated": populated, "native_observation": not self.host.is_fixture}
            if (not os.path.lexists(self._file(birth)) and value["LoadState"] == "not-found"
                    and value["ActiveState"] == "inactive" and value["MainPID"] == "0" and not populated
                    and value["Job"] in ("", "0") and not value["FragmentPath"] and not value["DropInPaths"]):
                return {**base, "state": "absent", "unit_sha256": None}
            require(response.returncode == 0, "Checker manager observation failed")
            path = self._owned_unit(birth, intent)
            require(value["Description"] == check_description(birth) and value["FragmentPath"] == str(path)
                    and not value["DropInPaths"] and value["NeedDaemonReload"] == "no"
                    and value["LoadState"] == "loaded", "Loaded checker configuration differs")
            base["unit_sha256"] = intent["unit_sha256"]
            if not populated and value["MainPID"] == "0" and value["Job"] in ("", "0"):
                if value["ActiveState"] in ("failed", "inactive"):
                    return {**base, "state": "absent"}
                if value["ActiveState"] == "active" and value["SubState"] == "exited":
                    self._effective(value, intent)
                    base["aggregate_resources"] = self.backend.resource_observation(resource_slice)
                    base["network_budget"] = self.backend.network_observation(
                        intent.get("network_budget"), intent.get("network_namespace"))
                    base["device_storage"] = self.backend.storage_observation(
                        intent.get("storage_profile"), intent.get("storage_binding"), intent.get("storage_layout"))
                    numbers = ("ExecMainPID", "ExecMainCode", "ExecMainStatus",
                               "ExecMainStartTimestampMonotonic", "ExecMainExitTimestampMonotonic")
                    require(all(value[key].isdigit() for key in numbers), "Checker exit receipt is incomplete")
                    require(re.fullmatch("[0-9a-f]{32}", value["InvocationID"])
                            and int(value["ExecMainPID"]) > 1 and value["ExecMainCode"] == "1"
                            and value["ExecMainStatus"] == "0" and value["Result"] == "success",
                            "Checker did not exit successfully")
                    start, end = (int(value[key]) / 1_000_000 for key in numbers[-2:])
                    # Linux CLOCK_BOOTTIME and systemd CLOCK_MONOTONIC are not
                    # interchangeable across suspend. Only compare this pair.
                    require(0 < start <= end and end - start <= 245, "Checker manager duration differs")
                    running = self.records.get(birth, "check_running")
                    if intent.get("storage_profile") is not None:
                        require(running is not None and running["record"]["observation"].get(
                            "runtime_storage", {}).get("named_write_destinations_verified") is True,
                            "Checker storage destinations were not observed while running")
                    if running:
                        previous = running["record"]["observation"]
                        require(previous["invocation_id"] == value["InvocationID"]
                                and previous["pid"] == int(value["ExecMainPID"]),
                                "Completed checker differs from its observed process")
                    return {**base, "state": "complete"}
            if populated and value["MainPID"].isdigit() and int(value["MainPID"]) > 1:
                self._effective(value, intent)
                base["aggregate_resources"] = self.backend.resource_observation(resource_slice)
                base["network_budget"] = self.backend.network_observation(
                    intent.get("network_budget"), intent.get("network_namespace"))
                base["device_storage"] = self.backend.storage_observation(
                    intent.get("storage_profile"), intent.get("storage_binding"), intent.get("storage_layout"))
                pid = int(value["MainPID"])
                require(re.fullmatch("[0-9a-f]{32}", value["InvocationID"]), "Checker invocation is missing")
                ticks = process_start_ticks((self.host.proc_root / str(pid) / "stat").read_text(), pid)
                membership = (self.host.proc_root / str(pid) / "cgroup").read_text().splitlines()
                require("0::" + device_resources.service_cgroup(name, resource_slice) in membership,
                        "Checker PID is outside its cgroup")
                group = self.host.cgroup_root / device_resources.service_cgroup(name, resource_slice).lstrip("/")
                limits = {key: (group / key).read_text().strip()
                          for key in ("memory.max", "memory.swap.max", "pids.max", "cpu.max")}
                cpu = limits["cpu.max"].split()
                require(limits["memory.max"] == str(CHECK_MEMORY) and limits["memory.swap.max"] == "0"
                        and limits["pids.max"] == str(CHECK_TASKS) and len(cpu) == 2
                        and all(part.isdigit() for part in cpu) and int(cpu[0]) == int(cpu[1]) > 0,
                        "Checker kernel resource limits differ")
                observed = {"pid": pid, "invocation_id": value["InvocationID"], "start_ticks": ticks,
                            "boot_id": self.host.boot_id}
                if intent.get("network_namespace") is not None:
                    observed["network_namespace"] = verify_process_namespace(
                        intent["network_namespace"], pid, proc_root=self.host.proc_root, owner_uid=self.host.owner_uid)
                if intent.get("storage_profile") is not None:
                    observed["runtime_storage"] = runtime_storage.observe_process(
                        self.host, pid, intent["runtime_storage"], file_max=MAX_CAPTURE,
                        output_paths={name: self.target(birth) / name for name in ("stdout", "stderr")})
                previous = self.records.get(birth, "check_running")
                if previous is None:
                    self.records.put(birth, "check_running", observed)
                else:
                    earlier = previous["record"]["observation"]
                    require(all(earlier[key] == observed[key] for key in
                                ("pid", "invocation_id", "start_ticks", "boot_id")),
                            "Running checker incarnation changed")
                    if intent.get("storage_profile") is not None:
                        before_outputs = earlier["runtime_storage"]["outputs"]
                        after_outputs = observed["runtime_storage"]["outputs"]
                        require(all(all(before_outputs[name][key] == after_outputs[name][key]
                                        for key in ("path", "device", "inode"))
                                    for name in ("stdout", "stderr")), "Running checker capture identity changed")
            return {**base, "state": "pending"}
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            if isinstance(exc, SupervisionError):
                raise
            raise ObservationUncertain("Checker process observation is unavailable") from exc

    @staticmethod
    def _effective(value, intent):
        expected = {"Type": "oneshot", "RemainAfterExit": "yes", "Restart": "no", "KillMode": "control-group",
                    "User": intent["user"], "Group": intent["group"], "MemoryMax": str(CHECK_MEMORY),
                    "MemorySwapMax": "0", "TasksMax": str(CHECK_TASKS), "NoNewPrivileges": "yes",
                    "ProtectSystem": "strict", "Delegate": "no",
                    "Slice": intent.get("resource_slice") or "system.slice",
                    "NetworkNamespacePath": intent["network_namespace"]["path"] if intent.get("network_namespace") else "",
                    "StandardInput": "file", "StandardOutput": "file", "StandardError": "file"}
        require(all(value[key] == wanted for key, wanted in expected.items()), "Effective checker restrictions differ")
        require(value["CPUQuotaPerSecUSec"] in ("1s", "1000000")
                and value["TimeoutStartUSec"] in ("4min 5s", "245000000", "245s"),
                "Effective checker CPU or time bound differs")

    def stop(self, birth):
        """Do not release the application claim until the checker is empty."""
        intent = self._intent(birth)
        if intent is None:
            return self.inspect(birth)
        if not os.path.lexists(self._file(birth)):
            observation = self.inspect(birth)
            require(observation["state"] == "absent", "Missing checker unit does not prove absence")
            return observation
        self._owned_unit(birth, intent)
        try:
            result = self.host.run(["systemctl", "stop", check_name(birth)], timeout=25)
            if result.returncode:
                raise SubmissionUncertain("Checker stop did not confirm success")
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SubmissionUncertain("Checker stop observation was interrupted") from exc
        observation = self.inspect(birth)
        require(observation["state"] == "absent", "Checker has not fully stopped")
        return observation

    def result(self, birth):
        """Poll once; completed evidence is immutable, including failures."""
        previous = self.records.get(birth, "check_result")
        if previous is not None:
            return previous
        intent = self._intent(birth)
        if intent is None:
            return None
        expired = self.state.clock() >= intent["deadline_clock"] or self.wall_clock() >= intent["challenge"]["expires_unix_seconds"]
        observation = self.stop(birth) if expired else self.inspect(birth)
        if observation["state"] == "pending":
            return None
        receipt = {"passed": False, "native_observation": observation["native_observation"],
                   "observation": observation, "failure": None, "qualification_passed": False}
        try:
            require(not expired and observation["state"] == "complete", "Checker expired or did not complete")
            if intent.get("storage_profile") is not None:
                running = self.records.get(birth, "check_running")["record"]["observation"]
                for name, expected in running["runtime_storage"]["outputs"].items():
                    require(runtime_storage.identity((self.target(birth) / name).lstat())
                            == {key: expected[key] for key in ("device", "inode")},
                            "Completed checker capture identity changed")
            raw = private_read(self.target(birth) / "stdout", MAX_OUTPUT, self.host.owner_uid)
            stderr = private_read(self.target(birth) / "stderr", MAX_CAPTURE, self.host.owner_uid)
            require(not stderr, "Checker wrote unexpected diagnostic output")
            require(private_read(self.target(birth) / "challenge.json", 16384, self.host.owner_uid)
                    == canonical(intent["challenge"]), "Owner challenge capture changed")
            checker = Path(intent["checker_path"])
            info = checker.lstat()
            require(stat.S_ISREG(info.st_mode) and info.st_uid == self.host.owner_uid and info.st_nlink == 1
                    and not info.st_mode & 0o222 and info.st_size <= 65536
                    and hashlib.sha256(checker.read_bytes()).hexdigest() == intent["checker_sha256"],
                    "Checker source changed before completion")
            result = validate_result(raw, intent, observed_unix_seconds=self.wall_clock())
            receipt.update(passed=result["passed"], failure=result["failure"], result=result,
                           stdout_sha256=hashlib.sha256(raw).hexdigest(),
                           stderr_sha256=hashlib.sha256(stderr).hexdigest())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            receipt["failure"] = (type(exc).__name__ + ": " + str(exc))[:512]
        self.records.put(birth, "check_result", receipt)
        return self.records.get(birth, "check_result")

    def fresh(self, birth):
        intent = self._intent(birth)
        return bool(intent and intent["application_process"] == birth["process_observation"]
                    and intent["issued_clock"] <= self.state.clock() < intent["deadline_clock"]
                    and intent["challenge"]["issued_unix_seconds"] <= self.wall_clock()
                    < intent["challenge"]["expires_unix_seconds"])
