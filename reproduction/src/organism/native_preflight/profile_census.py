"""External-owner orchestration of the bounded freeze guard and file census."""

import hashlib
import json
import os
from pathlib import Path
import select
import re
import stat
import subprocess
import sys
import time

from .. import device_resources, process_storage


REQUEST = "census-ready-request.json"
RESPONSE = "census-ready-response.json"
REPORT = "census-ready-report.json"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def frame(process, receipt, *, timeout):
    require(select.select([process.stdout], [], [], timeout)[0], "Timed out waiting for the census guard")
    raw = process.stdout.readline(4097)
    require(raw.endswith(b"\n") and len(raw) <= 4096, "Guard response is missing or oversized")
    value = json.loads(raw)
    receipt["guard_frames"].append(value)
    return value


def capture(host, *, allowed_devices, receipt, collector=process_storage.observe_frozen):
    """Capture metadata and record cleanup; never substitute for full containment."""
    require(sys.platform == "linux" and os.geteuid() == 0 and host.is_fixture is False,
            "A native owner outside the device slice is required")
    group = device_resources._group(host, device_resources.CGROUP)
    require(Path(host.proc_root) == Path("/proc") and Path(host.cgroup_root) == Path("/sys/fs/cgroup"),
            "Owner capture requires the actual native kernel")
    membership = Path("/proc/self/cgroup").read_text().splitlines()
    require(len(membership) == 1 and membership[0].startswith("0::")
            and not (membership[0][3:] == device_resources.CGROUP
                     or membership[0][3:].startswith(device_resources.CGROUP + "/")),
            "The census owner cannot freeze itself")
    entry = group.stat()
    guard = Path(__file__).with_name("freeze_guard.py")
    command = [sys.executable, "-I", "-B", str(guard), "--device", str(entry.st_dev),
               "--inode", str(entry.st_ino), "--lease-seconds", "10"]
    receipt.update(schema="organism-owner-process-census-v1", guard_command=command,
                   guard_frames=[], guard_exit_verified=False, thaw_verified=False,
                   qualification_passed=False)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True, close_fds=True, bufsize=0,
                               env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"})
    receipt["guard_pid"] = process.pid
    failure = None
    try:
        armed = frame(process, receipt, timeout=2)
        require(armed["phase"] == "armed" and armed["pid"] == process.pid
                and armed["owner_pid"] == os.getpid() and armed["lease_seconds"] == 10
                and (armed["device"], armed["inode"]) == (entry.st_dev, entry.st_ino)
                and armed["cgroup"] == device_resources.CGROUP, "Guard armed another process group")
        process.stdin.write(b"freeze\n")
        frozen = frame(process, receipt, timeout=3)
        require(frozen["phase"] == "frozen", "Guard did not establish the requested freeze")
        receipt["census"] = collector(host, allowed_devices=allowed_devices)
        process.stdin.write(b"thaw\n")
        thawed = frame(process, receipt, timeout=3)
        require(thawed["phase"] == "thawed" and thawed["reason"] == "thaw", "Guard lease ended before census completion")
        receipt["freeze_interval_seconds"] = thawed["at_monotonic"] - frozen["at_monotonic"]
    except BaseException as exc:
        failure = exc
        receipt["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        # EOF releases the guard even if collection raised. Its independent
        # monotonic lease also releases the group if this owner is killed.
        process.stdin.close()
        try:
            process.wait(timeout=13)
            receipt["guard_exit_code"] = process.returncode
            receipt["guard_exit_verified"] = True
            for raw in process.stdout.read(16385).splitlines():
                require(len(raw) <= 4096, "Trailing guard response exceeds its bound")
                receipt["guard_frames"].append(json.loads(raw))
            stderr = process.stderr.read(16385)
            require(len(stderr) <= 16384, "Guard diagnostics exceed their bound")
            receipt["guard_stderr"] = stderr.decode()
        except BaseException as exc:
            receipt["guard_cleanup_failure"] = type(exc).__name__ + ": " + str(exc)
            if failure is None:
                failure = exc
        finally:
            process.stdout.close()
            process.stderr.close()
        current = group.stat()
        receipt["thaw_verified"] = (
            (current.st_dev, current.st_ino) == (entry.st_dev, entry.st_ino)
            and (group / "cgroup.freeze").read_text().strip() == "0"
            and process_storage.counters((group / "cgroup.events").read_text()).get("frozen") == 0)
    require(receipt["guard_exit_verified"] and receipt["thaw_verified"], "Census guard cleanup could not be verified")
    if failure is not None:
        raise failure
    require(receipt["guard_exit_code"] == 0, "Census guard failed")
    return receipt


def publish(path, value, *, maximum=32 * 1024**2):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "Census evidence output already exists")
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    require(len(raw) <= maximum, "Census evidence exceeds its output bound")
    temporary = path.with_name(path.name + ".pending")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(raw).hexdigest()


def private_read(path, owner_uid, *, maximum=8192):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        entry = os.fstat(stream.fileno())
        require(stat.S_ISREG(entry.st_mode) and entry.st_uid == owner_uid
                and entry.st_nlink == 1 and stat.S_IMODE(entry.st_mode) == 0o600
                and entry.st_size <= maximum, "Census handoff is not a bounded private owner file")
        raw = stream.read(maximum + 1)
        require(len(raw) == entry.st_size, "Census handoff changed during its read")
    return json.loads(raw)


def validate_request(value):
    require(set(value) == {"schema", "request_id", "release_id", "roles"}
            and value["schema"] == "organism-native-ready-census-request-v1"
            and re.fullmatch("[0-9a-f]{32}", value["request_id"])
            and re.fullmatch("[0-9a-f]{64}", value["release_id"])
            and set(value["roles"]) == {"receiver", "application", "inference"},
            "Unexpected native census request")
    for identity in value["roles"].values():
        require(set(identity) == {"pid", "start_ticks"}
                and type(identity["pid"]) is int and identity["pid"] > 1
                and type(identity["start_ticks"]) is int and identity["start_ticks"] >= 0,
                "Census request lacks an exact process start identity")
    require(len({v["pid"] for v in value["roles"].values()}) == 3, "Census roles must name distinct processes")
    return value


def check_roles(census, roles):
    for identity in roles.values():
        matches = [task for task in census["tasks"]
                   if task["tid"] == task["tgid"] == identity["pid"]
                   and task["start_ticks"] == identity["start_ticks"]]
        require(len(matches) == 1, "An expected native role is absent or has a reused PID")
    require(census["complete_frozen_task_census"] is True and census["native_observation"] is True
            and census["observed_direct_file_backing_verified"] is True
            and census["all_process_writes_contained"] is False,
            "Native role census did not establish the declared direct file backing")


def serve_ready(host, directory, *, release_id, allowed_devices):
    """Respond once to the owner driver's post-readiness census request."""
    directory = Path(directory)
    request_path = directory / REQUEST
    if not request_path.exists():
        return None
    require(not (directory / RESPONSE).exists(), "A native census request was already served")
    request = validate_request(private_read(request_path, host.owner_uid))
    require(request["release_id"] == release_id, "Native census request belongs to another release")
    receipt = {"request": request, "passed": False}
    try:
        capture(host, allowed_devices=allowed_devices, receipt=receipt)
        check_roles(receipt["census"], request["roles"])
        receipt["passed"] = True
    except Exception as exc:
        receipt["failure"] = type(exc).__name__ + ": " + str(exc)
    digest = publish(directory / REPORT, receipt)
    publish(directory / RESPONSE, {
        "schema": "organism-native-ready-census-response-v1", "request_id": request["request_id"],
        "release_id": release_id, "passed": receipt["passed"], "report_sha256": digest,
        "qualification_passed": False,
    }, maximum=8192)
    return receipt


def request_ready(directory, *, release_id, request_id, roles, timeout=30):
    """Wait outside the device slice; the external owner alone controls freezing."""
    require(type(timeout) is int and 1 <= timeout <= 30, "Census handoff wait exceeds its bound")
    directory = Path(directory)
    request = validate_request({"schema": "organism-native-ready-census-request-v1",
                                "request_id": request_id, "release_id": release_id, "roles": roles})
    publish(directory / REQUEST, request, maximum=8192)
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if (directory / RESPONSE).exists():
            response = private_read(directory / RESPONSE, os.geteuid())
            require(set(response) == {"schema", "request_id", "release_id", "passed", "report_sha256",
                                      "qualification_passed"}
                    and response["schema"] == "organism-native-ready-census-response-v1"
                    and response["request_id"] == request_id and response["release_id"] == release_id
                    and response["passed"] is True and response["qualification_passed"] is False,
                    "Owner census failed or belongs to another seed")
            path = directory / REPORT
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                entry = os.fstat(stream.fileno())
                require(stat.S_ISREG(entry.st_mode) and entry.st_uid == os.geteuid()
                        and entry.st_nlink == 1 and stat.S_IMODE(entry.st_mode) == 0o600
                        and entry.st_size <= 32 * 1024**2
                        and hashlib.file_digest(stream, "sha256").hexdigest() == response["report_sha256"],
                        "Owner census report identity differs")
            return response
        time.sleep(0.1)
    raise TimeoutError("External owner census did not complete before its handoff deadline")
