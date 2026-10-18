"""Owner-installed aggregate CPU/memory boundary for one native lab device.

This module renders and observes the fixed profile. It never creates a slice,
changes limits, moves processes, or grants a model control of host resources.
Scripted hosts are explicitly fixtures, not evidence of native enforcement.
"""

import hashlib
import os
from pathlib import Path
import re
import stat


SLICE = "system-organism.slice"
CGROUP = "/system.slice/" + SLICE
RECEIVER_UNIT = "organism-receiver.service"
MEMORY_MAX = 6 * 1024**3
TASKS_MAX = 256
SLICE_FIELDS = (
    "Id", "LoadState", "ActiveState", "SubState", "ControlGroup", "FragmentPath",
    "DropInPaths", "NeedDaemonReload", "MemoryAccounting", "CPUAccounting",
    "TasksAccounting", "MemoryMax", "MemorySwapMax", "TasksMax", "CPUQuotaPerSecUSec",
)
RECEIVER_FIELDS = (
    "Id", "LoadState", "ActiveState", "SubState", "MainPID", "ControlGroup", "Slice",
    "FragmentPath", "NeedDaemonReload", "Delegate", "KillMode", "User",
)


class DeviceResourceError(ValueError):
    pass


def validate_slice(value):
    if value is not None and value != SLICE:
        raise DeviceResourceError("Only the fixed owner-installed device slice is supported")
    return value


def service_cgroup(name, resource_slice=None):
    validate_slice(resource_slice)
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,240}\.service", name):
        raise DeviceResourceError("Invalid resource-bound service name")
    return (CGROUP if resource_slice else "/system.slice") + "/" + name


def render_slice():
    return "\n".join((
        "[Unit]", "Description=Owner-controlled organism device resource budget",
        "[Slice]", "MemoryAccounting=yes", "CPUAccounting=yes", "TasksAccounting=yes",
        "MemoryHigh=" + str(5 * 1024**3), "MemoryMax=" + str(MEMORY_MAX),
        "MemorySwapMax=0", "CPUQuota=400%", "TasksMax=" + str(TASKS_MAX), "",
    ))


def _read(path, *, maximum=65536, owner_uid=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_size > maximum
                or owner_uid is not None and (
                    info.st_uid != owner_uid or info.st_nlink != 1 or info.st_mode & 0o022)):
            raise DeviceResourceError("Resource profile must be a bounded owner-controlled regular file")
        raw = source.read(maximum + 1)
    if len(raw) > maximum:
        raise DeviceResourceError("Resource observation exceeds its bound")
    return raw.decode("utf-8")


def _show(host, name, fields):
    reply = host.run(["systemctl", "show", name, "--no-pager", "--property=" + ",".join(fields)])
    if reply.returncode or len(reply.stdout.encode()) > 65536:
        raise DeviceResourceError("Resource manager observation failed")
    value = {}
    for line in reply.stdout.splitlines():
        key, separator, item = line.partition("=")
        if not separator or key in value:
            raise DeviceResourceError("Malformed resource manager observation")
        value[key] = item
    if set(value) != set(fields):
        raise DeviceResourceError("Resource manager observation is incomplete")
    return value


def _group(host, relative):
    """Reject links in every component below the supplied cgroup mount."""
    root = Path(host.cgroup_root)
    path = root
    for component in Path(relative).parts[1:]:
        path = path / component
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != host.owner_uid:
            raise DeviceResourceError("Device cgroup is not an owner-controlled directory")
    return path


def _counter(text):
    if not re.fullmatch(r"[0-9]+", text):
        raise DeviceResourceError("Device resource counter is unavailable")
    return int(text)


def _counters(text):
    value = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2 or parts[0] in value:
            raise DeviceResourceError("Malformed device resource counters")
        value[parts[0]] = _counter(parts[1])
    if not value:
        raise DeviceResourceError("Device resource counters are missing")
    return value


def inspect_slice(host):
    """Read the parent limit and counters, including all its descendants."""
    path = host.unit_root / SLICE
    text = _read(path, maximum=16384, owner_uid=host.owner_uid)
    if text != render_slice():
        raise DeviceResourceError("Owner device slice differs from the fixed profile")
    value = _show(host, SLICE, SLICE_FIELDS)
    expected = {
        "Id": SLICE, "LoadState": "loaded", "ActiveState": "active", "SubState": "active",
        "ControlGroup": CGROUP, "FragmentPath": str(path), "DropInPaths": "",
        "NeedDaemonReload": "no", "MemoryAccounting": "yes", "CPUAccounting": "yes",
        "TasksAccounting": "yes", "MemoryMax": str(MEMORY_MAX), "MemorySwapMax": "0",
        "TasksMax": str(TASKS_MAX),
    }
    if any(value[key] != item for key, item in expected.items()):
        raise DeviceResourceError("Loaded device slice differs from the fixed profile")
    if value["CPUQuotaPerSecUSec"] not in ("4s", "4000000"):
        raise DeviceResourceError("Device slice CPU allowance differs")
    group = _group(host, CGROUP)
    limits = {key: _read(group / key, maximum=4096).strip()
              for key in ("memory.max", "memory.swap.max", "pids.max", "cpu.max")}
    cpu = limits["cpu.max"].split()
    if (limits["memory.max"] != str(MEMORY_MAX) or limits["memory.swap.max"] != "0"
            or limits["pids.max"] != str(TASKS_MAX) or len(cpu) != 2
            or not all(re.fullmatch("[0-9]+", item) for item in cpu)
            or int(cpu[1]) <= 0 or int(cpu[0]) != 4 * int(cpu[1])):
        raise DeviceResourceError("Kernel aggregate limits differ from the fixed device profile")
    counters = {key: _counter(_read(group / key, maximum=4096).strip())
                for key in ("memory.current", "memory.peak", "memory.swap.current", "pids.current")}
    events = _counters(_read(group / "memory.events", maximum=4096))
    cpu_stat = _counters(_read(group / "cpu.stat", maximum=4096))
    if not {"oom", "oom_kill", "max", "high", "low"} <= set(events) or "usage_usec" not in cpu_stat:
        raise DeviceResourceError("Required aggregate accounting fields are missing")
    return {
        "slice": SLICE, "cgroup": CGROUP, "properties": value, "kernel_limits": limits,
        "counters": counters, "memory_events": events, "cpu_stat": cpu_stat,
        "unit_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "native_observation": not host.is_fixture, "qualification_passed": False,
    }


def verify_receiver(host, *, pid=None):
    """Require the actual listener/preparer process inside the same boundary."""
    pid = os.getpid() if pid is None else pid
    if type(pid) is not int or pid <= 1:
        raise DeviceResourceError("Receiver process identity is unavailable")
    result = inspect_slice(host)
    value = _show(host, RECEIVER_UNIT, RECEIVER_FIELDS)
    path = host.unit_root / RECEIVER_UNIT
    # Its command and privileges belong to owner provisioning. Resource
    # admission verifies that this exact process is under the fixed parent.
    _read(path, maximum=32768, owner_uid=host.owner_uid)
    expected = {
        "Id": RECEIVER_UNIT, "LoadState": "loaded", "ActiveState": "active", "SubState": "running",
        "MainPID": str(pid), "ControlGroup": service_cgroup(RECEIVER_UNIT, SLICE), "Slice": SLICE,
        "FragmentPath": str(path), "NeedDaemonReload": "no", "Delegate": "no", "KillMode": "control-group",
    }
    if any(value[key] != item for key, item in expected.items()) or value["User"] not in ("", "root"):
        raise DeviceResourceError("Receiving service is outside the owner aggregate boundary")
    group = _group(host, service_cgroup(RECEIVER_UNIT, SLICE))
    membership = _read(host.proc_root / str(pid) / "cgroup", maximum=4096).splitlines()
    if membership != ["0::" + service_cgroup(RECEIVER_UNIT, SLICE)]:
        raise DeviceResourceError("Actual receiver PID is outside the device resource boundary")
    processes = _read(group / "cgroup.procs", maximum=65536).splitlines()
    if str(pid) not in processes:
        raise DeviceResourceError("Receiver PID is absent from its kernel process group")
    return {**result, "receiver": {"pid": pid, "properties": value}}
