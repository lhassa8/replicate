"""Read-only file-channel census of an already frozen owner device slice.

An external owner must freeze and subsequently thaw the slice. This observer
does neither. It checks every thread, including descendants, and records open
descriptors and writable shared mappings. It does not certify future opens,
socket peers, external helpers, registered asynchronous I/O or all writes.
"""

import os
from pathlib import Path
import re
import stat
import time

from .device_resources import CGROUP, TASKS_MAX, _group
from .device_storage import StorageError


MAX_GROUPS = 64
MAX_FDS = 4096
MAX_MAPS = 4 * 1024**2
MAX_CAPTURE_BYTES = 32 * 1024**2
LINUX_O_PATH = 0o10000000


def require(value, message):
    if not value:
        raise StorageError(message)


def read(path, maximum):
    with Path(path).open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, "Process storage input exceeds its bound")
    return raw.decode("utf-8")


def numbered(path, maximum):
    names = [entry.name for entry in Path(path).iterdir()]
    require(len(names) <= maximum and all(re.fullmatch(r"0|[1-9][0-9]*", n) for n in names),
            "Process storage directory has an invalid or oversized inventory")
    return sorted(int(n) for n in names)


def counters(raw):
    result = {}
    for line in raw.splitlines():
        parts = line.split()
        require(len(parts) == 2 and parts[0] not in result and parts[1].isdigit(),
                "Process storage counters are malformed")
        result[parts[0]] = int(parts[1])
    return result


def freeze_state(group):
    events = counters(read(group / "cgroup.events", 4096))
    require(read(group / "cgroup.freeze", 32).strip() == "1" and events.get("frozen") == 1,
            "Device slice must already be fully frozen by its external owner")
    return events


def thread_inventory(host):
    root = _group(host, CGROUP)
    pending, groups, threads = [root], {}, {}
    while pending:
        path = pending.pop()
        entry = path.lstat()
        require(stat.S_ISDIR(entry.st_mode) and entry.st_uid == host.owner_uid,
                "Process census encountered a linked or foreign cgroup")
        relative = "/" + str(path.relative_to(host.cgroup_root))
        kind = read(path / "cgroup.type", 128).strip()
        require(kind == "domain", "The native profile requires domain cgroups")
        raw = read(path / "cgroup.threads", 65536)
        values = raw.splitlines()
        require(all(re.fullmatch(r"[1-9][0-9]*", v) and int(v) > 1 for v in values)
                and len(values) == len(set(values)), "Cgroup thread inventory is ambiguous")
        for value in values:
            tid = int(value)
            require(tid not in threads, "A thread appears in more than one cgroup")
            threads[tid] = relative
        groups[relative] = {"device": entry.st_dev, "inode": entry.st_ino,
                            "type": kind, "threads": sorted(map(int, values))}
        require(len(groups) <= MAX_GROUPS and len(threads) <= TASKS_MAX,
                "Device process census exceeds its group or task bound")
        for child in path.iterdir():
            mode = child.lstat().st_mode
            require(not stat.S_ISLNK(mode), "Cgroup census cannot follow links")
            if stat.S_ISDIR(mode):
                pending.append(child)
    require(threads and os.getpid() not in threads, "The observer must be outside a populated device slice")
    return groups, threads


def start_identity(raw, tid):
    match = re.match(r"([0-9]+) \(", raw)
    closing = raw.rfind(")")
    fields = raw[closing + 1:].split()
    require(match and int(match[1]) == tid and closing > match.end()
            and len(fields) >= 20 and fields[19].isdigit(), "Process start identity is unavailable")
    return {"tid": tid, "start_ticks": int(fields[19])}


def fd_metadata(raw):
    result = {}
    for line in raw.splitlines():
        key, separator, value = line.partition(":")
        if key in ("pos", "flags", "mnt_id", "ino"):
            require(separator and key not in result, "Duplicate descriptor metadata")
            value = value.strip()
            require(re.fullmatch("[0-7]+" if key == "flags" else "[0-9]+", value),
                    "Descriptor metadata is malformed")
            result[key] = int(value, 8 if key == "flags" else 10)
    require(set(result) == {"pos", "flags", "mnt_id", "ino"}, "Descriptor metadata is incomplete")
    return result


def file_identity(entry):
    return {"device": str(os.major(entry.st_dev)) + ":" + str(os.minor(entry.st_dev)),
            "inode": entry.st_ino, "mode": entry.st_mode, "rdev": entry.st_rdev,
            "links": entry.st_nlink}


def captured(path, maximum, budget):
    raw = read(path, maximum)
    budget["bytes"] += len(raw.encode())
    require(budget["bytes"] <= MAX_CAPTURE_BYTES, "Aggregate process capture exceeds its bound")
    return raw


def descriptor(process, number, budget):
    path = process / "fd" / str(number)
    before = path.stat()
    target = os.readlink(path)
    require(len(target.encode()) <= 8192, "Descriptor target exceeds its bound")
    raw = captured(process / "fdinfo" / str(number), 65536, budget)
    info = fd_metadata(raw)
    require(info["ino"] == before.st_ino and file_identity(path.stat()) == file_identity(before)
            and os.readlink(path) == target, "Descriptor changed during its observation")
    return {"fd": number, "target": target, **file_identity(before), **info,
            "fdinfo_raw": raw}


def parse_maps(raw):
    require(len(raw.encode()) <= MAX_MAPS, "Mapped-file observation exceeds its bound")
    rows, previous_end = [], 0
    for line in raw.splitlines():
        parts = line.split(None, 5)
        require(len(parts) >= 5, "Mapped-file record is incomplete")
        address, permissions, offset, device, inode = parts[:5]
        require(re.fullmatch(r"[0-9a-f]+-[0-9a-f]+", address)
                and re.fullmatch(r"[r-][w-][x-][ps]", permissions)
                and re.fullmatch(r"[0-9a-f]+", offset)
                and re.fullmatch(r"[0-9a-f]+:[0-9a-f]+", device) and inode.isdigit(),
                "Mapped-file record is malformed")
        start, end = (int(value, 16) for value in address.split("-"))
        require(previous_end <= start < end, "Mapped-file ranges overlap or are unordered")
        previous_end = end
        major, minor = (int(value, 16) for value in device.split(":"))
        rows.append({"address": address, "permissions": permissions, "offset": int(offset, 16),
                     "device": str(major) + ":" + str(minor), "inode": int(inode),
                     "path": parts[5] if len(parts) == 6 else ""})
    require(rows, "Process mapping inventory is empty")
    return rows


def task(host, tid, group, budget):
    process = Path(host.proc_root) / str(tid)
    before = start_identity(read(process / "stat", 65536), tid)
    membership = read(process / "cgroup", 4096)
    require(membership.splitlines() == ["0::" + group], "Task escaped its recorded cgroup")
    status = read(process / "status", 65536)
    fields = {}
    for line in status.splitlines():
        key, _, value = line.partition(":")
        if key in ("Pid", "Tgid", "Threads"):
            require(key not in fields and value.strip().isdigit(), "Task status is ambiguous")
            fields[key] = int(value)
    require(set(fields) == {"Pid", "Tgid", "Threads"} and fields["Pid"] == tid
            and fields["Tgid"] > 1 and 0 < fields["Threads"] <= TASKS_MAX, "Task status is incomplete")
    numbers = numbered(process / "fd", MAX_FDS)
    descriptors = [descriptor(process, number, budget) for number in numbers]
    raw_maps = captured(process / "maps", MAX_MAPS, budget)
    mappings = parse_maps(raw_maps)
    namespaces = {name: file_identity((process / "ns" / name).stat()) for name in ("mnt", "net", "pid", "cgroup")}
    require(numbered(process / "fd", MAX_FDS) == numbers
            and start_identity(read(process / "stat", 65536), tid) == before,
            "Task or descriptor inventory changed during capture")
    children_raw = read(Path(host.proc_root) / str(fields["Tgid"]) / "task" / str(tid) / "children", 65536)
    children = children_raw.split()
    require(all(re.fullmatch(r"[1-9][0-9]*", n) for n in children)
            and len(children) == len(set(children)), "Task child inventory is ambiguous")
    return {**before, "tgid": fields["Tgid"], "thread_count": fields["Threads"], "cgroup": group,
            "namespaces": namespaces, "children": sorted(map(int, children)), "descriptors": descriptors,
            "maps_raw": raw_maps, "mappings": mappings}


def classify(tasks, allowed_devices, *, null_rdev=None):
    # Independent cross-platform readback can supply Linux's encoded /dev/null
    # device number. Live collection keeps the local kernel's representation.
    null_rdev = os.makedev(1, 3) if null_rdev is None else null_rdev
    require(isinstance(allowed_devices, (list, tuple)) and 1 <= len(allowed_devices) <= 3
            and len(set(allowed_devices)) == len(allowed_devices)
            and all(isinstance(v, str) and re.fullmatch(r"[0-9]+:[0-9]+", v) for v in allowed_devices),
            "Explicit distinct backing-device identities are required")
    violations, unresolved, files, mappings = [], [], [], []
    for item in tasks:
        for fd in item["descriptors"]:
            detail = {"tid": item["tid"], **fd}
            path_only = bool(fd["flags"] & LINUX_O_PATH)
            writable = not path_only and fd["flags"] & 3 in (1, 2)
            if stat.S_ISREG(fd["mode"]) and writable:
                files.append(detail)
                if fd["device"] not in allowed_devices:
                    violations.append({"kind": "file", "tid": item["tid"], "fd": fd["fd"],
                                       "device": fd["device"], "inode": fd["inode"]})
            elif stat.S_ISSOCK(fd["mode"]) or stat.S_ISFIFO(fd["mode"]) or fd["target"].startswith("anon_inode:"):
                unresolved.append({"kind": "ipc_or_kernel_handle", "tid": item["tid"], "fd": fd["fd"],
                                   "target": fd["target"]})
            elif writable and not (stat.S_ISCHR(fd["mode"]) and fd["rdev"] == null_rdev):
                unresolved.append({"kind": "special_write_handle", "tid": item["tid"], "fd": fd["fd"],
                                   "target": fd["target"]})
            elif path_only or stat.S_ISDIR(fd["mode"]):
                unresolved.append({"kind": "path_handle", "tid": item["tid"], "fd": fd["fd"],
                                   "target": fd["target"]})
        for mapping in item["mappings"]:
            if mapping["permissions"][1] != "w" or mapping["permissions"][3] != "s":
                continue
            mappings.append({"tid": item["tid"], **mapping})
            if mapping["inode"] == 0:
                unresolved.append({"kind": "anonymous_shared_mapping", "tid": item["tid"],
                                   "address": mapping["address"]})
            elif mapping["device"] not in allowed_devices:
                violations.append({"kind": "shared_mapping", "tid": item["tid"],
                                   "address": mapping["address"], "device": mapping["device"],
                                   "inode": mapping["inode"]})
    return {"writable_regular_files": files, "writable_shared_mappings": mappings,
            "backing_device_violations": violations, "unresolved_channels": unresolved,
            "observed_direct_file_backing_verified": not violations,
            "all_process_writes_contained": False}


def stable_task(value):
    # Timer/event counters can change in the kernel while userspace is frozen.
    # Preserve their raw observations but compare descriptor identity/access.
    return {**value, "descriptors": [
        {k: v for k, v in fd.items() if k != "fdinfo_raw"} for fd in value["descriptors"]]}


def observe_frozen(host, *, allowed_devices):
    started = time.monotonic()
    classify([], allowed_devices)
    group = _group(host, CGROUP)
    freeze_state(group)
    before, members = thread_inventory(host)
    budget = {"bytes": 0}
    tasks = [task(host, tid, relative, budget) for tid, relative in sorted(members.items())]
    by_process = {}
    for item in tasks:
        by_process.setdefault(item["tgid"], set()).add(item["tid"])
        require(set(item["children"]) <= set(members), "A live or unreaped child is outside the frozen census")
    for pid, tids in by_process.items():
        require(set(numbered(Path(host.proc_root) / str(pid) / "task", TASKS_MAX)) == tids
                and all(item["thread_count"] == len(tids) for item in tasks if item["tgid"] == pid),
                "A process has an unobserved thread outside the frozen census")
    repeated_budget = {"bytes": 0}
    repeated = [task(host, tid, relative, repeated_budget) for tid, relative in sorted(members.items())]
    require([stable_task(item) for item in tasks] == [stable_task(item) for item in repeated],
            "Frozen process file-channel identities changed during the census")
    require(thread_inventory(host) == (before, members), "Device cgroup population changed during the census")
    freeze_state(group)
    result = classify(tasks, allowed_devices)
    return {"schema": "organism-frozen-process-storage-v1", "cgroup": CGROUP, "groups": before,
            "tasks": tasks, "process_count": len(by_process), "thread_count": len(tasks),
            "allowed_devices": list(allowed_devices), "captured_proc_bytes": budget["bytes"],
            "elapsed_seconds": time.monotonic() - started, "complete_frozen_task_census": True,
            "native_observation": not host.is_fixture, "file_contents_read": False,
            "freeze_or_thaw_performed": False, "qualification_passed": False, **result}
