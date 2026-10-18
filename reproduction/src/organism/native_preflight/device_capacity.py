"""Read-only whole-host capacity admission for the dedicated 8 GiB seed test.

Three owner-provisioned disks bound persistent writes, including host helpers
and logs outside the organism volumes. File-backed loops count against their
backing disk, not as additional capacity. This is sampled host admission; it
does not prove every process descriptor or authorize disk/host configuration.
"""

import os
from pathlib import Path
import platform
import re
import stat
import time

from ..device_storage import parse_mountinfo


GIB = 1024**3
ROLE_LIMITS = {"root": 8 * GIB, "data": 22 * GIB, "control": 2 * GIB}
ROLE_MOUNTS = {"root": "/", "data": "/var/lib/organism/data", "control": "/var/lib/organism/control"}
MEMORY_MIN = 7 * GIB
MEMORY_MAX = 8 * GIB
MEMORY_FILESYSTEMS = frozenset((
    "proc", "sysfs", "devtmpfs", "tmpfs", "devpts", "cgroup2", "pstore", "bpf",
    "securityfs", "debugfs", "tracefs", "configfs", "fusectl", "efivarfs",
    "mqueue", "hugetlbfs", "binfmt_misc", "nsfs", "autofs",
))
BLOCK_FILESYSTEMS = frozenset(("ext4", "vfat", "squashfs"))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path, maximum=4096):
    with Path(path).open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, "Host capacity input exceeds its bound")
    return raw.decode()


def inventory(sys_root=Path("/sys")):
    root = Path(sys_root) / "class/block"
    entries = sorted(root.iterdir())
    require(0 < len(entries) <= 1024, "Block-device inventory is unavailable or unbounded")
    result = {}
    for path in entries:
        require(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", path.name), "Unexpected kernel block-device name")
        device = read(path / "dev").strip()
        sectors = read(path / "size").strip()
        require(re.fullmatch(r"[0-9]+:[0-9]+", device)
                and re.fullmatch(r"[0-9]+", sectors), "Invalid kernel block-device counters")
        item = {"device": device, "bytes": int(sectors) * 512}
        if (path / "partition").exists():
            item.update(kind="partition", parent=path.resolve().parent.name)
        elif (path / "device").exists():
            item["kind"] = "disk"
        elif re.fullmatch(r"loop[0-9]+", path.name):
            item["kind"] = "loop"
            if item["bytes"]:
                backing = read(path / "loop/backing_file").strip()
                require(backing.startswith("/") and ".." not in Path(backing).parts,
                        "Loop backing file is not an explicit local path")
                info = Path(backing).stat()
                require(stat.S_ISREG(info.st_mode), "Loop backing storage is not a regular file")
                item.update(backing_path=backing,
                            backing_device=f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}")
        else:
            item["kind"] = "unsupported"
        result[path.name] = item
    return result


def validate(blocks, mounts, *, memory_bytes, swaps):
    """Reconstruct capacity from explicit kernel observations; perform no I/O."""
    require(type(memory_bytes) is int and MEMORY_MIN <= memory_bytes <= MEMORY_MAX,
            "Combined seed requires the declared 8 GiB Linux memory profile")
    swap_lines = [line for line in swaps.splitlines() if line.strip()]
    require(len(swap_lines) == 1 and swap_lines[0].split() == [
        "Filename", "Type", "Size", "Used", "Priority"], "The dedicated seed host must have no swap")
    require(isinstance(blocks, dict) and 0 < len(blocks) <= 1024, "Incomplete block-device inventory")
    devices = {}
    for name, item in blocks.items():
        require(type(item.get("bytes")) is int and item["bytes"] >= 0
                and item.get("kind") in ("disk", "partition", "loop", "unsupported"),
                "Invalid block-device observation")
        device = item.get("device")
        require(isinstance(device, str) and re.fullmatch(r"[0-9]+:[0-9]+", device)
                and device not in devices, "Duplicate or invalid kernel block-device identity")
        devices[device] = name

    def physical(device, *, visiting=()):
        require(device in devices and device not in visiting, "Unknown or cyclic mounted block storage")
        name = devices[device]
        item = blocks[name]
        if item["kind"] == "disk":
            require(item["bytes"] > 0, "Mounted disk has no observed capacity")
            return name
        if item["kind"] == "partition":
            parent = blocks.get(item.get("parent"), {})
            require(parent.get("kind") == "disk" and item["bytes"] <= parent["bytes"],
                    "Partition has no bounded physical parent")
            return physical(parent["device"], visiting=(*visiting, device))
        if item["kind"] == "loop":
            require(item["bytes"] > 0 and "backing_device" in item,
                    "Mounted loop has no observed local backing storage")
            return physical(item["backing_device"], visiting=(*visiting, device))
        raise ValueError("Unsupported mounted block storage")

    roles = {}
    for role, target in ROLE_MOUNTS.items():
        candidates = [mount for mount in mounts if mount["target"] == target]
        require(len(candidates) == 1, "Whole-host profile requires one unambiguous " + role + " mount")
        mount = candidates[0]
        require(mount["filesystem"] == "ext4" and mount["root"] == "/"
                and "rw" in mount["options"], "Whole-host role is not a writable complete ext4 filesystem")
        disk = physical(mount["device"])
        # Data/control must be physical disks, not files inside the root disk.
        if role != "root":
            require(blocks[devices[mount["device"]]]["kind"] == "disk",
                    "Data and control require their own physical disks")
        size = blocks[disk]["bytes"]
        require(size <= ROLE_LIMITS[role], "Whole-host role exceeds its allocated disk capacity")
        roles[role] = {"mount": target, "mount_device": mount["device"],
                       "disk": disk, "disk_device": blocks[disk]["device"], "capacity_bytes": size}
    required_disks = {value["disk"] for value in roles.values()}
    require(len(required_disks) == 3, "Root, data and control must use distinct physical disks")
    require({name for name, value in blocks.items() if value["kind"] == "disk"} == required_disks,
            "Additional physical storage is outside the whole-host budget")
    for name, item in blocks.items():
        if item["bytes"] > 0:
            require(item["kind"] != "unsupported", "An active unsupported block device is present")
            require(physical(item["device"]) in required_disks, "Block storage escapes the three bounded disks")
    for mount in mounts:
        filesystem = mount["filesystem"]
        if filesystem in BLOCK_FILESYSTEMS:
            require(physical(mount["device"]) in required_disks, "Mounted filesystem escapes bounded storage")
        else:
            require(filesystem in MEMORY_FILESYSTEMS, "Remote or unsupported filesystem on the seed host")
    total = sum(role["capacity_bytes"] for role in roles.values())
    require(total <= 32 * GIB, "Whole-host persistent capacity exceeds 32 GiB")
    return {
        "profile": "owner-root8-data22-control2-v1", "roles": roles,
        "total_physical_storage_bytes": total, "memory_bytes": memory_bytes,
        "swap_devices": 0, "all_persistent_block_storage_bounded": True,
        "process_write_containment_verified": False, "qualification_passed": False,
    }


def observe(*, expected_roles=None):
    require(platform.system() == "Linux" and os.geteuid() == 0,
            "Whole-host capacity observation requires the dedicated owner Linux device")
    started = time.monotonic()
    blocks = inventory()
    raw_mounts = read("/proc/self/mountinfo", 4 * 1024**2)
    memory = read("/proc/meminfo", 65536)
    match = re.findall(r"^MemTotal:\s+([0-9]+) kB$", memory, flags=re.MULTILINE)
    require(len(match) == 1, "Kernel memory capacity is unavailable")
    swaps = read("/proc/swaps", 65536)
    result = validate(blocks, parse_mountinfo(raw_mounts), memory_bytes=int(match[0]) * 1024, swaps=swaps)
    require(blocks == inventory() and raw_mounts == read("/proc/self/mountinfo", 4 * 1024**2)
            and swaps == read("/proc/swaps", 65536), "Host storage changed during capacity observation")
    if expected_roles is not None:
        require(result["roles"] == expected_roles, "Whole-host disk identities changed after seed admission")
    return {
        **result, "block_devices": blocks, "mountinfo": raw_mounts,
        "meminfo": memory, "swaps": swaps,
        "native_observation": True, "started_monotonic_seconds": started,
        "completed_monotonic_seconds": time.monotonic(),
    }
