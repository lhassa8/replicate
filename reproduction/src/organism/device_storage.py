"""Read-only accounting of the owner's two bounded native filesystems.

The data volume holds payloads, work and temporary files. A separate control
volume keeps owner records writable when data is full. This observer does not
install mounts, move files, or establish complete process write containment.
"""

import base64
import hashlib
import os
from pathlib import Path
import re
import stat
import struct
import time


PROFILE = "split-ext4-v1"
ROOT = Path("/var/lib/organism")
VOLUME_LIMITS = {"data": 30 * 1024**3, "control": 2 * 1024**3}
TOTAL_LIMIT = 32 * 1024**3
CONTROL_HEADROOM = 128 * 1024**2
CONTROL_INODE_HEADROOM = 4096
MAX_MOUNTINFO = 4 * 1024**2
CONTROL_PATHS = ("enrollment_file", "state_directory", "identity_directory", "discovery_file")
DATA_PATHS = ("capsule_store", "prepared_store", "work_root", "context_root", "incoming_directory")


class StorageError(ValueError):
    pass


def validate_profile(value):
    if value is not None and value != PROFILE:
        raise StorageError("Only the split owner storage profile is supported")
    return value


def validate_layout(config, *, root=ROOT):
    """Validate configured roles; actual mounts and files still require observation."""
    if not isinstance(config, dict):
        raise StorageError("Explicit owner storage role mapping is required")
    root = Path(root)
    result = {}
    for role, fields in (("control", CONTROL_PATHS), ("data", DATA_PATHS)):
        parent = root / role
        for field in fields:
            value = config.get(field)
            if (not isinstance(value, str) or not value.startswith("/")
                    or Path(value).as_posix() != value or ".." in Path(value).parts
                    or any(ord(char) < 32 or ord(char) == 127 for char in value)
                    or Path(value) == parent or not Path(value).is_relative_to(parent)):
                raise StorageError("Configured storage path is outside its role: " + field)
            result[field] = value
    # A transfer/cache directory may not contain owner-controlled work roots,
    # and no configured role may alias another one.
    paths = [Path(value) for value in result.values()]
    logs = root / "data/logs"
    if any(path.is_relative_to(logs) or logs.is_relative_to(path) for path in paths):
        raise StorageError("Configured storage overlaps the reserved application log directory")
    if any(left.is_relative_to(right) or right.is_relative_to(left)
           for index, left in enumerate(paths) for right in paths[index + 1:]):
        raise StorageError("Configured storage roles overlap")
    return result


def _unescape(value):
    escapes = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}
    if re.search(r"\\(?!040|011|012|134)", value):
        raise StorageError("Malformed mountinfo path escape")
    return re.sub(r"\\(040|011|012|134)", lambda match: escapes[match[1]], value)


def parse_mountinfo(text):
    if not isinstance(text, str) or not 0 < len(text.encode()) <= MAX_MOUNTINFO:
        raise StorageError("Mount observation is missing or oversized")
    result, ids = [], set()
    for line in text.splitlines():
        before, separator, after = line.partition(" - ")
        fields, filesystem = before.split(), after.split()
        if (not separator or len(fields) < 6 or len(filesystem) != 3
                or not fields[0].isdigit() or int(fields[0]) <= 0
                or not fields[1].isdigit() or not re.fullmatch(r"[0-9]+:[0-9]+", fields[2])
                or int(fields[0]) in ids):
            raise StorageError("Malformed mountinfo record")
        ids.add(int(fields[0]))
        root, target = map(_unescape, fields[3:5])
        # nsfs represents a namespace handle as e.g. net:[4026532871],
        # rather than a filesystem pathname (Linux fs/nsfs.c: ns_dname).
        # Its mount target must still be absolute. It cannot serve as a
        # data/control filesystem because select_mounts requires ext4 at /.
        namespace_root = (filesystem[:2] == ["nsfs", "nsfs"]
                          and re.fullmatch(r"(net|mnt|uts|ipc|pid|user|cgroup|time):\[([1-9][0-9]{0,19})\]",
                                           root))
        if namespace_root and int(namespace_root[2]) > 2**64 - 1:
            namespace_root = None
        if not target.startswith("/") or not (root.startswith("/") or namespace_root):
            raise StorageError("Mount target or filesystem root is invalid")
        result.append({
            "mount_id": int(fields[0]), "parent_id": int(fields[1]), "device": fields[2],
            "root": root, "target": target, "options": fields[5].split(","),
            "optional_fields": fields[6:], "filesystem": filesystem[0],
            "source": _unescape(filesystem[1]), "super_options": filesystem[2].split(","),
        })
    return result


def select_mounts(mounts, *, root=ROOT):
    """Require complete, distinct filesystems without aliases or nested mounts."""
    root = Path(root)
    selected = {}
    for role in VOLUME_LIMITS:
        target = str(root / role)
        matches = [item for item in mounts if item["target"] == target]
        if len(matches) != 1:
            raise StorageError("Required owner volume is missing or overmounted")
        mount = matches[0]
        if (mount["root"] != "/" or mount["filesystem"] != "ext4"
                or not {"rw", "nosuid", "nodev"} <= set(mount["options"])
                or "ro" in mount["options"] or "rw" not in mount["super_options"]
                or "ro" in mount["super_options"]
                or any(option.startswith(("journal_dev=", "journal_path="))
                       for option in mount["options"] + mount["super_options"])
                or role == "control" and "noexec" not in mount["options"]
                or role == "data" and "noexec" in mount["options"]):
            raise StorageError("Owner volume type, root or mount restrictions differ")
        selected[role] = mount
    devices = {item["device"] for item in selected.values()}
    if len(devices) != len(VOLUME_LIMITS):
        raise StorageError("Data and owner control must use distinct filesystems")
    allowed_ids = {item["mount_id"] for item in selected.values()}
    for mount in mounts:
        if mount["mount_id"] not in allowed_ids and (
                mount["device"] in devices or Path(mount["target"]).is_relative_to(root)):
            raise StorageError("An alias or nested mount breaks the declared storage boundary")
    return selected


def _read(path, maximum):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        raw = source.read(maximum + 1)
    if len(raw) > maximum:
        raise StorageError("Kernel storage observation exceeds its bound")
    return raw.decode("utf-8")


def observe_core_policy(host):
    """Require disabled kernel core output on the dedicated owner device.

    A piped handler bypasses RLIMIT_CORE and executes outside the crashing
    process's namespaces. This is a read-only admission check, not a sysctl
    installer or proof of complete process write containment.
    """
    kernel = Path(host.proc_root) / "sys/kernel"
    try:
        pattern = _read(kernel / "core_pattern", 4096)
        uses_pid = _read(kernel / "core_uses_pid", 32)
    except (OSError, UnicodeError) as exc:
        raise StorageError("Kernel core-dump policy could not be observed") from exc
    # Do not strip whitespace: a space can itself be a core filename.
    if pattern not in ("", "\n") or uses_pid not in ("0", "0\n"):
        raise StorageError("Dedicated device must disable core output with an empty pattern and zero PID suffix")
    return {"schema": "organism-core-dump-policy-v1",
            "core_pattern_raw": pattern, "core_uses_pid_raw": uses_pid,
            "kernel_core_output_disabled_at_observation": True,
            "native_observation": not host.is_fixture, "qualification_passed": False}


def _owned_directory(path, owner_uid, *, private=False):
    entry = Path(path).lstat()
    if (not stat.S_ISDIR(entry.st_mode) or entry.st_uid != owner_uid
            or entry.st_mode & (0o077 if private else 0o022)):
        raise StorageError("Storage root must remain an owner-controlled directory")
    return entry


def capacity_counters(capacity, values):
    """Count filesystem metadata as used space, not as free deployment capacity."""
    required = ("f_frsize", "f_blocks", "f_bfree", "f_bavail", "f_files", "f_ffree", "f_favail", "f_flag")
    if (type(capacity) is not int or capacity <= 0
            or set(values) != set(required) or any(type(values[key]) is not int for key in required)
            or values["f_frsize"] <= 0 or values["f_blocks"] <= 0
            or values["f_blocks"] * values["f_frsize"] > capacity
            or not 0 <= values["f_bavail"] <= values["f_bfree"] <= values["f_blocks"]
            or not 0 <= values["f_favail"] <= values["f_ffree"] <= values["f_files"]
            or values["f_flag"] & os.ST_RDONLY):
        raise StorageError("Filesystem capacity or free-space counters are invalid")
    free = values["f_bfree"] * values["f_frsize"]
    available = values["f_bavail"] * values["f_frsize"]
    return {
        "capacity_bytes": capacity, "filesystem_data_bytes": values["f_blocks"] * values["f_frsize"],
        "free_bytes": free, "unprivileged_available_bytes": available,
        "used_including_metadata_bytes": capacity - free, "inodes": values["f_files"],
        "free_inodes": values["f_ffree"], "unprivileged_available_inodes": values["f_favail"],
        "raw_statvfs": values,
    }


def internal_ext4_superblock(raw, capacity):
    """Check Linux's on-disk ext4 identity and require a journal on this device."""
    if not isinstance(raw, bytes) or len(raw) != 1024 or type(capacity) is not int or capacity <= 0:
        raise StorageError("Ext4 superblock capture is incomplete")
    integer = lambda offset: struct.unpack_from("<I", raw, offset)[0]
    magic = struct.unpack_from("<H", raw, 0x38)[0]
    log_size, compat, incompat = integer(0x18), integer(0x5C), integer(0x60)
    if (magic != 0xEF53 or log_size > 6 or not compat & 4 or incompat & 8
            or integer(0xE0) == 0 or integer(0xE4) != 0 or raw[0xD0:0xE0] != b"\0" * 16
            or raw[0x68:0x78] == b"\0" * 16):
        raise StorageError("Ext4 must have its own internal journal and filesystem identity")
    blocks = integer(0x04) + ((integer(0x150) << 32) if incompat & 0x80 else 0)
    size = blocks * (1024 << log_size)
    if not 0 < size <= capacity:
        raise StorageError("Ext4 filesystem exceeds its containing device")
    return {"filesystem_uuid_hex": raw[0x68:0x78].hex(), "filesystem_bytes": size,
            "block_size": 1024 << log_size, "internal_journal_inode": integer(0xE0),
            "external_journal": False, "superblock_sha256": hashlib.sha256(raw).hexdigest(),
            "superblock_base64": base64.b64encode(raw).decode()}


def _superblock(host, mount, capacity):
    event = _read(Path(host.sys_root) / "dev/block" / mount["device"] / "uevent", 4096)
    names = [line.removeprefix("DEVNAME=") for line in event.splitlines() if line.startswith("DEVNAME=")]
    if len(names) != 1 or not re.fullmatch(r"[a-zA-Z0-9_-]+", names[0]):
        raise StorageError("Kernel block device name is unavailable")
    device = Path(host.dev_root) / names[0]
    fd = os.open(device, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        entry = os.fstat(fd)
        number = str(os.major(entry.st_rdev)) + ":" + str(os.minor(entry.st_rdev))
        if not stat.S_ISBLK(entry.st_mode) or number != mount["device"]:
            raise StorageError("Observed block node differs from the mounted device")
        raw = os.pread(fd, 1024, 1024)
    finally:
        os.close(fd)
    return internal_ext4_superblock(raw, capacity)


def _volume(host, role, mount):
    target = Path(mount["target"])
    before = _owned_directory(target, host.owner_uid, private=role == "control")
    device_path = Path(host.sys_root) / "dev/block" / mount["device"] / "size"
    raw_size = _read(device_path, 128).strip()
    if not re.fullmatch(r"[0-9]+", raw_size):
        raise StorageError("Block device capacity is unavailable")
    capacity = int(raw_size) * 512
    if not 0 < capacity <= VOLUME_LIMITS[role]:
        raise StorageError("Owner volume exceeds its device storage allowance")
    filesystem = _superblock(host, mount, capacity)
    fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        entry = os.fstat(fd)
        if (entry.st_dev, entry.st_ino) != (before.st_dev, before.st_ino):
            raise StorageError("Owner volume changed while opening")
        observed_device = str(os.major(entry.st_dev)) + ":" + str(os.minor(entry.st_dev))
        if observed_device != mount["device"]:
            raise StorageError("Path device differs from the kernel mount record")
        value = os.fstatvfs(fd)
        counters = capacity_counters(capacity, {key: getattr(value, key) for key in (
            "f_frsize", "f_blocks", "f_bfree", "f_bavail", "f_files", "f_ffree", "f_favail", "f_flag")})
    finally:
        os.close(fd)
    after = _owned_directory(target, host.owner_uid, private=role == "control")
    if (after.st_dev, after.st_ino) != (entry.st_dev, entry.st_ino) or _read(device_path, 128).strip() != raw_size:
        raise StorageError("Owner storage device changed during inspection")
    return {
        **counters, "binding": {"target": str(target), "mount_id": mount["mount_id"],
                               "device": mount["device"], "inode": entry.st_ino,
                               "filesystem_uuid_hex": filesystem["filesystem_uuid_hex"]},
        "mount": mount, "kernel_size_sectors": int(raw_size),
        "filesystem": filesystem,
    }


def observe(host, *, root=ROOT, expected_binding=None):
    """Observe current owner volumes, without requiring free data space to stop."""
    root = Path(root)
    started = time.monotonic()
    if not root.is_absolute() or ".." in root.parts:
        raise StorageError("Absolute owner storage root required")
    for path in (root.parent, root):
        _owned_directory(path, host.owner_uid)
    namespace_path = Path(host.proc_root) / "self/ns/mnt"
    namespace = namespace_path.stat()
    mountinfo_path = Path(host.proc_root) / "self/mountinfo"
    raw = _read(mountinfo_path, MAX_MOUNTINFO)
    selected = select_mounts(parse_mountinfo(raw), root=root)
    volumes = {role: _volume(host, role, mount) for role, mount in selected.items()}
    after = select_mounts(parse_mountinfo(_read(mountinfo_path, MAX_MOUNTINFO)), root=root)
    current_namespace = namespace_path.stat()
    if after != selected or (namespace.st_dev, namespace.st_ino) != (current_namespace.st_dev, current_namespace.st_ino):
        raise StorageError("Owner mount namespace changed during inspection")
    binding = {role: value["binding"] for role, value in volumes.items()}
    if expected_binding is not None and expected_binding != binding:
        raise StorageError("Storage differs from the persisted volume identities")
    total = sum(value["capacity_bytes"] for value in volumes.values())
    if total > TOTAL_LIMIT:
        raise StorageError("Combined owner volumes exceed the device allowance")
    return {
        "schema": "organism-native-device-storage-v1", "profile": PROFILE, "binding": binding,
        "boot_id": host.boot_id, "mount_namespace": {"device": namespace.st_dev, "inode": namespace.st_ino},
        "mountinfo_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "volumes": volumes, "capacity_bytes": total,
        "used_including_metadata_bytes": sum(value["used_including_metadata_bytes"] for value in volumes.values()),
        "control_admission_headroom": (volumes["control"]["free_bytes"] >= CONTROL_HEADROOM
                                       and volumes["control"]["free_inodes"] >= CONTROL_INODE_HEADROOM),
        "started_monotonic_seconds": started, "completed_monotonic_seconds": time.monotonic(),
        "native_observation": not host.is_fixture, "process_write_containment_verified": False,
        "qualification_passed": False,
    }


def admit(host, layout, *, root=ROOT, expected_binding=None):
    """Require owner paths and control headroom before new work or admission.

    Owner stop/retirement must not call this function. Missing path suffixes
    may be created later by the trusted loader; every existing component must
    already belong to its declared volume and remain owner controlled.
    """
    layout = validate_layout(layout, root=root)
    core_policy = observe_core_policy(host)
    observation = observe(host, root=root, expected_binding=expected_binding)
    for role, fields in (("control", CONTROL_PATHS), ("data", DATA_PATHS)):
        parent = Path(root) / role
        binding = observation["binding"][role]
        for field in fields:
            path = Path(layout[field])
            components = path.relative_to(parent).parts
            current = parent
            for index, component in enumerate(components):
                current /= component
                try:
                    entry = current.lstat()
                except FileNotFoundError:
                    break
                number = str(os.major(entry.st_dev)) + ":" + str(os.minor(entry.st_dev))
                leaf_file = field in ("enrollment_file", "discovery_file") and index == len(components) - 1
                valid_type = stat.S_ISREG(entry.st_mode) if leaf_file else stat.S_ISDIR(entry.st_mode)
                if (not valid_type or entry.st_uid != host.owner_uid or entry.st_mode & 0o022
                        or number != binding["device"] or leaf_file and entry.st_nlink != 1):
                    raise StorageError("Configured storage path changed ownership, type or volume: " + field)
    if not observation["control_admission_headroom"]:
        raise StorageError("Owner control storage lacks admission headroom")
    if observe_core_policy(host) != core_policy:
        raise StorageError("Kernel core-dump policy changed during storage admission")
    return {**observation, "layout": layout, "configured_paths_verified": True,
            "core_dump_policy": core_policy}
