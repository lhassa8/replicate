"""Bounded temporary paths and output files for owner-supervised processes.

This verifies named write destinations, resolves the complete mounted view,
and samples the required disabled core-dump policy. It does not verify every
write channel. Full device qualification also needs receiver containment,
inherited descriptors, descendants and crash behavior.
"""

import hashlib
import os
from pathlib import Path
import re
import stat

from .device_storage import MAX_MOUNTINFO, StorageError, observe_core_policy, parse_mountinfo


FILE_MAX = 64 * 1024**2
DIRECTORIES = ("tmp", "var-tmp", "cache", "home")
HIDDEN_PATHS = ("/run/systemd/journal", "/run/dbus", "/dev/shm", "/dev/mqueue", "/dev/hugepages", "/dev/log")


def identity(entry):
    return {"device": entry.st_dev, "inode": entry.st_ino}


def _directory(path, uid, gid, *, create=False):
    if create:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        else:
            os.chown(path, uid, gid, follow_symlinks=False)
            for directory in (path, path.parent):
                fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
    entry = path.lstat()
    if (not stat.S_ISDIR(entry.st_mode) or entry.st_uid != uid or entry.st_gid != gid
            or stat.S_IMODE(entry.st_mode) != 0o700):
        raise StorageError("Runtime directory is not private to its declared owner")
    return identity(entry)


def prepare(work, volume_binding, token, *, uid, gid, owner_uid, owner_gid):
    """Create only this incarnation's paths; never clear existing work or logs."""
    if not re.fullmatch("[0-9a-f]{32}", token):
        raise StorageError("Invalid runtime storage incarnation")
    work = Path(work)
    data = Path(volume_binding["data"]["target"])
    if (not work.is_relative_to(data) or work.resolve() != work
            or ":" in str(work) or any(ord(char) < 32 or ord(char) == 127 for char in str(work))):
        raise StorageError("Runtime work is outside its declared data volume")
    work_identity = _directory(work, uid, gid)
    number = str(os.major(work_identity["device"])) + ":" + str(os.minor(work_identity["device"]))
    if number != volume_binding["data"]["device"]:
        raise StorageError("Runtime work is on a different device")
    runtime = work / ".runtime"
    _directory(runtime, uid, gid, create=True)
    directories = {}
    for name in DIRECTORIES:
        path = runtime / name
        directories[name] = {"path": str(path), **_directory(path, uid, gid, create=True)}
    logs = data / "logs"
    _directory(logs, owner_uid, owner_gid, create=True)
    logs /= token
    _directory(logs, owner_uid, owner_gid, create=True)
    outputs = {}
    for name in ("stdout", "stderr"):
        path = logs / name
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            pass
        else:
            try:
                os.fchown(fd, owner_uid, owner_gid)
                os.fsync(fd)
            finally:
                os.close(fd)
        entry = path.lstat()
        if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != owner_uid or entry.st_nlink != 1
                or stat.S_IMODE(entry.st_mode) != 0o600 or entry.st_size > FILE_MAX
                or entry.st_dev != work_identity["device"]):
            raise StorageError("Runtime output is not a bounded owner file on the data volume")
        outputs[name] = {"path": str(path), **identity(entry)}
    fd = os.open(logs, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"schema": "organism-runtime-storage-v1", "directories": directories, "outputs": outputs,
            "working_directory": {"path": str(work), **work_identity}, "file_max_bytes": FILE_MAX}


def environment(value):
    if value is None:
        return {"HOME": "/nonexistent"}
    paths = {name: entry["path"] for name, entry in value["directories"].items()}
    return {"HOME": paths["home"], "XDG_CACHE_HOME": paths["cache"], "HF_HOME": paths["cache"],
            "TMPDIR": paths["tmp"], "TMP": paths["tmp"], "TEMP": paths["tmp"]}


def directives(value, quote):
    if value is None:
        return []
    paths = value["directories"]
    return [
        "PrivateTmp=no",
        "BindPaths=" + " ".join(quote(paths[name]["path"]) + ":" + quote(target) + ":norbind"
                                for name, target in (("tmp", "/tmp"), ("var-tmp", "/var/tmp"))),
        "InaccessiblePaths=" + " ".join("-" + path for path in HIDDEN_PATHS),
        "ReadOnlyPaths=/run /dev /proc /sys",
        "BindReadOnlyPaths=/proc:/proc /sys:/sys /dev/pts:/dev/pts",
        "MemoryPressureWatch=off",
        "LimitCORE=0", "RestrictNamespaces=yes",
    ]


def inspect_mounts(raw, value, *, visible_mount_ids=None):
    """Account for reachable mounts; covered layers remain recorded separately."""
    mounts = parse_mountinfo(raw)
    ids = {mount["mount_id"] for mount in mounts}
    visible = ids if visible_mount_ids is None else set(visible_mount_ids)
    if not visible or not visible <= ids:
        raise StorageError("Runtime mount visibility does not match its table")
    work = value["working_directory"]
    device = str(os.major(work["device"])) + ":" + str(os.minor(work["device"]))
    allowed = {"/tmp", "/var/tmp", work["path"]}
    writable = []
    for mount in mounts:
        options = set(mount["options"])
        if ("ro" in options) == ("rw" in options):
            raise StorageError("Runtime mount access mode is malformed")
        if mount["mount_id"] not in visible:
            continue
        if "ro" in options:
            continue
        if (mount["target"] not in allowed or mount["device"] != device
                or mount["filesystem"] != "ext4" or not {"nosuid", "nodev"} <= options
                or "rw" not in mount["super_options"] or "ro" in mount["super_options"]):
            raise StorageError("Runtime has an undeclared writable mount: " + mount["target"])
        writable.append(mount)
    if ({mount["target"] for mount in writable} != allowed
            or any(sum(mount["target"] == target and mount["mount_id"] in visible
                       for mount in mounts) != 1 for target in allowed)):
        raise StorageError("Runtime writable mounts are missing or overmounted")
    return {"schema": "organism-runtime-mounts-v1",
            "mount_count": len(mounts), "mountinfo_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "visible_mount_ids": sorted(visible), "covered_mount_ids": sorted(ids - visible),
            "writable_mounts": writable, "writable_mounts_verified": True}


def _mount_visibility(host, process, raw):
    """Resolve every listed target through the process root and read its actual mount ID."""
    table = parse_mountinfo(raw)
    by_id = {mount["mount_id"]: mount for mount in table}
    paths = sorted({mount["target"] for mount in table})
    if len(paths) > 1024 or not hasattr(os, "O_PATH"):
        raise StorageError("Runtime mount resolution is unsupported or exceeds its bound")
    resolved = {}
    for target in paths:
        if Path(target).as_posix() != target or ".." in Path(target).parts:
            raise StorageError("Runtime mount target is malformed")
        try:
            fd = os.open(str(process / "root") + target,
                         os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                entry = os.fstat(fd)
                if stat.S_ISLNK(entry.st_mode):
                    raise StorageError("Runtime mount target resolves to a symlink")
                with (Path(host.proc_root) / "self/fdinfo" / str(fd)).open("rb") as stream:
                    info = stream.read(4097)
                if len(info) > 4096:
                    raise StorageError("Runtime descriptor metadata exceeds its bound")
                ids = re.findall(rb"^mnt_id:\s*([0-9]+)\s*$", info, re.MULTILINE)
                if len(ids) != 1:
                    raise StorageError("Runtime target mount ID is unavailable")
                mount_id = int(ids[0])
                actual = by_id.get(mount_id)
                device = str(os.major(entry.st_dev)) + ":" + str(os.minor(entry.st_dev))
                if actual is None or actual["device"] != device:
                    raise StorageError("Runtime target mount identity differs from its table")
                resolved[target] = mount_id
            finally:
                os.close(fd)
        except OSError as exc:
            raise StorageError("Runtime mount target could not be resolved: " + target) from exc
    return resolved


def _mountinfo(process):
    try:
        with (process / "mountinfo").open("rb") as stream:
            raw = stream.read(MAX_MOUNTINFO + 1)
        if len(raw) > MAX_MOUNTINFO:
            raise StorageError("Runtime mount table exceeds its bound")
        return raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise StorageError("Runtime mount table could not be observed") from exc


def observe_process(host, pid, value, *, output_paths=None, file_max=FILE_MAX):
    """Recheck mounted paths, cwd, environment and the two output descriptors."""
    if type(pid) is not int or pid <= 1:
        raise StorageError("Runtime storage requires a live process identity")
    core_policy = observe_core_policy(host)
    process = Path(host.proc_root) / str(pid)
    namespace = (process / "ns/mnt").stat()
    before_stat = (process / "stat").read_bytes()
    before_mounts = _mountinfo(process)
    resolved = _mount_visibility(host, process, before_mounts)
    mounts = inspect_mounts(before_mounts, value, visible_mount_ids=resolved.values())
    root = process / "root"
    if not os.statvfs(root / "run").f_flag & os.ST_RDONLY:
        raise StorageError("Runtime /run filesystem is not read-only")
    hidden = {}
    for path in HIDDEN_PATHS:
        target = root / path.lstrip("/")
        try:
            entry = target.lstat()
        except FileNotFoundError:
            hidden[path] = "absent"
        else:
            # PrivateDevices retains this conventional symlink. Its permissions
            # are not access permissions for the target; require the exact
            # target beneath the directory already observed as masked.
            if (path == "/dev/log" and stat.S_ISLNK(entry.st_mode) and entry.st_uid == host.owner_uid
                    and os.readlink(target) == "/run/systemd/journal/dev-log"
                    and isinstance(hidden.get("/run/systemd/journal"), dict)
                    and hidden["/run/systemd/journal"]["mode"] == 0):
                hidden[path] = {"symlink": "/run/systemd/journal/dev-log", "target_parent_masked": True}
                continue
            if entry.st_uid != host.owner_uid or stat.S_IMODE(entry.st_mode) != 0:
                raise StorageError("Runtime masked path remains accessible: " + path
                                   + " (mode=" + oct(stat.S_IMODE(entry.st_mode)) + ")")
            hidden[path] = {"mode": 0, **identity(entry)}
    entries = {"/tmp": value["directories"]["tmp"], "/var/tmp": value["directories"]["var-tmp"]}
    entries.update({item["path"]: item for item in value["directories"].values()})
    work = value["working_directory"]
    entries[work["path"]] = work
    checked = {}
    for destination, expected in entries.items():
        target = root / destination.lstrip("/")
        entry = target.stat()
        if not stat.S_ISDIR(entry.st_mode) or identity(entry) != {key: expected[key] for key in ("device", "inode")}:
            raise StorageError("Runtime writable directory differs from its prepared identity")
        if os.statvfs(target).f_flag & os.ST_RDONLY:
            raise StorageError("Runtime write destination is unexpectedly read-only")
        checked[destination] = identity(entry)
    if identity((process / "cwd").stat()) != {key: work[key] for key in ("device", "inode")}:
        raise StorageError("Runtime working directory differs from its start intent")
    raw_environment = (process / "environ").read_bytes()
    if len(raw_environment) > 65536:
        raise StorageError("Runtime environment capture exceeds its bound")
    actual = {}
    for field in raw_environment.split(b"\0"):
        if not field:
            continue
        key, separator, item = field.partition(b"=")
        if not separator or key in actual:
            raise StorageError("Runtime environment is malformed")
        actual[key] = item
    if any(actual.get(key.encode()) != item.encode() for key, item in environment(value).items()):
        raise StorageError("Runtime temporary or cache environment differs")
    outputs = {}
    for number, name in ((1, "stdout"), (2, "stderr")):
        expected = value["outputs"][name] if output_paths is None else {"path": str(output_paths[name])}
        destination = Path(expected["path"])
        target = destination.lstat()
        descriptor = (process / "fd" / str(number)).stat()
        if (not stat.S_ISREG(target.st_mode) or target.st_uid != host.owner_uid or target.st_nlink != 1
                or stat.S_IMODE(target.st_mode) != 0o600 or target.st_size > file_max
                or identity(descriptor) != identity(target)
                or output_paths is None and identity(target) != {key: expected[key] for key in ("device", "inode")}):
            raise StorageError("Runtime output descriptor differs from its bounded owner capture")
        outputs[name] = {"path": str(destination), **identity(target), "bytes": target.st_size}
    limits = (process / "limits").read_text()
    for label, maximum in (("Max file size", file_max), ("Max core file size", 0)):
        matches = [line[len(label):].split() for line in limits.splitlines() if line.startswith(label + " ")]
        if len(matches) != 1 or matches[0][:2] != [str(maximum), str(maximum)]:
            raise StorageError("Runtime file or core resource limit differs")
    # Starttime is field 22; the comm field may contain spaces and parentheses.
    def start_identity(raw):
        end = raw.rfind(b")")
        fields = raw[end + 1:].split()
        if end < 1 or len(fields) < 20 or not fields[19].isdigit():
            raise StorageError("Runtime process identity is unavailable")
        return raw[:raw.find(b"(")], fields[19]
    if (identity((process / "ns/mnt").stat()) != identity(namespace)
            or start_identity((process / "stat").read_bytes()) != start_identity(before_stat)):
        raise StorageError("Runtime process changed during storage inspection")
    if (_mountinfo(process) != before_mounts
            or _mount_visibility(host, process, before_mounts) != resolved):
        raise StorageError("Runtime mount table changed during storage inspection")
    if observe_core_policy(host) != core_policy:
        raise StorageError("Kernel core-dump policy changed during process inspection")
    return {"schema": "organism-runtime-storage-observation-v1", "pid": pid,
            "mount_namespace": identity(namespace), "directories": checked, "outputs": outputs,
            "mounts": mounts,
            "run_read_only": True, "masked_paths": hidden, "core_dump_policy": core_policy,
            "named_write_destinations_verified": True, "native_observation": not host.is_fixture,
            "all_process_writes_contained": False, "qualification_passed": False}
