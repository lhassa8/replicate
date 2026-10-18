"""Use the verified receiver's mount view in a separate native owner observer."""

from contextlib import ExitStack
import os
from pathlib import Path
import platform

from .. import device_resources
from ..systemd_supervision import NativeLinuxHost, process_start_ticks
from ..training_inputs import require


NAMESPACES = ("mnt", "net", "user", "pid", "ipc", "uts", "cgroup")


def namespace(path):
    info = Path(path).stat()
    require(info.st_dev > 0 and info.st_ino > 0, "Namespace identity is unavailable")
    return {"device": info.st_dev, "inode": info.st_ino}


def namespaces(host):
    return {name: namespace(host.proc_root / "self/ns" / name) for name in NAMESPACES}


def receiver_identity(host):
    reply = host.run(["systemctl", "show", device_resources.RECEIVER_UNIT,
                      "--property=MainPID", "--value"])
    require(reply.returncode == 0 and reply.stdout.strip().isdigit(),
            "Owner observer cannot identify the receiving service")
    pid = int(reply.stdout.strip())
    require(pid > 1 and pid != os.getpid(), "Owner observer cannot be the receiver")
    observed = device_resources.verify_receiver(host, pid=pid)
    require(observed["native_observation"] is True, "Fixture receiver cannot bind a native observer")
    raw = device_resources._read(host.proc_root / str(pid) / "stat", maximum=8192)
    return {
        "unit": device_resources.RECEIVER_UNIT, "pid": pid,
        "start_ticks": int(process_start_ticks(raw, pid)),
        "cgroup": observed["receiver"]["properties"]["ControlGroup"],
        "mount_namespace": namespace(host.proc_root / str(pid) / "ns/mnt"),
    }


def validate(value, receiver, *, storage=None):
    require(set(value) == {
        "schema", "observer_pid", "receiver_before", "receiver_after", "before", "after",
        "transition_performed", "native_observation", "qualification_passed",
    } and value["schema"] == "organism-owner-mount-namespace-v1",
        "Unexpected owner mount-namespace evidence")
    identity = value["receiver_before"]
    require(identity == value["receiver_after"]
            and set(identity) == {"unit", "pid", "start_ticks", "cgroup", "mount_namespace"}
            and identity["unit"] == device_resources.RECEIVER_UNIT
            and identity["cgroup"] == device_resources.service_cgroup(
                device_resources.RECEIVER_UNIT, device_resources.SLICE)
            and type(identity["pid"]) is int and identity["pid"] > 1
            and type(identity["start_ticks"]) is int and identity["start_ticks"] > 0
            and {key: identity[key] for key in ("pid", "start_ticks")} == receiver,
            "Owner mount observation belongs to another receiver incarnation")
    require(type(value["observer_pid"]) is int and value["observer_pid"] > 1
            and value["observer_pid"] != identity["pid"]
            and value["native_observation"] is True and value["qualification_passed"] is False,
            "Owner mount observer changed its role or evidence scope")
    before, after = value["before"], value["after"]
    require(set(before) == set(after) == set(NAMESPACES),
            "Owner namespace inventory is incomplete")
    for item in [*before.values(), *after.values(), identity["mount_namespace"]]:
        require(set(item) == {"device", "inode"}
                and all(type(number) is int and number > 0 for number in item.values()),
                "Owner namespace identity is invalid")
    require(after["mnt"] == identity["mount_namespace"]
            and all(before[name] == after[name] for name in NAMESPACES if name != "mnt")
            and type(value["transition_performed"]) is bool
            and value["transition_performed"] == (before["mnt"] != after["mnt"]),
            "Owner observer changed another namespace or missed the receiver mount view")
    if storage is not None:
        require(storage["native_observation"] is True
                and storage["configured_paths_verified"] is True
                and storage["mount_namespace"] == after["mnt"],
                "Owner storage observation used a different mount namespace")
    return value


def join_receiver(host=None):
    """Join only mounts; retain the observer's separate network and process roles."""
    require(platform.system() == "Linux" and os.geteuid() == 0 and hasattr(os, "setns"),
            "Mount observation requires a native Linux root owner with setns")
    host = NativeLinuxHost() if host is None else host
    require(not host.is_fixture and host.owner_uid == 0 and host.proc_root == Path("/proc"),
            "Mount observation requires the actual native proc filesystem")
    require([path.name for path in (host.proc_root / "self/task").iterdir()] == [str(os.getpid())],
            "Join the receiver mount view before starting observer threads")
    before = namespaces(host)
    receiver = receiver_identity(host)
    changed = before["mnt"] != receiver["mount_namespace"]
    transitioned = False
    with ExitStack() as cleanup:
        descriptors = []
        for path in (host.proc_root / "self/ns/mnt",
                     host.proc_root / str(receiver["pid"]) / "ns/mnt"):
            # Namespace descriptors are opened through the kernel's proc links.
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            cleanup.callback(os.close, fd)
            descriptors.append(fd)
        original_fd, receiver_fd = descriptors
        for fd, expected in zip(descriptors, (before["mnt"], receiver["mount_namespace"])):
            info = os.fstat(fd)
            require({"device": info.st_dev, "inode": info.st_ino} == expected,
                    "Namespace descriptor changed while opening")
        require(receiver_identity(host) == receiver, "Receiver changed before the owner mount join")
        try:
            if changed:
                os.setns(receiver_fd, os.CLONE_NEWNS)
                transitioned = True
            value = {
                "schema": "organism-owner-mount-namespace-v1", "observer_pid": os.getpid(),
                "receiver_before": receiver, "receiver_after": receiver_identity(host),
                "before": before, "after": namespaces(host), "transition_performed": changed,
                "native_observation": True, "qualification_passed": False,
            }
            return validate(value, {key: receiver[key] for key in ("pid", "start_ticks")})
        except BaseException:
            if transitioned:
                os.setns(original_fd, os.CLONE_NEWNS)
                require(namespaces(host) == before, "Failed owner mount join could not restore its original view")
            raise
