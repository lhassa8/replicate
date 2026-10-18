"""Standalone owner guard: freeze only the fixed device slice, then always thaw.

The guard owns the freeze write. Loss of its owner's input pipe, a requested
thaw or the fixed lease expiry releases the slice. It runs outside that slice
in a separate session. No model, peer or unprivileged process may invoke it.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import stat
import sys
import time


CGROUP = "/system.slice/system-organism.slice"
ROOT = Path("/sys/fs/cgroup")
LOCK = Path("/run/organism-census.lock")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read_at(directory, name):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
    try:
        raw = os.read(fd, 4097)
        require(len(raw) <= 4096, "Guard kernel input exceeds its bound")
        return raw.decode()
    finally:
        os.close(fd)


def events(directory):
    values = {}
    for line in read_at(directory, "cgroup.events").splitlines():
        key, value = line.split()
        require(key not in values and value.isdigit(), "Invalid guard event counters")
        values[key] = int(value)
    return values


def write_state(descriptor, directory, value):
    os.lseek(descriptor, 0, os.SEEK_SET)
    require(os.write(descriptor, str(value).encode()) == 1, "Incomplete freeze-control write")
    until = time.monotonic() + 2
    while time.monotonic() < until:
        if events(directory).get("frozen") == value:
            return
        time.sleep(0.01)
    raise TimeoutError("Device slice did not complete its freeze transition")


def receive(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not select.select([0], [], [], remaining)[0]:
        return "lease_expired"
    raw = os.read(0, 129)
    if not raw:
        return "owner_pipe_closed"
    require(raw in (b"freeze\n", b"thaw\n"), "Invalid owner guard command")
    return raw.decode().strip()


def emit(value):
    try:
        os.write(1, (json.dumps(value) + "\n").encode())
    except BrokenPipeError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--inode", type=int, required=True)
    parser.add_argument("--lease-seconds", type=int, default=10)
    args = parser.parse_args()
    require(sys.platform == "linux" and os.geteuid() == 0, "Only the native Linux owner may freeze the device")
    require(3 <= args.lease_seconds <= 10, "The census freeze lease must be three to ten seconds")
    membership = Path("/proc/self/cgroup").read_text().splitlines()
    require(len(membership) == 1 and membership[0].startswith("0::")
            and not (membership[0][3:] == CGROUP or membership[0][3:].startswith(CGROUP + "/")),
            "The freeze guard must run outside the device slice")
    lock = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    directory = descriptor = None
    freeze_attempted = False
    reason = "not_requested"
    try:
        entry = os.fstat(lock)
        require(stat.S_ISREG(entry.st_mode) and entry.st_uid == 0 and entry.st_nlink == 1
                and stat.S_IMODE(entry.st_mode) == 0o600, "Owner freeze lock identity differs")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        for component in Path(CGROUP).parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=directory)
            os.close(directory)
            directory = child
            require(os.fstat(directory).st_uid == 0, "Guard cgroup is not owner controlled")
        entry = os.fstat(directory)
        require((entry.st_dev, entry.st_ino) == (args.device, args.inode), "Guard cgroup identity changed")
        require(read_at(directory, "cgroup.type").strip() == "domain"
                and read_at(directory, "cgroup.freeze").strip() == "0"
                and events(directory).get("frozen") == 0, "Guard cannot take over an existing freeze")
        descriptor = os.open("cgroup.freeze", os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        deadline = time.monotonic() + args.lease_seconds
        emit({"phase": "armed", "pid": os.getpid(), "owner_pid": os.getppid(),
              "cgroup": CGROUP, "device": entry.st_dev, "inode": entry.st_ino,
              "deadline_monotonic": deadline, "lease_seconds": args.lease_seconds})
        reason = receive(deadline)
        if reason == "freeze":
            require(time.monotonic() < deadline, "Freeze command arrived after its lease")
            freeze_attempted = True
            write_state(descriptor, directory, 1)
            emit({"phase": "frozen", "at_monotonic": time.monotonic()})
            reason = receive(deadline)
            require(reason in ("thaw", "owner_pipe_closed", "lease_expired"), "Unexpected command during freeze")
    finally:
        if freeze_attempted:
            write_state(descriptor, directory, 0)
            emit({"phase": "thawed", "reason": reason, "at_monotonic": time.monotonic()})
        else:
            emit({"phase": "not_frozen", "reason": reason})
        for fd in (descriptor, directory, lock):
            if fd is not None:
                os.close(fd)


if __name__ == "__main__":
    main()
