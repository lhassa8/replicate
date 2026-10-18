"""Owner-declared Linux network-namespace identity, independent of task logic."""

import fcntl
import os
from pathlib import Path
import re
import stat


NS_GET_NSTYPE = 0xB703
CLONE_NEWNET = 0x40000000


def namespace_path(value):
    value = str(value)
    if not re.fullmatch(r"/run/netns/[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", value):
        raise ValueError("Expected a named owner network namespace under /run/netns")
    return value


def inspect_namespace(path, *, proc_root=Path("/proc"), owner_uid=0):
    path = Path(namespace_path(path))
    for parent in (path.parent.parent, path.parent):
        entry = parent.lstat()
        if not stat.S_ISDIR(entry.st_mode) or entry.st_uid != owner_uid or entry.st_mode & 0o022:
            raise ValueError("Network namespace parent is not owner-controlled")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        entry = os.fstat(fd)
        if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != owner_uid
                or fcntl.ioctl(fd, NS_GET_NSTYPE) != CLONE_NEWNET):
            raise ValueError("Expected an owner-created Linux network namespace")
        host = (proc_root / "1/ns/net").stat()
        if (entry.st_dev, entry.st_ino) == (host.st_dev, host.st_ino):
            raise ValueError("The host network namespace is not a private lab namespace")
        return {"path": str(path), "device": entry.st_dev, "inode": entry.st_ino,
                "host_device": host.st_dev, "host_inode": host.st_ino}
    finally:
        os.close(fd)


def verify_process_namespace(binding, pid, *, proc_root=Path("/proc"), owner_uid=0):
    if type(pid) is not int or pid <= 1:
        raise ValueError("Invalid process namespace observation")
    if inspect_namespace(binding["path"], proc_root=proc_root, owner_uid=owner_uid) != binding:
        raise ValueError("Owner network namespace changed after the start intent")
    process = (proc_root / str(pid) / "ns/net").stat()
    if (process.st_dev, process.st_ino) != (binding["device"], binding["inode"]):
        raise ValueError("Observed process is outside its owner-declared network namespace")
    return {"pid": pid, "device": process.st_dev, "inode": process.st_ino}
