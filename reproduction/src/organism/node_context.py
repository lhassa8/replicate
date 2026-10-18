"""Owner-prepared device identity outside the immutable reproductive capsule."""

import base64
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from .capsule import canonical
from .discovery_book import DiscoveryBook
from .enrollment import Enrollment, unique_object
from .peer_identity import context


SCHEMA = "organism-node-context-v1"
FILES = frozenset(("enrollment.json", "discovery.json", "key.pem", "certificate.pem", "owner.pem"))
MAX_FILE_BYTES = 65536


class NodeContextError(ValueError):
    pass


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_file(path, *, owner_uid, group=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        entry = os.fstat(source.fileno())
        if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != owner_uid or entry.st_nlink != 1
                or entry.st_mode & 0o022 or entry.st_size > MAX_FILE_BYTES):
            raise NodeContextError("Invalid node-context file")
        if group is not None and (entry.st_gid != group or stat.S_IMODE(entry.st_mode) != 0o440):
            raise NodeContextError("Node context is not read-only to its application group")
        raw = source.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise NodeContextError("Node-context file exceeds its bound")
    return raw


@dataclass(frozen=True)
class NodeContext:
    path: Path
    sha256: str
    policy: Enrollment
    book: DiscoveryBook
    device_id: str
    individual_id: str
    release_id: str
    owner_uid: int
    group: int
    tls: object

    def verify(self):
        return load_context(self.path, self.sha256, device_id=self.device_id, individual_id=self.individual_id,
                            release_id=self.release_id, owner_uid=self.owner_uid, group=self.group)


def load_context(path, expected_sha256, *, device_id, individual_id, release_id, owner_uid=0, group=None):
    group = os.getegid() if group is None else group
    path = Path(path).absolute()
    entry = path.lstat()
    if (not stat.S_ISDIR(entry.st_mode) or entry.st_uid != owner_uid or entry.st_gid != group
            or stat.S_IMODE(entry.st_mode) != 0o550):
        raise NodeContextError("Node context directory is not owner-controlled and group-readable")
    if set(item.name for item in path.iterdir()) != FILES | {"context.json"}:
        raise NodeContextError("Node context has missing or unexpected files")
    raw = read_file(path / "context.json", owner_uid=owner_uid, group=group)
    if digest(raw) != expected_sha256:
        raise NodeContextError("Node context differs from the admitted process command")
    body = json.loads(raw, object_pairs_hook=unique_object)
    fields = {"schema", "device_id", "individual_id", "release_id", "policy_id", "book_id", "owner_public_key", "files"}
    if not isinstance(body, dict) or set(body) != fields or body["schema"] != SCHEMA:
        raise NodeContextError("Invalid node context schema")
    if (body["device_id"], body["individual_id"], body["release_id"]) != (device_id, individual_id, release_id):
        raise NodeContextError("Node context belongs to another device, individual or release")
    if not isinstance(body["files"], dict) or set(body["files"]) != FILES:
        raise NodeContextError("Node context file inventory differs")
    contents = {}
    for name in FILES:
        contents[name] = read_file(path / name, owner_uid=owner_uid, group=group)
        if body["files"][name] != {"bytes": len(contents[name]), "sha256": digest(contents[name])}:
            raise NodeContextError("Node context content changed")
    public = base64.b64decode(body["owner_public_key"], validate=True)
    policy = Enrollment(contents["enrollment.json"], public)
    book = DiscoveryBook(contents["discovery.json"], policy, public)
    if policy.policy_id != body["policy_id"] or book.book_id != body["book_id"]:
        raise NodeContextError("Node context enrollment or discovery identity differs")
    policy.peer_key(device_id)
    tls = context(policy, device_id, path, owner_uid=owner_uid, read_group=group)
    return NodeContext(path, expected_sha256, policy, book, device_id, individual_id, release_id, owner_uid, group, tls)


class NodeContextProvisioner:
    """Copy this device's owner-installed identity, never a received parent key."""

    def __init__(self, directory, policy, device_id, owner_public, signed_book, identity_directory, *, group):
        self.directory = Path(directory).absolute()
        try:
            self.directory.mkdir(mode=0o755)
        except FileExistsError:
            pass
        else:
            self.directory.chmod(0o755)
        entry = self.directory.lstat()
        if (not stat.S_ISDIR(entry.st_mode) or entry.st_uid != os.geteuid() or entry.st_mode & 0o022
                or not entry.st_mode & 0o001):
            raise NodeContextError("Node-context store must be owner-controlled")
        self.policy, self.device_id, self.owner_public, self.group = policy, device_id, owner_public, group
        self.book = DiscoveryBook(signed_book, policy, owner_public)
        self.identity_directory = Path(identity_directory)
        context(policy, device_id, self.identity_directory)  # Validate the owner's local key/certificate pair.
        self.contents = {"enrollment.json": policy.signed, "discovery.json": self.book.signed}
        for name in ("key.pem", "certificate.pem", "owner.pem"):
            self.contents[name] = read_file(self.identity_directory / name, owner_uid=os.geteuid())

    def prepare(self, individual_id, release_id):
        if not re.fullmatch("[0-9a-f]{32}", individual_id) or not re.fullmatch("[0-9a-f]{64}", release_id):
            raise NodeContextError("Invalid context individual or release")
        body = {"schema": SCHEMA, "device_id": self.device_id, "individual_id": individual_id,
                "release_id": release_id, "policy_id": self.policy.policy_id, "book_id": self.book.book_id,
                "owner_public_key": base64.b64encode(self.owner_public).decode(),
                "files": {name: {"bytes": len(raw), "sha256": digest(raw)} for name, raw in self.contents.items()}}
        raw = canonical(body)
        expected = digest(raw)
        lock = os.open(self.directory / ".context.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            entry = os.fstat(lock)
            if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.geteuid()
                    or entry.st_nlink != 1 or entry.st_mode & 0o077):
                raise NodeContextError("Invalid node-context ownership lock")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.directory / individual_id
            if not os.path.lexists(path):
                entries = [item for item in self.directory.iterdir() if item.name != ".context.lock"]
                if (len(entries) >= self.policy.body["max_births_per_device"]
                        or any(not re.fullmatch("[0-9a-f]{32}", item.name) for item in entries)):
                    raise NodeContextError("Unknown contexts or exhausted context allowance require owner inspection")
                pending = Path(tempfile.mkdtemp(prefix=".context-pending-", dir=self.directory))
                try:
                    for name, data in {**self.contents, "context.json": raw}.items():
                        fd = os.open(pending / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                        with os.fdopen(fd, "wb") as output:
                            output.write(data); output.flush()
                            os.fchown(output.fileno(), os.geteuid(), self.group)
                            os.fchmod(output.fileno(), 0o440)
                            os.fsync(output.fileno())
                    os.chown(pending, os.geteuid(), self.group)
                    pending.chmod(0o550)
                    fd = os.open(pending, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    load_context(pending, expected, device_id=self.device_id, individual_id=individual_id,
                                 release_id=release_id, owner_uid=os.geteuid(), group=self.group)
                    os.rename(pending, path)
                    fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                finally:
                    if pending.exists():
                        # Only this operation's unpublished directory is removed.
                        pending.chmod(0o700)
                        for item in pending.iterdir():
                            item.unlink()
                        pending.rmdir()
            return load_context(path, expected, device_id=self.device_id, individual_id=individual_id,
                                release_id=release_id, owner_uid=os.geteuid(), group=self.group)
        finally:
            os.close(lock)
