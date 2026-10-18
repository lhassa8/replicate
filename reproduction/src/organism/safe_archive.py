"""Bounded runtime extraction into a new private directory.

Only regular files, directories and resolvable internal symbolic links are
supported. No archive ownership, special files, hard links, sparse files or
replacement entries are applied. Links are created only after all files.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import gzip
import hashlib
import os
from pathlib import Path
import re
import shutil
import tarfile


class ArchiveError(ValueError):
    pass


@dataclass(frozen=True)
class ArchiveLimits:
    bytes: int = 512 * 1024**2
    members: int = 30000
    metadata_bytes: int = 64 * 1024**2

    def validate(self):
        for value, maximum in ((self.bytes, 1024**3), (self.members, 50000),
                               (self.metadata_bytes, 64 * 1024**2)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ArchiveError("Invalid archive resource allowance")


def safe_relative(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 1024:
        raise ArchiveError("Invalid archive path")
    parts = value.split("/")
    if any(part in ("", ".", "..") or not re.fullmatch(r"[A-Za-z0-9_.+-]{1,160}", part) for part in parts):
        raise ArchiveError("Unsafe or noncanonical archive path")
    return value


def link_destination(name, target):
    if not isinstance(target, str) or not 1 <= len(target) <= 1024 or target.startswith("/"):
        raise ArchiveError("Invalid archive link")
    parts = name.split("/")[:-1]
    for item in target.split("/"):
        if item == "..":
            if not parts:
                raise ArchiveError("Archive link escapes its component")
            parts.pop()
        elif item == ".":
            continue
        elif not re.fullmatch(r"[A-Za-z0-9_.+-]{1,160}", item):
            raise ArchiveError("Unsafe archive link")
        else:
            parts.append(item)
    if not parts:
        raise ArchiveError("Archive link targets the component root")
    return "/".join(parts)


class BoundedReader:
    """Limit expanded tar bytes, including headers, metadata and trailing data."""
    def __init__(self, source, maximum):
        self.source, self.maximum, self.read_bytes = source, maximum, 0

    def read(self, size):
        if size < 0:
            raise ArchiveError("Unbounded archive read is not permitted")
        result = self.source.read(min(size, 65536, self.maximum - self.read_bytes + 1))
        self.read_bytes += len(result)
        if self.read_bytes > self.maximum:
            raise ArchiveError("Expanded archive stream exceeds its allowance")
        return result


def remove_new_tree(path):
    """Only for a freshly created, privately owned preparation directory."""
    for parent, directories, _ in os.walk(path, followlinks=False):
        os.chmod(parent, 0o700)
        for name in directories:
            child = Path(parent) / name
            if not child.is_symlink():
                child.chmod(0o700)
    shutil.rmtree(path)


def case_sensitive_directory(directory):
    """Probe the new empty destination; never assume the host's filesystem."""
    probe = Path(directory) / ".CaseSensitivityProbe"
    with probe.open("xb") as target:
        target.write(b"")
    try:
        return not (Path(directory) / ".casesensitivityprobe").exists()
    finally:
        probe.unlink()


def extract_archive(source, destination, *, limits=ArchiveLimits(), strip_prefix=None, allowed_roots=None):
    """Extract without executing anything; destination must not already exist."""
    limits.validate()
    source, destination = Path(source), Path(destination)
    if strip_prefix is not None:
        safe_relative(strip_prefix)
        if "/" in strip_prefix:
            raise ArchiveError("Only one explicit archive root may be stripped")
    if source.is_symlink() or not source.is_file():
        raise ArchiveError("Archive input must be a regular file")
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    entries, explicit, folded = {}, set(), {}
    count, total = 0, 0
    prefix_seen = False
    case_sensitive = False

    def register(name, kind):
        names = name.split("/")
        for index in range(1, len(names) + 1):
            relative = "/".join(names[:index])
            expected = kind if index == len(names) else "directory"
            prior = folded.setdefault(relative.casefold(), relative)
            if prior != relative and not case_sensitive:
                raise ArchiveError("Archive paths collide on a case-insensitive filesystem")
            if relative in entries and entries[relative]["type"] != expected:
                raise ArchiveError("Archive file, directory or link paths overlap")
            entries.setdefault(relative, {"type": expected})
            if expected == "directory":
                (destination / relative).mkdir(mode=0o700, exist_ok=True)

    try:
        case_sensitive = case_sensitive_directory(destination)
        with ExitStack() as stack:
            raw = stack.enter_context(source.open("rb"))
            compressed = raw.read(2) == b"\x1f\x8b"
            raw.seek(0)
            decoded = stack.enter_context(gzip.GzipFile(fileobj=raw)) if compressed else raw
            bounded = BoundedReader(decoded, limits.bytes + limits.metadata_bytes)
            archive = stack.enter_context(tarfile.open(fileobj=bounded, mode="r|", encoding="utf-8", errors="strict"))
            for member in archive:
                count += 1
                if count > limits.members:
                    raise ArchiveError("Archive member allowance exhausted")
                original = safe_relative(member.name.rstrip("/") if member.isdir() else member.name)
                if member.mode & 0o7000 or member.sparse is not None:
                    raise ArchiveError("Privileged modes and sparse archive entries are not supported")
                if not (member.isfile() or member.isdir() or member.issym()):
                    raise ArchiveError("Unsupported archive entry type")
                if member.size < 0 or (not member.isfile() and member.size != 0):
                    raise ArchiveError("Invalid archive entry length")
                if strip_prefix:
                    if original == strip_prefix and member.isdir():
                        if prefix_seen:
                            raise ArchiveError("Duplicate archive entry")
                        prefix_seen = True
                        continue
                    if not original.startswith(strip_prefix + "/"):
                        raise ArchiveError("Archive contains an unexpected top-level path")
                    name = original[len(strip_prefix) + 1:]
                else:
                    name = original
                if allowed_roots and name.split("/")[0] not in allowed_roots:
                    raise ArchiveError("Archive contains an unexpected top-level path")
                if name in explicit:
                    raise ArchiveError("Duplicate archive entry")
                explicit.add(name)
                kind = "file" if member.isfile() else ("directory" if member.isdir() else "symlink")
                register(name, kind)
                target = destination / name
                if member.isfile():
                    total += member.size
                    if total > limits.bytes:
                        raise ArchiveError("Extracted file allowance exhausted")
                    digest, remaining = hashlib.sha256(), member.size
                    with archive.extractfile(member) as content, target.open("xb") as output:
                        while remaining:
                            block = content.read(min(remaining, 65536))
                            if not block:
                                raise ArchiveError("Truncated archive file")
                            remaining -= len(block)
                            digest.update(block)
                            output.write(block)
                        if content.read(1):
                            raise ArchiveError("Archive file exceeds its declared length")
                        output.flush()
                        os.fsync(output.fileno())
                        os.fchmod(output.fileno(), 0o755 if member.mode & 0o100 else 0o644)
                    entries[name].update(bytes=member.size, sha256=digest.hexdigest(),
                                         executable=bool(member.mode & 0o100))
                elif member.issym():
                    entries[name].update(target=member.linkname, destination=link_destination(name, member.linkname))
            # Drain remaining compressed data to validate the gzip trailer and
            # apply the stream ceiling even to padding after the tar terminator.
            while bounded.read(65536):
                pass
        for name, entry in entries.items():
            if entry["type"] != "symlink":
                continue
            seen, current = {name}, entry["destination"]
            for _ in range(40):
                if current in seen:
                    raise ArchiveError("Archive links contain a cycle")
                seen.add(current)
                if current not in entries:
                    raise ArchiveError("Archive link target is missing")
                value = entries[current]
                if value["type"] != "symlink":
                    break
                current = value["destination"]
            else:
                raise ArchiveError("Archive link chain exceeds its allowance")
            # All parent paths were registered as directories. No archive
            # member is ever written through a link.
            os.symlink(entry["target"], destination / name)
        if not any(entry["type"] == "file" for entry in entries.values()):
            raise ArchiveError("Archive has no regular content")
        return {"members": count, "regular_bytes": total, "expanded_stream_bytes": bounded.read_bytes,
                "files": sum(entry["type"] == "file" for entry in entries.values()),
                "links": sum(entry["type"] == "symlink" for entry in entries.values()),
                "case_sensitive_filesystem": case_sensitive}
    except BaseException:
        remove_new_tree(destination)
        raise
