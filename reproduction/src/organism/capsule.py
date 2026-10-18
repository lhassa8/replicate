"""Signed, bounded capsule verification and atomic local staging.

This is the payload-integrity layer, not enrollment, peer transport, process
activation, or qualification. Trust keys come from the owner's receiving
substrate; a capsule cannot supply its own trust anchor.
"""

from contextlib import contextmanager
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


SCHEMA = "organism-capsule-v1"
PLATFORMS = {"linux-x86_64", "linux-arm64"}
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PAYLOAD_BYTES = 8 * 1024**3
MAX_FILES = 128
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class CapsuleError(ValueError):
    pass


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CapsuleError("Duplicate JSON key")
        result[key] = value
    return result


def _exact_keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise CapsuleError("Unexpected manifest fields")


def _path(value):
    if not isinstance(value, str) or len(value) > 384:
        raise CapsuleError("Invalid payload path")
    parts = value.split("/")
    if any(not _PART.fullmatch(part) or part in {".", ".."} for part in parts):
        raise CapsuleError("Noncanonical or unsafe payload path")
    if str(PurePosixPath(value)) != value:
        raise CapsuleError("Noncanonical payload path")
    return value


def validate_body(body: dict, *, require_trained: bool = True, max_payload_bytes: int = MAX_PAYLOAD_BYTES):
    _exact_keys(body, {
        "schema", "experiment_id", "model", "platforms", "agent", "knowledge",
        "self_test", "notices", "files",
    })
    if body["schema"] != SCHEMA or not isinstance(body["experiment_id"], str) or not 1 <= len(body["experiment_id"]) <= 128:
        raise CapsuleError("Unsupported capsule schema or experiment identity")
    files = body["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise CapsuleError("Payload inventory exceeds bounds")
    total = 0
    for name, entry in files.items():
        _path(name)
        _exact_keys(entry, {"bytes", "sha256"})
        if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
            raise CapsuleError("Invalid payload size")
        if not isinstance(entry["sha256"], str) or not _HEX.fullmatch(entry["sha256"]):
            raise CapsuleError("Invalid payload hash")
        total += entry["bytes"]
    if total > max_payload_bytes:
        raise CapsuleError("Capsule exceeds storage allowance")
    model = body["model"]
    _exact_keys(model, {"path", "parameters", "trained_for_organism", "source_model", "source_revision", "quantization"})
    if type(model["parameters"]) is not int or not 2_000_000_000 <= model["parameters"] <= 4_000_000_000:
        raise CapsuleError("Declared model is outside the required parameter range")
    if type(model["trained_for_organism"]) is not bool or (require_trained and not model["trained_for_organism"]):
        raise CapsuleError("A trained organism model is required by this receiving policy")
    for key in ("source_model", "source_revision", "quantization"):
        if not isinstance(model[key], str) or not 1 <= len(model[key]) <= 256:
            raise CapsuleError("Missing model provenance")
    platforms = body["platforms"]
    if not isinstance(platforms, dict) or set(platforms) != PLATFORMS:
        raise CapsuleError("Both supported platform packages are required")
    referenced = [model["path"]] + [body[key] for key in ("agent", "knowledge", "self_test", "notices")]
    for platform in sorted(platforms):
        _exact_keys(platforms[platform], {"python", "inference", "dependencies"})
        referenced.extend(platforms[platform].values())
    if len(referenced) != len(set(referenced)):
        raise CapsuleError("Each capsule component needs its own payload")
    if {_path(name) for name in referenced} != set(files):
        raise CapsuleError("Incomplete or unreferenced capsule payloads")
    return total


def sign_manifest(body: dict, private_key: Ed25519PrivateKey, *, require_trained: bool = True) -> bytes:
    validate_body(body, require_trained=require_trained)
    payload = canonical(body)
    public = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return canonical({
        "body": body,
        "release_id": hashlib.sha256(payload).hexdigest(),
        "signer_id": hashlib.sha256(public).hexdigest(),
        "signature": base64.b64encode(private_key.sign(payload)).decode("ascii"),
    })


def verify_manifest(data: bytes, trusted_keys: dict[str, bytes], *, require_trained=True, max_payload_bytes=MAX_PAYLOAD_BYTES):
    if not 0 < len(data) <= MAX_MANIFEST_BYTES:
        raise CapsuleError("Manifest exceeds bounds")
    try:
        envelope = json.loads(data, object_pairs_hook=_unique_object)
        _exact_keys(envelope, {"body", "release_id", "signer_id", "signature"})
        signer = envelope["signer_id"]
        if not isinstance(signer, str) or signer not in trusted_keys:
            raise CapsuleError("Unknown release signer")
        public = trusted_keys[signer]
        if hashlib.sha256(public).hexdigest() != signer:
            raise CapsuleError("Owner trust-store key identity mismatch")
        payload = canonical(envelope["body"])
        if hashlib.sha256(payload).hexdigest() != envelope["release_id"]:
            raise CapsuleError("Release identity mismatch")
        signature = base64.b64decode(envelope["signature"], validate=True)
        Ed25519PublicKey.from_public_bytes(public).verify(signature, payload)
        validate_body(envelope["body"], require_trained=require_trained, max_payload_bytes=max_payload_bytes)
    except CapsuleError:
        raise
    except (InvalidSignature, TypeError, KeyError, UnicodeError, ValueError, RecursionError) as exc:
        raise CapsuleError("Invalid signed manifest") from exc
    return envelope


@contextmanager
def _read_beneath(root_fd: int, relative: str):
    """Open each component relative to an owned descriptor; never follow links."""
    parts = _path(relative).split("/")
    directory = os.dup(root_fd)
    source = None
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        source = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(source).st_mode):
            raise CapsuleError("Payload must be a regular file")
        with os.fdopen(source, "rb") as stream:
            source = None
            yield stream
    finally:
        if source is not None:
            os.close(source)
        os.close(directory)


def stage_capsule(
    source: Path, store: Path, manifest: bytes, trusted_keys: dict[str, bytes],
    *, require_trained=True, max_payload_bytes=MAX_PAYLOAD_BYTES,
) -> Path:
    """Copy and hash every signed component, then publish a complete directory.

    A device-wide advisory lock serializes cooperative receiving processes.
    It releases on process death. A crash can leave an uncommitted `.pending-`
    directory; it is never returned or eligible for activation.
    """
    envelope = verify_manifest(
        manifest, trusted_keys, require_trained=require_trained, max_payload_bytes=max_payload_bytes,
    )
    store.mkdir(mode=0o700, parents=True, exist_ok=True)
    store_fd = os.open(store, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    source_fd = lock_fd = None
    pending = None
    try:
        lock_fd = os.open(".install.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=store_fd)
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise CapsuleError("Invalid receiving lock")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CapsuleError("Another capsule installation is in progress") from exc
        destination = store / envelope["release_id"]
        if os.path.lexists(destination):
            raise CapsuleError("Release already exists; do not overwrite it")
        total = sum(entry["bytes"] for entry in envelope["body"]["files"].values())
        if shutil.disk_usage(store).free < total + 64 * 1024**2:
            raise CapsuleError("Insufficient free space for bounded staging")
        source_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=store))
        for name, expected in envelope["body"]["files"].items():
            target = pending / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            count = 0
            digest = hashlib.sha256()
            with _read_beneath(source_fd, name) as original, target.open("xb") as output:
                if os.fstat(original.fileno()).st_size != expected["bytes"]:
                    raise CapsuleError("Payload size changed")
                while block := original.read(1024 * 1024):
                    count += len(block)
                    if count > expected["bytes"]:
                        raise CapsuleError("Payload exceeded signed size")
                    digest.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                raise CapsuleError("Payload hash or length mismatch")
        with (pending / ".manifest.json").open("xb") as handle:
            handle.write(manifest)
            handle.flush()
            os.fsync(handle.fileno())
        for directory, _, _ in os.walk(pending, topdown=False):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        os.rename(pending, destination)
        pending = None
        os.fsync(store_fd)
        return destination
    finally:
        if pending is not None:
            shutil.rmtree(pending)
        if source_fd is not None:
            os.close(source_fd)
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(store_fd)


def verify_capsule_directory(
    directory: Path, manifest: bytes, trusted_keys: dict[str, bytes],
    *, require_trained=True, max_payload_bytes=MAX_PAYLOAD_BYTES,
) -> dict:
    """Reverify a complete published release after interruption or restart.

    This does not load or activate any code. It also does not repair or replace
    damaged files, which remain available for failure inspection.
    """
    envelope = verify_manifest(manifest, trusted_keys, require_trained=require_trained,
                               max_payload_bytes=max_payload_bytes)
    root_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        marker_fd = os.open(".manifest.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
        with os.fdopen(marker_fd, "rb") as marker:
            if not stat.S_ISREG(os.fstat(marker.fileno()).st_mode) or marker.read(MAX_MANIFEST_BYTES + 1) != manifest:
                raise CapsuleError("Stored release manifest differs")
        expected_files = set(envelope["body"]["files"]) | {".manifest.json"}
        expected_directories = {""}
        for name in envelope["body"]["files"]:
            parts = name.split("/")
            expected_directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
        for parent, directories, files, directory_fd in os.fwalk(".", dir_fd=root_fd, follow_symlinks=False):
            prefix = "" if parent == "." else parent.removeprefix("./")
            for name in directories + files:
                relative = f"{prefix}/{name}" if prefix else name
                entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if name in directories:
                    if relative not in expected_directories or not stat.S_ISDIR(entry.st_mode):
                        raise CapsuleError("Unlisted or unsafe stored directory")
                elif relative not in expected_files or not stat.S_ISREG(entry.st_mode):
                    raise CapsuleError("Unlisted or unsafe stored payload")
        for name, expected in envelope["body"]["files"].items():
            with _read_beneath(root_fd, name) as stream:
                if os.fstat(stream.fileno()).st_size != expected["bytes"]:
                    raise CapsuleError("Stored payload length differs")
                total, digest = 0, hashlib.sha256()
                while block := stream.read(1024 * 1024):
                    total += len(block)
                    if total > expected["bytes"]:
                        raise CapsuleError("Stored payload exceeded its signed size")
                    digest.update(block)
                if total != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                    raise CapsuleError("Stored payload integrity differs")
    finally:
        os.close(root_fd)
    return envelope
