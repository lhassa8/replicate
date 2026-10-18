"""Prepare a complete signed reproduction capsule and a native read-only tree.

Preparation runs as the trusted receiving owner. It never executes archive
contents or downloads dependencies. Static ELF checks do not prove executable
compatibility or useful readiness.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import tempfile

from .capsule import PLATFORMS, canonical, stage_capsule, verify_capsule_directory, verify_manifest
from .safe_archive import ArchiveLimits, extract_archive, remove_new_tree, safe_relative
from .systemd_supervision import ProcessSpec


SCHEMA = "organism-prepared-installation-v1"
MAX_INSTALLATIONS = 2
MAX_STORE_BYTES = 24 * 1024**3
MAX_INDEX_BYTES = 16 * 1024**2
MAX_TREE_ENTRIES = 60000
ROLE_LIMITS = {"python": 512, "inference": 256, "dependencies": 512,
               "agent": 32, "knowledge": 64, "self_test": 32, "notices": 32}
PYTHON = "runtime/python/bin/python3.12"
INFERENCE = "runtime/inference/llama-server"
ENTRYPOINT = "runtime/agent/organism/entrypoint.py"
SELF_TEST = "runtime/self_test/readiness.py"
LIBRARIES = ("runtime/dependencies/supplemental/lib", "runtime/python/lib", "runtime/inference")


class PreparationError(ValueError):
    pass


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def assert_directory(path, *, private=False):
    entry = path.lstat()
    if (not stat.S_ISDIR(entry.st_mode) or entry.st_uid != os.geteuid()
            or stat.S_IMODE(entry.st_mode) & (0o077 if private else 0o022)):
        raise PreparationError("Preparation directory is not controlled by its service owner")


def seal_tree(root):
    for parent, directories, files in os.walk(root, topdown=False, followlinks=False):
        for name in files:
            path = Path(parent) / name
            if not path.is_symlink():
                mode = path.stat().st_mode
                path.chmod(0o555 if mode & 0o100 else 0o444)
        for name in directories:
            path = Path(parent) / name
            if not path.is_symlink():
                path.chmod(0o555)
        Path(parent).chmod(0o555)


def tree_inventory(root):
    """Full no-follow inventory, including the carried opposite-platform files."""
    root = Path(root)
    assert_directory(root)
    result = {}
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            relative = path.relative_to(root).as_posix()
            if relative == ".installation.json":
                continue
            safe_relative(relative)
            if len(result) >= MAX_TREE_ENTRIES:
                raise PreparationError("Prepared installation has too many entries")
            entry = path.lstat()
            mode = stat.S_IMODE(entry.st_mode)
            if entry.st_uid != os.geteuid():
                raise PreparationError("Prepared entry belongs to another owner")
            if stat.S_ISLNK(entry.st_mode):
                target = os.readlink(path)
                if not path.resolve(strict=True).is_relative_to(root.resolve()):
                    raise PreparationError("Prepared link escapes its installation")
                result[relative] = {"type": "symlink", "target": target}
            elif stat.S_ISDIR(entry.st_mode):
                if mode != 0o555:
                    raise PreparationError("Prepared directory is not read-only")
                result[relative] = {"type": "directory", "mode": mode}
            elif stat.S_ISREG(entry.st_mode):
                if entry.st_nlink != 1 or mode not in (0o444, 0o555):
                    raise PreparationError("Prepared file is not a private immutable copy")
                result[relative] = {"type": "file", "mode": mode, "bytes": entry.st_size, "sha256": digest(path)}
            else:
                raise PreparationError("Prepared installation contains a special file")
    return result


def native_inventory(root, platform):
    expected = {"linux-x86_64": 62, "linux-arm64": 183}[platform]
    objects = {}
    for role in ("python", "inference", "dependencies"):
        for path in sorted((root / "runtime" / role).rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            with path.open("rb") as source:
                header = source.read(64)
            if header[:4] != b"\x7fELF":
                continue
            if (len(header) < 64 or header[4:7] != b"\x02\x01\x01"
                    or struct.unpack_from("<H", header, 18)[0] != expected):
                raise PreparationError("Native ELF architecture differs from the selected platform")
            objects[path.relative_to(root).as_posix()] = digest(path)
    if PYTHON not in objects or INFERENCE not in objects:
        raise PreparationError("Prepared Python and inference executables must be native ELF files")
    for relative in (PYTHON, INFERENCE, ENTRYPOINT,
                     "runtime/dependencies/site-packages/strands/__init__.py",
                     "runtime/dependencies/site-packages/cryptography/__init__.py"):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise PreparationError("Required prepared application path is missing")
    for relative in (PYTHON, INFERENCE):
        if not (root / relative).stat().st_mode & stat.S_IXUSR:
            raise PreparationError("Prepared native entry is not executable")
    # A final package must carry the collector's receipt and copied non-base
    # libraries. The raw dependency candidates alone cannot pass preparation.
    supplemental = root / "runtime/dependencies/supplemental"
    receipt_path = supplemental / "library-resolution.json"
    if receipt_path.is_symlink() or not receipt_path.is_file() or receipt_path.stat().st_size > 1024**2:
        raise PreparationError("Native supplemental-library receipt is missing or invalid")
    receipt = json.loads(receipt_path.read_bytes())
    libraries = receipt.get("carried_libraries")
    if not isinstance(libraries, dict) or not 1 <= len(libraries) <= 128:
        raise PreparationError("Native supplemental-library inventory is missing")
    for name, expected_file in libraries.items():
        safe_relative(name)
        if "/" in name:
            raise PreparationError("Supplemental library must have one basename")
        path = supplemental / "lib" / name
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != expected_file["bytes"]
                or digest(path) != expected_file["sha256"]):
            raise PreparationError("Supplemental library differs from its collector receipt")
        if path.relative_to(root).as_posix() not in objects:
            raise PreparationError("Supplemental library must be a matching native ELF object")
    if not (supplemental / "notices").is_dir() or not any((supplemental / "notices").iterdir()):
        raise PreparationError("Supplemental dependency notices are missing")
    return {"elf_objects": objects, "supplemental_libraries": len(libraries),
            "native_execution_verified": False, "dynamic_loading_verified": False}


def verify_prepared(path, manifest, keys, platform, *, require_trained=True, max_payload_bytes=8 * 1024**3):
    path = Path(path)
    assert_directory(path)
    index_path = path / ".installation.json"
    entry = index_path.lstat()
    if (not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1 or entry.st_uid != os.geteuid()
            or stat.S_IMODE(entry.st_mode) != 0o444 or entry.st_size > MAX_INDEX_BYTES):
        raise PreparationError("Prepared installation index is invalid")
    data = index_path.read_bytes()
    index = json.loads(data)
    envelope = verify_capsule_directory(path / "capsule", manifest, keys, require_trained=require_trained,
                                        max_payload_bytes=max_payload_bytes)
    if (index.get("schema"), index.get("release_id"), index.get("platform"), index.get("manifest_sha256")) != (
            SCHEMA, envelope["release_id"], platform, hashlib.sha256(manifest).hexdigest()):
        raise PreparationError("Prepared installation identity differs")
    if tree_inventory(path) != index["inventory"]:
        raise PreparationError("Prepared installation contents changed")
    if native_inventory(path, platform) != index["native"]:
        raise PreparationError("Prepared native inventory changed")
    return index


@dataclass(frozen=True)
class PreparedInstallation:
    path: Path
    manifest: bytes
    keys: dict
    platform: str
    require_trained: bool
    max_payload_bytes: int

    @property
    def capsule(self):
        return self.path / "capsule"

    def verify(self):
        return verify_prepared(self.path, self.manifest, self.keys, self.platform,
                               require_trained=self.require_trained, max_payload_bytes=self.max_payload_bytes)

    def process_spec(self, work_directory, *, lab_cidrs=("127.0.0.0/24",), user="organism", group="organism",
                     node_context=None):
        index = self.verify()
        work_directory = Path(work_directory).absolute()
        if work_directory.resolve().is_relative_to(self.path.resolve()):
            raise PreparationError("Mutable work must be outside the read-only installation")
        if not work_directory.is_dir() or work_directory.is_symlink():
            raise PreparationError("Mutable work must be an existing real directory")
        command = (str(self.path / PYTHON), "-I", str(self.path / ENTRYPOINT),
                   "--installation", str(self.path), "--work-directory", str(work_directory))
        readonly = (self.path,)
        if node_context is not None:
            node_context.verify()
            if node_context.release_id != json.loads(self.manifest)["release_id"]:
                raise PreparationError("Node context belongs to another prepared release")
            command += ("--node-context", str(node_context.path), "--node-context-sha256", node_context.sha256)
            readonly += (node_context.path,)
        readiness = None
        # Legacy explicit package fixtures have no executable checker. Native
        # receiver activation requires this declaration before claiming a start.
        if SELF_TEST in index["inventory"]:
            checker = index["inventory"][SELF_TEST]
            model_name = json.loads(self.manifest)["body"]["model"]["path"]
            model = index["inventory"]["capsule/" + model_name]
            runtime = index["inventory"][INFERENCE]
            if checker["type"] != "file" or checker["bytes"] > 65536:
                raise PreparationError("Carried readiness checker must be a bounded regular file")
            readiness = {"checker_path": str(self.path / SELF_TEST), "checker_sha256": checker["sha256"],
                         "model_path": str(self.capsule / model_name), "model_sha256": model["sha256"],
                         "model_bytes": model["bytes"], "runtime_path": str(self.path / INFERENCE),
                         "runtime_sha256": runtime["sha256"]}
        return ProcessSpec(
            command, work_directory, readonly, user=user, group=group, lab_cidrs=lab_cidrs,
            library_paths=tuple(self.path / relative for relative in LIBRARIES),
            readiness=readiness,
        )


class PackagePreparer:
    def __init__(self, directory, policy, *, max_store_bytes=MAX_STORE_BYTES):
        self.directory = Path(directory).absolute()
        try:
            self.directory.mkdir(mode=0o755, parents=False)
        except FileExistsError:
            pass
        else:
            self.directory.chmod(0o755)
        assert_directory(self.directory)
        if not self.directory.stat().st_mode & 0o001:
            raise PreparationError("Prepared-store root must be traversable by its application account")
        self.policy = policy
        if type(max_store_bytes) is not int or not 1 <= max_store_bytes <= MAX_STORE_BYTES:
            raise PreparationError("Invalid prepared-store allowance")
        self.max_store_bytes = max_store_bytes

    @contextmanager
    def _locked(self):
        fd = os.open(self.directory / ".prepare.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            entry = os.fstat(fd)
            if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.geteuid() or entry.st_nlink != 1
                    or stat.S_IMODE(entry.st_mode) & 0o077):
                raise PreparationError("Invalid preparation lock")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PreparationError("Another native package preparation is in progress") from exc
            yield
        finally:
            os.close(fd)

    def _usage(self):
        total, installations = 0, 0
        for path in self.directory.iterdir():
            if path.name == ".prepare.lock":
                continue
            if (not re.fullmatch(r"[0-9a-f]{64}-linux-(?:arm64|x86_64)", path.name)
                    or path.is_symlink() or not path.is_dir()):
                raise PreparationError("Unknown or interrupted preparation requires owner inspection")
            installations += 1
            for parent, _, files in os.walk(path, followlinks=False):
                for name in files:
                    info = (Path(parent) / name).lstat()
                    if stat.S_ISREG(info.st_mode):
                        total += info.st_size
        return total, installations

    def prepare(self, source, manifest, platform):
        if platform not in PLATFORMS:
            raise PreparationError("Unsupported preparation platform")
        options = {"require_trained": self.policy.body["require_trained"],
                   "max_payload_bytes": self.policy.body["max_payload_bytes"]}
        envelope = verify_manifest(manifest, self.policy.release_keys, **options)
        if envelope["body"]["experiment_id"] != self.policy.body["experiment_id"]:
            raise PreparationError("Package belongs to another experiment")
        destination = self.directory / (envelope["release_id"] + "-" + platform)
        with self._locked():
            usage, count = self._usage()
            if usage > self.max_store_bytes or count > MAX_INSTALLATIONS:
                raise PreparationError("Prepared-store allowance exhausted")
            result = PreparedInstallation(destination, manifest, self.policy.release_keys, platform, **options)
            if destination.exists():
                result.verify()
                return result
            if count >= MAX_INSTALLATIONS:
                raise PreparationError("Prepared installation count exhausted")
            payload = sum(item["bytes"] for item in envelope["body"]["files"].values())
            reserve = payload + sum(ROLE_LIMITS.values()) * 1024**2 + MAX_INDEX_BYTES
            if usage + reserve > self.max_store_bytes or shutil.disk_usage(self.directory).free < reserve + 64 * 1024**2:
                raise PreparationError("Insufficient reserved space for native preparation")
            verify_capsule_directory(source, manifest, self.policy.release_keys, **options)
            pending = Path(tempfile.mkdtemp(prefix=".preparing-", dir=self.directory))
            try:
                staging = pending / "reproduction-stage"
                copied = stage_capsule(Path(source), staging, manifest, self.policy.release_keys, **options)
                os.rename(copied, pending / "capsule")
                (staging / ".install.lock").unlink()
                staging.rmdir()
                runtime = pending / "runtime"
                runtime.mkdir(mode=0o700)
                native = envelope["body"]["platforms"][platform]
                extracted = {}
                for role in ROLE_LIMITS:
                    name = native[role] if role in native else envelope["body"][role]
                    extracted[role] = extract_archive(
                        pending / "capsule" / name, runtime / role,
                        limits=ArchiveLimits(bytes=ROLE_LIMITS[role] * 1024**2),
                        strip_prefix={"python": "python", "inference": "llama-b9850"}.get(role),
                        allowed_roots=("site-packages", "supplemental") if role == "dependencies"
                        else (("organism",) if role == "agent" else None),
                    )
                native_report = native_inventory(pending, platform)
                seal_tree(pending)
                inventory = tree_inventory(pending)
                index = {"schema": SCHEMA, "release_id": envelope["release_id"], "platform": platform,
                         "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                         "inventory": inventory, "extraction": extracted, "native": native_report,
                         "qualification_passed": False, "native_execution_verified": False}
                data = canonical(index)
                if len(data) > MAX_INDEX_BYTES:
                    raise PreparationError("Prepared installation index exceeds its allowance")
                # Root alone may add the index; restore the sealed mode before
                # verification/publication. Never mutate an existing release.
                pending.chmod(0o700)
                with (pending / ".installation.json").open("xb") as target:
                    target.write(data)
                    target.flush()
                    os.fchmod(target.fileno(), 0o444)
                    os.fsync(target.fileno())
                pending.chmod(0o555)
                verify_prepared(pending, manifest, self.policy.release_keys, platform, **options)
                for parent, _, _ in os.walk(pending, topdown=False, followlinks=False):
                    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                os.rename(pending, destination)
                pending = None
                fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            finally:
                if pending is not None:
                    remove_new_tree(pending)
            return result
