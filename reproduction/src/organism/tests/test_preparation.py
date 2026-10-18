"""Archive and package preparation checks; fixture ELF headers are not programs."""

from dataclasses import replace
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from organism.capsule import canonical, sign_manifest
from organism.preparation import (
    ENTRYPOINT, PYTHON, PackagePreparer, PreparationError, native_inventory,
)
from organism.receiver import LocalReceiver
from organism.receiver_state import AdmissionError
from organism.safe_archive import ArchiveError, ArchiveLimits, case_sensitive_directory, extract_archive
from organism.supervision import Supervisor
from organism.systemd_supervision import SystemdBackend
from organism.tests import test_receiver_state as receiver_fixture
from organism.tests import test_supervision as supervision_fixture


def archive_bytes(entries, *, compressed=False):
    target = io.BytesIO()
    with tarfile.open(fileobj=target, mode="w", format=tarfile.PAX_FORMAT) as output:
        for item in entries:
            name, data = item[:2]
            kind, link, mode = item[2:] if len(item) > 2 else (tarfile.REGTYPE, "", 0o644)
            entry = tarfile.TarInfo(name)
            entry.type, entry.linkname, entry.mode = kind, link, mode
            entry.size = len(data) if kind == tarfile.REGTYPE else 0
            output.addfile(entry, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    data = target.getvalue()
    return gzip.compress(data, mtime=0) if compressed else data


def fixture_elf(platform):
    # Deliberately just a header and label. Never passed to exec or a linker.
    data = bytearray(64)
    data[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<H", data, 18, 183 if platform == "linux-arm64" else 62)
    return bytes(data) + b"EXPLICIT UNIT FIXTURE: NOT AN EXECUTABLE OR MODEL\n"


def runtime_archives(platform):
    binary = fixture_elf(platform)
    library = {"bytes": len(binary), "sha256": hashlib.sha256(binary).hexdigest()}
    return {
        "python": archive_bytes([
            ("python/bin/python3.12", binary, tarfile.REGTYPE, "", 0o755),
            ("python/lib/libpython.so", binary),
            ("python/bin/python3", b"", tarfile.SYMTYPE, "python3.12", 0o777),
        ], compressed=True),
        "inference": archive_bytes([
            ("llama-b9850", b"", tarfile.DIRTYPE, "", 0o755),
            ("llama-b9850/llama-server", binary, tarfile.REGTYPE, "", 0o755),
        ]),
        "dependencies": archive_bytes([
            ("site-packages/strands/__init__.py", b"# fixture only\n"),
            ("site-packages/cryptography/__init__.py", b"# fixture only\n"),
            ("site-packages/fixture.so", binary),
            ("supplemental/lib/libfixture.so", binary),
            ("supplemental/library-resolution.json", canonical({
                "scope": "Unit fixture, no native linker execution",
                "carried_libraries": {"libfixture.so": library},
            })),
            ("supplemental/notices/fixture.txt", b"Authored test fixture. Not third-party library content.\n"),
        ]),
    }


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "input.tar"
        self.destination = self.root / "expanded"

    def extract(self, entries, **kwargs):
        self.archive.write_bytes(archive_bytes(entries))
        return extract_archive(self.archive, self.destination, **kwargs)

    def test_streamed_files_and_internal_link_chain_with_forward_targets(self):
        result = self.extract([
            ("bin/start", b"", tarfile.SYMTYPE, "../lib/first", 0o777),
            ("lib/first", b"", tarfile.SYMTYPE, "real", 0o777),
            ("lib/real", b"fixture data", tarfile.REGTYPE, "", 0o775),
        ])
        self.assertEqual(result["files"], 1)
        self.assertEqual(result["links"], 2)
        self.assertEqual((self.destination / "bin/start").read_bytes(), b"fixture data")
        self.assertEqual((self.destination / "lib/real").stat().st_mode & 0o777, 0o755)

    def test_bounded_gzip_and_explicit_prefix(self):
        self.archive.write_bytes(archive_bytes([
            ("python", b"", tarfile.DIRTYPE, "", 0o755), ("python/bin/file", b"content"),
        ], compressed=True))
        result = extract_archive(self.archive, self.destination, strip_prefix="python")
        self.assertEqual((self.destination / "bin/file").read_bytes(), b"content")
        self.assertFalse((self.destination / "python").exists())
        self.assertEqual(result["regular_bytes"], 7)

    def test_traversal_absolute_noncanonical_and_control_paths_leave_no_partial(self):
        for name in ("../outside", "/outside", "a/../outside", "a//file", "a\\file", "a/\nfile"):
            with self.subTest(name=name), self.assertRaises(ArchiveError):
                self.extract([(name, b"data")])
            self.assertFalse(self.destination.exists())
            self.assertFalse((self.root / "outside").exists())

    def test_duplicate_paths_and_file_directory_overlap_are_rejected(self):
        for entries in ([(("same"), b"a"), ("same", b"b")],
                        [("parent", b"file"), ("parent/child", b"child")],
                        [("parent/child", b"child"), ("parent", b"file")]):
            with self.subTest(entries=entries), self.assertRaises(ArchiveError):
                self.extract(entries)
            self.assertFalse(self.destination.exists())

    def test_case_collisions_require_actual_filesystem_support(self):
        if case_sensitive_directory(self.root):
            result = self.extract([("File", b"a"), ("file", b"b")])
            self.assertTrue(result["case_sensitive_filesystem"])
            self.assertEqual((self.destination / "File").read_bytes(), b"a")
            self.assertEqual((self.destination / "file").read_bytes(), b"b")
        else:
            with self.assertRaisesRegex(ArchiveError, "case-insensitive"):
                self.extract([("File", b"a"), ("file", b"b")])
            self.assertFalse(self.destination.exists())

    def test_link_escape_cycle_missing_target_and_write_through_link_are_rejected(self):
        for entries in (
            [("link", b"", tarfile.SYMTYPE, "../outside", 0o777), ("file", b"a")],
            [("link", b"", tarfile.SYMTYPE, "/outside", 0o777), ("file", b"a")],
            [("a", b"", tarfile.SYMTYPE, "b", 0o777), ("b", b"", tarfile.SYMTYPE, "a", 0o777), ("file", b"a")],
            [("a", b"", tarfile.SYMTYPE, "missing", 0o777), ("file", b"a")],
            [("a", b"", tarfile.SYMTYPE, "target", 0o777), ("a/file", b"data"), ("target/file", b"data")],
        ):
            with self.subTest(entries=entries), self.assertRaises(ArchiveError):
                self.extract(entries)
            self.assertFalse(self.destination.exists())

    def test_special_entries_hardlinks_and_privileged_bits_are_rejected(self):
        for kind in (tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE):
            with self.subTest(kind=kind), self.assertRaises(ArchiveError):
                self.extract([("special", b"", kind, "target", 0o644), ("target", b"data")])
            self.assertFalse(self.destination.exists())
        with self.assertRaises(ArchiveError):
            self.extract([("privileged", b"data", tarfile.REGTYPE, "", 0o4755)])

    def test_byte_member_and_expanded_metadata_allowances_are_enforced(self):
        with self.assertRaisesRegex(ArchiveError, "file allowance"):
            self.extract([("file", b"12345")], limits=ArchiveLimits(bytes=4))
        with self.assertRaisesRegex(ArchiveError, "member allowance"):
            self.extract([("one", b"1"), ("two", b"2")], limits=ArchiveLimits(members=1))
        with self.assertRaisesRegex(ArchiveError, "stream exceeds"):
            self.extract([("file", b"1")], limits=ArchiveLimits(bytes=1, metadata_bytes=512))
        self.assertFalse(self.destination.exists())

    def test_gzip_trailer_corruption_and_truncated_file_do_not_publish(self):
        data = bytearray(archive_bytes([("file", b"data")], compressed=True))
        data[-5] ^= 0xff
        self.archive.write_bytes(data)
        with self.assertRaises((OSError, EOFError, tarfile.TarError)):
            extract_archive(self.archive, self.destination)
        self.assertFalse(self.destination.exists())
        self.archive.write_bytes(archive_bytes([("file", b"x" * 4096)])[:1024])
        with self.assertRaises((ArchiveError, tarfile.TarError)):
            extract_archive(self.archive, self.destination)
        self.assertFalse(self.destination.exists())

    def test_existing_destination_is_preserved(self):
        self.destination.mkdir()
        (self.destination / "owner-file").write_text("keep")
        with self.assertRaises(FileExistsError):
            self.extract([("file", b"data")])
        self.assertEqual((self.destination / "owner-file").read_text(), "keep")

    def test_unexpected_component_roots_and_empty_content_are_rejected(self):
        with self.assertRaisesRegex(ArchiveError, "top-level"):
            self.extract([("unexpected/file", b"data")], strip_prefix="python")
        with self.assertRaisesRegex(ArchiveError, "top-level"):
            self.extract([("unexpected/file", b"data")], allowed_roots=("site-packages",))
        with self.assertRaisesRegex(ArchiveError, "no regular"):
            self.extract([("directory", b"", tarfile.DIRTYPE, "", 0o755)])


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = receiver_fixture.ReceiverStateTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        for platform, names in f.capsule_body["platforms"].items():
            for role, data in runtime_archives(platform).items():
                (f.source / names[role]).write_bytes(data)
        for role, entries in {
            "agent": [("organism/__init__.py", b"# fixture only\n"),
                      ("organism/entrypoint.py", b"# Fixture entry point. Never executed.\n")],
            "knowledge": [("equipment.txt", b"Authored knowledge fixture\n")],
            "self_test": [("check.txt", b"Authored check fixture\n")],
            "notices": [("NOTICE.txt", b"Authored fixture notice\n")],
        }.items():
            (f.source / f.capsule_body[role]).write_bytes(archive_bytes(entries))
        self.sign()
        self.receiver = LocalReceiver(f.state, f.root / "store")
        observed = f.state.advertisement()
        self.receiver.receive(f.parent, f.operation, observed["revision"], f.manifest, f.source,
                              admission_ticket=observed["admission_ticket"])
        self.source = self.receiver.store / f.state.snapshot()["current_birth"]["release_id"]
        self.preparer = PackagePreparer(f.root / "prepared", f.policy)
        self.addCleanup(self.unseal_prepared)

    def unseal_prepared(self):
        if self.preparer.directory.exists():
            for parent, directories, _ in os.walk(self.preparer.directory, followlinks=False):
                Path(parent).chmod(0o700)
                for name in directories:
                    path = Path(parent) / name
                    if not path.is_symlink():
                        path.chmod(0o700)

    def sign(self):
        f = self.fixture
        for name in f.capsule_body["files"]:
            data = (f.source / name).read_bytes()
            f.capsule_body["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        f.manifest = sign_manifest(f.capsule_body, f.release_key)

    def prepare(self, platform="linux-arm64"):
        return self.preparer.prepare(self.source, self.fixture.manifest, platform)

    def test_prepared_checker_declaration_binds_installed_source_model_and_runtime(self):
        f = self.fixture
        worker = b"# Explicit prepared checker fixture, never executed.\n"
        (f.source / f.capsule_body["self_test"]).write_bytes(archive_bytes([("readiness.py", worker)]))
        self.sign()
        (f.source / ".manifest.json").write_bytes(f.manifest)
        prepared = self.preparer.prepare(f.source, f.manifest, "linux-arm64")
        work = f.root / "checker-work"
        work.mkdir()
        spec = prepared.process_spec(work)
        spec.validate()
        self.assertEqual(spec.readiness["checker_sha256"], hashlib.sha256(worker).hexdigest())
        self.assertEqual(spec.readiness["checker_path"], str(prepared.path / "runtime/self_test/readiness.py"))
        model = f.capsule_body["model"]["path"]
        self.assertEqual(spec.readiness["model_sha256"], f.capsule_body["files"][model]["sha256"])
        self.assertEqual(spec.readiness["model_bytes"], f.capsule_body["files"][model]["bytes"])
        self.assertEqual(spec.readiness["model_path"], str(prepared.capsule / model))
        self.assertEqual(spec.readiness["runtime_sha256"],
                         hashlib.sha256((prepared.path / "runtime/inference/llama-server").read_bytes()).hexdigest())

    def test_oversized_prepared_checker_cannot_be_declared_for_activation(self):
        f = self.fixture
        (f.source / f.capsule_body["self_test"]).write_bytes(archive_bytes([("readiness.py", b"x" * 65537)]))
        self.sign()
        (f.source / ".manifest.json").write_bytes(f.manifest)
        prepared = self.preparer.prepare(f.source, f.manifest, "linux-arm64")
        work = f.root / "checker-work"
        work.mkdir()
        with self.assertRaisesRegex(PreparationError, "bounded regular"):
            prepared.process_spec(work)

    def test_prepared_child_carries_exact_complete_capsule_for_both_platforms(self):
        prepared = self.prepare()
        index = prepared.verify()
        self.assertFalse(index["native_execution_verified"])
        self.assertFalse(index["native"]["dynamic_loading_verified"])
        self.assertEqual(index["platform"], "linux-arm64")
        for name, entry in self.fixture.capsule_body["files"].items():
            path = prepared.capsule / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
            self.assertEqual(path.stat().st_nlink, 1)
        self.assertEqual((prepared.capsule / ".manifest.json").read_bytes(), self.fixture.manifest)
        self.assertEqual(prepared.path.stat().st_mode & 0o777, 0o555)
        self.assertEqual((prepared.path / PYTHON).stat().st_mode & 0o777, 0o555)
        self.assertEqual((prepared.path / ENTRYPOINT).stat().st_mode & 0o777, 0o444)

    def test_native_platform_choice_does_not_require_extracting_foreign_executables(self):
        arm = self.prepare("linux-arm64")
        x86 = self.prepare("linux-x86_64")
        self.assertNotEqual(arm.path, x86.path)
        self.assertIn("linux-x86_64", (arm.capsule / ".manifest.json").read_text())
        self.assertEqual(struct.unpack_from("<H", (arm.path / PYTHON).read_bytes(), 18)[0], 183)
        self.assertEqual(struct.unpack_from("<H", (x86.path / PYTHON).read_bytes(), 18)[0], 62)
        self.assertEqual(len(list(self.preparer.directory.glob("*-linux-*"))), 2)

    def test_repeated_preparation_reverifies_without_replacing_existing_installation(self):
        first = self.prepare()
        before = (first.path / ".installation.json").stat().st_ino
        second = self.prepare()
        self.assertEqual(first.path, second.path)
        self.assertEqual((second.path / ".installation.json").stat().st_ino, before)

    def test_post_publication_changes_are_rejected_and_preserved_for_inspection(self):
        prepared = self.prepare()
        entry = prepared.path / ENTRYPOINT
        entry.chmod(0o644)
        entry.write_text("# modified fixture\n")
        entry.chmod(0o444)
        with self.assertRaisesRegex(PreparationError, "contents changed"):
            self.prepare()
        self.assertEqual(entry.read_text(), "# modified fixture\n")

    def test_missing_foreign_capsule_component_prevents_reuse(self):
        prepared = self.prepare()
        name = self.fixture.capsule_body["platforms"]["linux-x86_64"]["python"]
        parent = (prepared.capsule / name).parent
        parent.chmod(0o755)
        (prepared.capsule / name).unlink()
        parent.chmod(0o555)
        with self.assertRaises((ValueError, OSError)):
            prepared.verify()

    def test_owner_store_and_prepared_capacity_are_required(self):
        self.preparer.directory.chmod(0o777)
        with self.assertRaisesRegex(PreparationError, "controlled"):
            PackagePreparer(self.preparer.directory, self.fixture.policy)
        self.preparer.directory.chmod(0o755)
        small = PackagePreparer(self.preparer.directory, self.fixture.policy, max_store_bytes=1024)
        with self.assertRaisesRegex(PreparationError, "space"):
            small.prepare(self.source, self.fixture.manifest, "linux-arm64")
        self.assertEqual([p.name for p in self.preparer.directory.iterdir()], [".prepare.lock"])

    def test_incomplete_prior_preparation_is_not_adopted_or_silently_deleted(self):
        partial = self.preparer.directory / ".preparing-old"
        partial.mkdir()
        (partial / "owner-evidence").write_text("keep")
        with self.assertRaisesRegex(PreparationError, "inspection"):
            self.prepare()
        self.assertEqual((partial / "owner-evidence").read_text(), "keep")

    def test_preparation_lock_prevents_competing_publications(self):
        with self.preparer._locked():
            with self.assertRaisesRegex(PreparationError, "in progress"):
                self.prepare()
        self.assertFalse(any(path.name.startswith(".preparing-") for path in self.preparer.directory.iterdir()))

    def test_failed_native_validation_removes_only_its_new_preparation(self):
        with patch("organism.preparation.native_inventory", side_effect=PreparationError("fixture failure")):
            with self.assertRaisesRegex(PreparationError, "fixture failure"):
                self.prepare()
        self.assertTrue(self.source.is_dir())
        self.assertEqual([path.name for path in self.preparer.directory.iterdir()], [".prepare.lock"])

    def test_raw_dependencies_without_collected_supplemental_libraries_are_rejected(self):
        prepared = self.prepare()
        receipt = prepared.path / "runtime/dependencies/supplemental/library-resolution.json"
        receipt.parent.chmod(0o755)
        receipt.unlink()
        receipt.parent.chmod(0o555)
        with self.assertRaisesRegex(PreparationError, "receipt"):
            native_inventory(prepared.path, "linux-arm64")

    def test_wrong_architecture_even_in_supplemental_library_is_rejected(self):
        prepared = self.prepare()
        target = prepared.path / "runtime/dependencies/supplemental/lib/libfixture.so"
        target.chmod(0o644)
        target.write_bytes(fixture_elf("linux-x86_64"))
        target.chmod(0o444)
        with self.assertRaisesRegex(PreparationError, "architecture"):
            native_inventory(prepared.path, "linux-arm64")

    def test_fixed_process_command_and_library_search_paths_come_from_verified_tree(self):
        prepared = self.prepare()
        work = self.fixture.root / "work"
        work.mkdir()
        spec = prepared.process_spec(work)
        spec.validate()
        self.assertEqual(spec.command[0], str(prepared.path / PYTHON))
        self.assertEqual(spec.command[1], "-I")
        self.assertEqual(spec.command[2], str(prepared.path / ENTRYPOINT))
        self.assertEqual(spec.readonly_paths, (prepared.path,))
        self.assertEqual(len(spec.library_paths), 3)
        with self.assertRaisesRegex(PreparationError, "outside"):
            prepared.process_spec(prepared.path / "mutable")
        alias = self.fixture.root / "aliased-installation"
        alias.symlink_to(prepared.path, target_is_directory=True)
        with self.assertRaisesRegex(PreparationError, "outside"):
            prepared.process_spec(alias / "runtime/agent")
        with self.assertRaisesRegex(ValueError, "read-only"):
            replace(spec, library_paths=(Path("/tmp/untrusted"),)).validate()

    def test_verified_preparer_connects_to_existing_supervisor_without_executing_fixture_programs(self):
        f = self.fixture
        host = supervision_fixture.FixtureLinuxHost(f.root, f.state)
        interpreter, guard = f.root / "owner-python", f.root / "owner-guard.py"
        interpreter.write_text("Owner interpreter fixture, never executed")
        interpreter.chmod(0o700)
        guard.write_text("Owner guard fixture, never executed")
        backend = SystemdBackend(f.state, guard_python=interpreter, guard_script=guard, host=host)
        supervisor = Supervisor(self.receiver, backend)
        work = f.root / "work"
        work.mkdir()
        with self.assertRaisesRegex(AdmissionError, "manifest differs"):
            supervisor.start_prepared(f.operation, f.manifest + b" ", self.preparer, work)
        self.assertFalse(any(self.preparer.directory.iterdir()))
        result = supervisor.start_prepared(f.operation, f.manifest, self.preparer, work)
        self.assertEqual(result["state"], "running")
        self.assertFalse(result["native_observation"])
        self.assertFalse(result["useful_readiness_verified"])
        birth = f.state.snapshot()["current_birth"]
        unit = backend.records.get(birth, "intent")["record"]["observation"]["unit_text"]
        self.assertIn("LD_LIBRARY_PATH=", unit)
        self.assertIn("/runtime/python/bin/python3.12", unit)
        self.assertIn("/runtime/agent/organism/entrypoint.py", unit)
        self.assertEqual(supervisor.start_prepared(f.operation, f.manifest, self.preparer, work)["state"], "running")
        self.assertEqual(sum(call[1] == "start" for call in host.calls), 1)

    def test_prepared_index_identity_and_permissions_cannot_be_changed(self):
        prepared = self.prepare()
        path = prepared.path / ".installation.json"
        path.chmod(0o644)
        with self.assertRaisesRegex(PreparationError, "index"):
            prepared.verify()
        body = json.loads(path.read_bytes())
        body["platform"] = "linux-x86_64"
        path.write_bytes(canonical(body))
        path.chmod(0o444)
        with self.assertRaisesRegex(PreparationError, "identity"):
            prepared.verify()


if __name__ == "__main__":
    unittest.main()
