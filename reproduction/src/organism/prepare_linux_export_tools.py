"""Prepare recorded Linux export tools without executing them or provisioning a host."""

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import zipfile

from .gguf_export import COMMIT, SOURCE_ARCHIVE_SHA256, bind_toolchain, identity
from .supplemental_package import elf_info
from .training_inputs import require, verified_path


WHEEL_NAME = ("sentencepiece-0.2.1-cp312-cp312-"
              "manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")
WHEEL_SHA256 = "0cdfecef430d985f1c2bcbfff3defd1d95dae876fbd0173376012d2d7d24044b"
BINARY_SHA256 = "ec5cf21cb4f8c5a48b0ab970a6efd4ea7db44b1638510aafd3c1c2955d9d9c6a"
HOST_LIBRARIES = {"libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2",
                  "librt.so.1", "ld-linux-x86-64.so.2"}


def relative_name(name):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and ".." not in path.parts
            and "\\" not in name and str(path) == name.rstrip("/"),
            "Noncanonical archive path")
    return path


def wheel_contents(path, metadata):
    require(path.name == WHEEL_NAME and identity(path) == {"bytes": 1387881, "sha256": WHEEL_SHA256},
            "Linux tokenizer wheel identity changed")
    require(metadata["info"]["name"] == "sentencepiece" and metadata["info"]["version"] == "0.2.1",
            "Tokenizer publisher metadata changed")
    entries = [entry for entry in metadata["urls"] if entry["filename"] == WHEEL_NAME]
    require(len(entries) == 1 and entries[0]["digests"]["sha256"] == WHEEL_SHA256
            and entries[0]["size"] == path.stat().st_size and entries[0]["yanked"] is False,
            "Wheel differs from the retained publisher declaration")
    files = {}
    with zipfile.ZipFile(path) as archive:
        require(len(archive.infolist()) <= 64
                and sum(item.file_size for item in archive.infolist()) <= 8 * 1024**2,
                "Tokenizer wheel exceeds its extraction bound")
        names = set()
        for item in archive.infolist():
            name = str(relative_name(item.filename))
            require(name not in names, "Duplicate tokenizer archive path")
            names.add(name)
            mode = item.external_attr >> 16
            require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), "Tokenizer archive contains a link or special file")
            if not item.is_dir():
                files[name] = archive.read(item)
    record = "sentencepiece-0.2.1.dist-info/RECORD"
    rows = list(csv.reader(io.StringIO(files[record].decode())))
    require(all(len(row) == 3 for row in rows) and len({row[0] for row in rows}) == len(rows)
            and {row[0] for row in rows} == set(files), "Tokenizer RECORD inventory differs")
    for name, digest, size in rows:
        if name == record:
            require(digest == size == "", "Unexpected tokenizer RECORD self-hash")
        else:
            observed = base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest()).decode().rstrip("=")
            require(digest == "sha256=" + observed and size == str(len(files[name])),
                    "Tokenizer RECORD hash or size differs")
    return files


def unpack_selected(archive_path, output, select, *, maximum_files, maximum_bytes):
    """Write selected regular files, then in-directory aliases with regular targets."""
    selected, total = {}, 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for item in archive:
            destination = select(item.name)
            if destination is None:
                continue
            relative_name(destination)
            require(destination not in selected and (item.isfile() or item.issym()),
                    "Duplicate or unsupported selected archive entry")
            total += item.size
            require(len(selected) < maximum_files and total <= maximum_bytes
                    and 0 <= item.size <= 16 * 1024**2, "Selected archive exceeds extraction bounds")
            if item.issym():
                target = relative_name(item.linkname)
                require(len(target.parts) == 1, "Library alias must stay in its directory")
                selected[destination] = ("alias", item.linkname, 0)
            else:
                selected[destination] = ("file", archive.extractfile(item).read(), item.mode & 0o111)
    for name, (kind, value, executable) in selected.items():
        if kind == "file":
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(value)
            path.chmod(0o755 if executable else 0o644)
    for name, (kind, value, _) in selected.items():
        if kind == "alias":
            target = str(PurePosixPath(name).parent / value)
            visited = {name}
            while target in selected and selected[target][0] == "alias":
                require(target not in visited, "Cyclic library alias")
                visited.add(target)
                target = str(PurePosixPath(target).parent / selected[target][1])
            require(target in selected and selected[target][0] == "file", "Unbound library alias target")
            (output / name).symlink_to(value)
    return selected


def library_requirements(output, files, elf):
    providers = {}
    for name in files:
        if ".so" not in Path(name).name:
            continue
        target = str((output / name).resolve().relative_to(output.resolve()))
        if target in elf:
            basename = Path(name).name
            require(basename not in providers or providers[basename] == target,
                    "Conflicting native library providers")
            providers[basename] = target
    host = {}
    checked = 0
    for item in elf.values():
        for name in item["needed"]:
            required = set(item["symbol_versions"].get(name, []))
            if name in providers:
                require(required <= set(elf[providers[name]]["defined_versions"]),
                        "A bundled library lacks required symbol versions")
                checked += 1
            else:
                require(name in HOST_LIBRARIES, "An unbundled non-glibc library is required")
                host.setdefault(name, set()).update(required)
    return {"bundled_dependency_edges_checked": checked,
            "host_library_symbol_versions_required": {name: sorted(values) for name, values in sorted(host.items())}}


def prepare(workspace, wheel, metadata_path, output, *, include_imatrix=False):
    require(type(include_imatrix) is bool, "Invalid calibration-tool selection")
    workspace, wheel, metadata_path, output = (
        Path(value).resolve() for value in (workspace, wheel, metadata_path, output))
    require(not output.exists() and output.parent.is_dir(), "Export tool output must be new")
    capsule_path = workspace / "organism/development-capsule-inputs-20261014.json"
    capsule = json.loads(capsule_path.read_bytes())
    archives = {}
    for role, key in (("binaries", "linux-x86_64/inference.tar.gz"),
                      ("supplemental", "linux-x86_64/dependencies.tar.gz")):
        entry = capsule["files"][key]
        archives[role] = verified_path(workspace, {"path": entry["source"],
                                                  "bytes": entry["bytes"], "sha256": entry["sha256"]})
    require(identity(archives["binaries"])["sha256"] == BINARY_SHA256, "Pinned Linux binaries changed")
    archives["source"] = verified_path(workspace, {"path": "tools/export/source.tar.gz",
                                                  "bytes": 35097371, "sha256": SOURCE_ARCHIVE_SHA256})
    release_path = workspace / "organism/evidence/runtime-sources-20261014-01/llama-release.json"
    release = json.loads(release_path.read_bytes())
    assets = [entry for entry in release["assets"] if entry["name"] == archives["binaries"].name]
    require(release["tag_name"] == "b9850" and release["target_commitish"] == COMMIT
            and len(assets) == 1 and assets[0]["digest"] == "sha256:" + BINARY_SHA256
            and assets[0]["size"] == archives["binaries"].stat().st_size,
            "Linux tools differ from the retained publisher declaration")
    dependency = wheel_contents(wheel, json.loads(metadata_path.read_bytes()))
    output.mkdir()
    report = {"schema": "organism-linux-export-preparation-v1",
              "started_at": datetime.now(timezone.utc).isoformat(), "prepared": False,
              "failure": None, "native_execution_verified": False, "qualification_passed": False,
              "source_commit_refetched": False, "cloud_resources_created": False,
              "training_environment_changed": False, "python_environment_included": False}
    try:
        (output / "archives").mkdir()
        for role, path in archives.items():
            shutil.copyfile(path, output / "archives" / (role + ".tar.gz"))
        shutil.copyfile(wheel, output / "archives" / WHEEL_NAME)
        (output / "provenance").mkdir()
        for name, path in (("capsule-inputs.json", capsule_path), ("llama-release.json", release_path),
                           ("sentencepiece-pypi.json", metadata_path)):
            shutil.copyfile(path, output / "provenance" / name)

        prefix = "llama.cpp-" + COMMIT + "/"

        def source_member(name):
            if not name.startswith(prefix):
                return None
            relative = name[len(prefix):]
            if (relative in ("LICENSE", "convert_hf_to_gguf.py")
                    or (relative.endswith(".py")
                        and relative.startswith(("conversion/", "gguf-py/gguf/")))):
                return "source/" + name
            return None

        unpack_selected(archives["source"], output, source_member, maximum_files=256,
                        maximum_bytes=8 * 1024**2)
        unpack_selected(
            archives["binaries"], output,
            lambda name: "bin/" + name if name.startswith("llama-b9850/")
            and (".so" in PurePosixPath(name).name
                 or PurePosixPath(name).name in
                 (("llama-quantize", "llama-server", "llama-imatrix") if include_imatrix
                  else ("llama-quantize", "llama-server"))) else None,
            maximum_files=64, maximum_bytes=64 * 1024**2)
        unpack_selected(
            archives["supplemental"], output,
            lambda name: name if name.startswith(("supplemental/lib/", "supplemental/notices/"))
            and not name.endswith("/") else None,
            maximum_files=128, maximum_bytes=32 * 1024**2)
        for name, value in dependency.items():
            path = output / "dependencies" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        files, aliases, elf = {}, {}, {}
        for path in sorted(output.rglob("*")):
            if path.is_file():
                name = str(path.relative_to(output))
                files[name] = identity(path)
                if path.is_symlink():
                    aliases[name] = os.readlink(path)
                else:
                    with path.open("rb") as stream:
                        magic = stream.read(4)
                    if magic == b"\x7fELF":
                        elf[name] = elf_info(path.read_bytes(), "EM_X86_64")
        requirements = library_requirements(output, files, elf)
        require(len(elf) >= 20 and len(aliases) >= 6, "Linux native inventory is incomplete")
        prior = {"declared_commit": COMMIT, "source_archive": identity(archives["source"]),
                 "files": files, "source_members_equal_retained_archive": True,
                 "upstream_commit_independently_refetched": False}
        source_root = "source/llama.cpp-" + COMMIT
        manifest = bind_toolchain(
            prior, source_root=source_root, source_archive="archives/source.tar.gz",
            converter=source_root + "/convert_hf_to_gguf.py",
            quantizer="bin/llama-b9850/llama-quantize", server="bin/llama-b9850/llama-server",
            dependency_root="dependencies", library_paths=("bin/llama-b9850", "supplemental/lib"),
            symlinks=aliases, system="Linux", machine="x86_64")
        if include_imatrix:
            manifest["imatrix"] = "bin/llama-b9850/llama-imatrix"
        (output / "tool-inputs.json").write_text(json.dumps(prior, indent=2) + "\n")
        (output / "toolchain.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (output / "elf-inventory.json").write_text(json.dumps(elf, indent=2) + "\n")
        report.update(prepared=True, files=len(files), library_aliases=len(aliases), elf_objects=len(elf),
                      retained_file_bytes=sum(item["bytes"] for name, item in files.items() if name not in aliases),
                      **requirements, toolchain=identity(output / "toolchain.json"),
                      python_runtime_required="CPython 3.12 with the existing training lock and NumPy 2.5.4",
                      native_abi_and_symbol_resolution_verified=False,
                      preparation_source=identity(Path(__file__)))
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "wheel", "metadata", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--include-imatrix", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.wheel, args.metadata, args.output,
                             include_imatrix=args.include_imatrix), indent=2))
