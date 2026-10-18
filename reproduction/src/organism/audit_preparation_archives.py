"""Check the preparer's extraction against retained real native archives.

Runs on Mac without executing any extracted binary. A separate tar reader
compares every regular file and link to the expanded filesystem tree.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import tarfile
import tempfile
import time

from .capsule import canonical
from .preparation import digest
from .safe_archive import ArchiveLimits, extract_archive


def compare_archive(source, expanded, prefix, platform):
    expected, native = {}, {}
    with tarfile.open(source, mode="r:*") as archive:
        for member in archive:
            name = member.name.rstrip("/")
            if prefix:
                if name == prefix:
                    continue
                if not name.startswith(prefix + "/"):
                    raise ValueError("Independent reader found another archive prefix")
                name = name[len(prefix) + 1:]
            if member.isdir():
                continue
            target = expanded / name
            if member.issym():
                if not target.is_symlink() or os.readlink(target) != member.linkname:
                    raise ValueError("Extracted link differs from the retained archive")
                expected[name] = {"type": "link", "target": member.linkname}
            elif member.isfile():
                value = hashlib.sha256()
                with archive.extractfile(member) as content:
                    for block in iter(lambda: content.read(1024 * 1024), b""):
                        value.update(block)
                if target.is_symlink() or target.stat().st_size != member.size or digest(target) != value.hexdigest():
                    raise ValueError("Extracted file differs from the retained archive")
                expected[name] = {"type": "file", "sha256": value.hexdigest(), "bytes": member.size}
                with target.open("rb") as content:
                    header = content.read(64)
                if header[:4] == b"\x7fELF":
                    machine = struct.unpack_from("<H", header, 18)[0]
                    if header[4:7] != b"\x02\x01\x01" or machine != {"linux-x86_64": 62, "linux-arm64": 183}[platform]:
                        raise ValueError("Extracted ELF differs from the expected platform")
                    native[name] = {"sha256": value.hexdigest(), "machine": machine}
            else:
                raise ValueError("Independent reader found another archive type")
    actual = set()
    for parent, directories, files in os.walk(expanded, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            if path.is_symlink() or path.is_file():
                actual.add(path.relative_to(expanded).as_posix())
    if actual != set(expected):
        raise ValueError("Extracted tree has missing or additional entries")
    return {"compared_files_and_links": len(expected), "tree_content_sha256": hashlib.sha256(canonical(expected)).hexdigest(),
            "elf_objects": native}


def run(root, output, *, temporary_root=None):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    evidence = root / "evidence/runtime-sources-20261014-01"
    downloaded = json.loads((evidence / "download-receipt.json").read_text())["files"]
    dependencies = json.loads((evidence / "package-audit.json").read_text())["results"]
    inputs = []
    for entry in downloaded:
        python = entry["name"].startswith("cpython")
        platform = "linux-arm64" if "aarch64" in entry["name"] or "arm64" in entry["name"] else "linux-x86_64"
        inputs.append((root / "vendor/downloads" / entry["name"], entry["sha256"], entry["bytes"], platform,
                       "python" if python else "inference", "python" if python else "llama-b9850"))
    for entry in dependencies:
        source = root / "vendor" / ("dependencies-" + entry["architecture"] + ".tar.gz")
        inputs.append((source, entry["archive_sha256"], source.stat().st_size, entry["architecture"], "dependencies", None))
    receipt = {"scope": "Real retained native archives, static extraction and independent byte comparison on Mac",
               "started_at": datetime.now(timezone.utc).isoformat(), "results": [], "passed": False,
               "native_execution_verified": False, "complete_runtime_dependency_closure": False,
               "model_payload_copied": False, "qualification_passed": False}
    started = time.monotonic()
    try:
        for source, expected, size, platform, role, prefix in inputs:
            record = {"archive": source.name, "sha256": expected, "bytes": size, "platform": platform, "role": role}
            receipt["results"].append(record)
            if source.stat().st_size != size or digest(source) != expected:
                raise ValueError("Retained source archive identity changed")
            with tempfile.TemporaryDirectory(prefix="organism-archive-audit-", dir=temporary_root) as temporary:
                expanded = Path(temporary) / "expanded"
                record["extraction"] = extract_archive(
                    source, expanded, strip_prefix=prefix,
                    limits=ArchiveLimits(bytes=(256 if role == "inference" else 512) * 1024**2),
                    allowed_roots=("site-packages",) if role == "dependencies" else None,
                )
                record["independent_comparison"] = compare_archive(source, expanded, prefix, platform)
                record["passed"] = True
            print(json.dumps({"archive": source.name, "passed": True, "files": record["extraction"]["files"]}), flush=True)
        receipt["passed"] = len(receipt["results"]) == 6
    except Exception as exc:
        receipt["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        receipt["source_hashes"] = {name: digest(root / name) for name in
                                    ("safe_archive.py", "preparation.py", "audit_preparation_archives.py")}
        (output / "report.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--temporary-root", type=Path)
    args = parser.parse_args()
    receipt = run(Path(__file__).resolve().parent, args.output, temporary_root=args.temporary_root)
    print(json.dumps({"passed": receipt["passed"], "failure": receipt.get("failure"),
                      "elapsed_seconds": receipt["elapsed_seconds"]}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
