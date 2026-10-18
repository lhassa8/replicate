"""Build carried native library components from pinned Ubuntu package bytes.

This is static preparation on the Mac. No ELF input or package maintainer script
is executed. Signature verification and runtime loading remain native checks.
"""

import argparse
import gzip
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import posixpath
import selectors
import subprocess
import tarfile
import tempfile
import time

from .native_preflight.libraries import BASE
from .download_supplemental import declared_package
from .ubuntu_supplemental import save, sha


MAX_EXPANDED = 128 * 1024**2
ZSTD = "/opt/homebrew/bin/zstd"
ARCHITECTURES = {"amd64": ("linux-x86_64", "EM_X86_64"), "arm64": ("linux-arm64", "EM_AARCH64")}


def deb_data(raw):
    if not raw.startswith(b"!<arch>\n"):
        raise ValueError("Debian package is not an ar archive")
    offset, entries = 8, {}
    while offset < len(raw):
        header = raw[offset:offset + 60]
        if len(header) != 60 or header[58:] != b"`\n":
            raise ValueError("Malformed Debian ar header")
        name = header[:16].decode("ascii").strip().removesuffix("/")
        length = int(header[48:58])
        if name in entries or not 0 <= length <= 16 * 1024**2 or offset + 60 + length > len(raw):
            raise ValueError("Duplicate, truncated or oversized Debian member")
        entries[name] = raw[offset + 60:offset + 60 + length]
        offset += 60 + length + length % 2
    if offset != len(raw) or entries.get("debian-binary") != b"2.0\n":
        raise ValueError("Unsupported Debian package layout")
    candidates = [name for name in entries if name in ("data.tar.zst", "data.tar.xz", "data.tar.gz")]
    if len(entries) != 3 or len(candidates) != 1 or not any(name.startswith("control.tar.") for name in entries):
        raise ValueError("Unexpected Debian members")
    name = candidates[0]
    return name, entries[name]


def decompress_data(name, raw):
    if name.endswith(".zst"):
        with tempfile.TemporaryDirectory(prefix="or-zstd-") as temporary:
            source = Path(temporary) / "data.zst"
            source.write_bytes(raw)
            with (Path(temporary) / "stderr").open("wb") as errors:
                process = subprocess.Popen([ZSTD, "-dc", "-M128MB", str(source)],
                                           stdout=subprocess.PIPE, stderr=errors)
                pieces, count, deadline = [], 0, time.monotonic() + 30
                try:
                    with selectors.DefaultSelector() as selector:
                        selector.register(process.stdout, selectors.EVENT_READ)
                        while True:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0 or not selector.select(remaining):
                                raise TimeoutError("Pinned package decompression timed out")
                            block = process.stdout.read1(64 * 1024)
                            if not block:
                                break
                            count += len(block)
                            if count > MAX_EXPANDED:
                                raise ValueError("Debian data exceeds its expansion allowance")
                            pieces.append(block)
                    if process.wait(timeout=max(0.01, deadline - time.monotonic())) != 0:
                        raise ValueError("Zstandard could not decode the package data")
                    return b"".join(pieces)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
                    process.stdout.close()
    if name.endswith(".xz"):
        decoder = lzma.LZMADecompressor(memlimit=128 * 1024**2)
        value = decoder.decompress(raw, max_length=MAX_EXPANDED + 1)
        if not decoder.eof or decoder.unused_data or len(value) > MAX_EXPANDED:
            raise ValueError("Debian XZ data is incomplete or exceeds its allowance")
        return value
    if name.endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            value = stream.read(MAX_EXPANDED + 1)
        if len(value) > MAX_EXPANDED:
            raise ValueError("Debian gzip data exceeds its allowance")
        return value
    raise ValueError("Unsupported Debian data compression")


def deb_entries(raw):
    name, payload = deb_data(raw)
    data = decompress_data(name, payload)
    entries, expanded = {}, 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for index, member in enumerate(archive, 1):
            name = member.name.removeprefix("./").rstrip("/")
            if name in ("", ".") and member.isdir():
                continue
            if (index > 10000 or name.startswith("/") or name in entries
                    or any(part in ("", ".", "..") for part in name.split("/"))):
                raise ValueError("Unsafe or duplicate Debian data path")
            if member.isdir():
                entries[name] = ("directory", None)
            elif member.issym():
                entries[name] = ("link", member.linkname)
            elif member.isfile():
                expanded += member.size
                if expanded > MAX_EXPANDED or member.size < 0:
                    raise ValueError("Debian file inventory exceeds its allowance")
                value = archive.extractfile(member).read(member.size + 1)
                if len(value) != member.size:
                    raise ValueError("Truncated Debian data file")
                entries[name] = ("file", value)
            else:
                raise ValueError("Special Debian data entries are not supported")
    return entries


def resolve_file(entries, name):
    visited = set()
    for _ in range(16):
        if name in visited or name.startswith("../") or name.startswith("/"):
            raise ValueError("Debian notice link is missing, cyclic or outside the package tree")
        visited.add(name)
        parts = name.split("/")
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            if prefix not in entries:
                raise ValueError("Debian notice path is absent from the package tree")
            kind, value = entries[prefix]
            if kind == "link":
                name = posixpath.normpath(posixpath.join(posixpath.dirname(prefix), value, *parts[index:]))
                break
            if index == len(parts) and kind == "file":
                return value
            if kind != "directory":
                raise ValueError("Expected a regular Debian notice file")
        else:
            raise ValueError("Debian notice resolved to a directory")
    raise ValueError("Debian notice link exceeds its resolution allowance")


def elf_info(raw, machine):
    from elftools.elf.elffile import ELFFile
    elf = ELFFile(io.BytesIO(raw))
    if elf.elfclass != 64 or not elf.little_endian or elf.header.e_machine != machine:
        raise ValueError("ELF object differs from the declared native architecture")
    needed, soname, versions, definitions = [], None, {}, set()
    dynamic = elf.get_section_by_name(".dynamic")
    if dynamic:
        for tag in dynamic.iter_tags():
            if tag.entry.d_tag == "DT_NEEDED":
                needed.append(tag.needed)
            elif tag.entry.d_tag == "DT_SONAME":
                soname = tag.soname
    requirements = elf.get_section_by_name(".gnu.version_r")
    if requirements:
        for dependency, auxiliaries in requirements.iter_versions():
            versions[dependency.name] = [aux.name for aux in auxiliaries]
    defined = elf.get_section_by_name(".gnu.version_d")
    if defined:
        for _, auxiliaries in defined.iter_versions():
            definitions.update(aux.name for aux in auxiliaries)
    return {"machine": machine, "needed": needed, "soname": soname,
            "symbol_versions": versions, "defined_versions": sorted(definitions)}


def collect_packages(directory, architecture, target, metadata):
    root, target = Path(directory), Path(target)
    receipt = json.loads((root / "receipt.json").read_bytes())
    if receipt["failures"]:
        raise ValueError("Ubuntu acquisition contains unresolved failures")
    _, machine = ARCHITECTURES[architecture]
    union, package_entries, packages = {}, {}, []
    selected = [entry for entry in receipt["packages"] if entry["architecture"] == architecture]
    if {entry["declaration"]["Package"] for entry in selected} != {
            "base-files", "gcc-14-base", "libgcc-s1", "libgomp1", "libstdc++6", "libssl3t64", "libcrypt1"}:
        raise ValueError("Expected the seven pinned library and notice packages")
    for entry in selected:
        expected = declared_package(metadata, architecture, entry["pocket"],
                                    entry["declaration"]["Package"], entry["declaration"]["Version"])
        if expected != entry["declaration"]:
            raise ValueError("Package declaration differs from the retained publisher metadata")
        path = root / entry["relative_path"]
        raw = path.read_bytes()
        if path.is_symlink() or sha(raw) != entry["declaration"]["SHA256"] or len(raw) != int(entry["declaration"]["Size"]):
            raise ValueError("Pinned Debian package changed")
        contents = deb_entries(raw)
        for name, value in contents.items():
            if name in union and union[name] != value:
                raise ValueError("Pinned Debian packages contain conflicting entries")
            union[name] = value
        package = entry["declaration"]["Package"]
        package_entries[package] = (entry, contents)
        packages.append(entry)
    target.mkdir(parents=True, exist_ok=False)
    (target / "lib").mkdir()
    (target / "notices").mkdir()
    (target / "notices/common-licenses").mkdir()
    for name in union:
        if name.startswith("usr/share/common-licenses/"):
            raw = resolve_file(union, name)
            (target / "notices/common-licenses" / PurePosixPath(name).name).write_bytes(raw)
    libraries, objects = {}, []
    for package, (entry, contents) in package_entries.items():
        notice = resolve_file(union, "usr/share/doc/" + package + "/copyright")
        if not notice.strip():
            raise ValueError("Package copyright notice is empty")
        (target / "notices" / (package + ".copyright")).write_bytes(notice)
        if package == "base-files":
            # Only common license texts and their provenance are carried from
            # base-files. OS configuration and utilities are never included.
            continue
        for name, (kind, raw) in contents.items():
            if kind != "file" or not raw.startswith(b"\x7fELF"):
                continue
            info = elf_info(raw, machine)
            name_in_bundle = info["soname"] or PurePosixPath(name).name
            if "/" in name_in_bundle or name_in_bundle in ("", ".", ".."):
                raise ValueError("Noncanonical native library identity")
            identity = sha(raw)
            if name_in_bundle in libraries:
                if libraries[name_in_bundle]["sha256"] != identity:
                    raise ValueError("Conflicting native SONAMEs in the supplemental package")
                continue
            (target / "lib" / name_in_bundle).write_bytes(raw)
            libraries[name_in_bundle] = {"sha256": identity, "bytes": len(raw), "package": package,
                                        "package_version": entry["declaration"]["Version"], "source_path": name,
                                        "publisher_package_sha256": entry["declaration"]["SHA256"]}
            objects.append({"path": "supplemental/lib/" + name_in_bundle, **info})
    expected = {"libgcc_s.so.1", "libstdc++.so.6", "libgomp.so.1", "libssl.so.3", "libcrypto.so.3", "libcrypt.so.1"}
    if not expected <= set(libraries):
        raise ValueError("Supplemental runtime libraries are missing")
    publisher = target / "publisher-metadata"
    publisher.mkdir()
    for pocket in sorted({entry["pocket"] for entry in selected}):
        location = publisher / pocket
        location.mkdir()
        for name in ("InRelease", "Packages.xz", "receipt.json"):
            (location / name).write_bytes((Path(metadata) / architecture / pocket / name).read_bytes())
    report = {"scope": "Static ELF inventory from pinned official Ubuntu packages; native loading and metadata signatures pending",
              "architecture": architecture, "carried_libraries": libraries, "elf_objects": objects,
              "packages": packages, "openpgp_verified": False, "native_execution_verified": False,
              "dynamic_loading_verified": False, "qualification_passed": False}
    save(target / "library-resolution.json", report)
    return report


def archive_objects(path, machine):
    objects = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.size > 128 * 1024**2:
                raise ValueError("Runtime object exceeds the static inspection bound")
            stream = archive.extractfile(member)
            header = stream.read(4)
            if header != b"\x7fELF":
                continue
            raw = header + stream.read()
            objects.append({"path": member.name, **elf_info(raw, machine)})
    return objects


def static_closure(objects):
    providers = {}
    for item in objects:
        for name in {PurePosixPath(item["path"]).name, item["soname"]} - {None}:
            providers.setdefault(name, []).append(item)
    unresolved, version_gaps, base_versions = [], [], set()
    for item in objects:
        for name in item["needed"]:
            lookup = PurePosixPath(name).name if name.startswith("$ORIGIN/") else name
            if lookup in BASE:
                continue
            if lookup not in providers:
                unresolved.append({"object": item["path"], "library": name})
        for name, required in item["symbol_versions"].items():
            if name in BASE:
                base_versions.update(value for value in required if value.startswith("GLIBC_"))
                continue
            offered = set().union(*(set(value["defined_versions"]) for value in providers.get(name, [])))
            missing = set(required) - offered
            if missing:
                version_gaps.append({"object": item["path"], "library": name, "versions": sorted(missing)})
    if unresolved or version_gaps:
        raise ValueError("Static library closure is incomplete: " + json.dumps({
            "unresolved": unresolved, "version_gaps": version_gaps})[:4000])
    versions = [value for value in base_versions if value[6:7].isdigit()]
    return {"elf_objects": len(objects), "unresolved_non_substrate_libraries": unresolved,
            "missing_non_substrate_symbol_versions": version_gaps,
            "highest_required_glibc_symbol_version": max(versions, key=lambda value: tuple(map(int, value[6:].split(".")))),
            "native_loading_verified": False}


def dependency_archive(base, supplemental, target):
    """Preserve the prior site-packages bytes and add the carried native files."""
    entries = {}
    with tarfile.open(base, "r:gz") as archive:
        for member in archive:
            if (not member.isfile() or not member.name.startswith("site-packages/")
                    or any(part in ("", ".", "..") for part in member.name.split("/"))
                    or member.name in entries):
                raise ValueError("Unexpected base dependency member")
            entries[member.name] = (archive.extractfile(member).read(), member.mode & 0o777)
    for path in Path(supplemental).rglob("*"):
        if path.is_file():
            entries["supplemental/" + path.relative_to(supplemental).as_posix()] = (path.read_bytes(), 0o644)
    if len(entries) > 30000 or sum(len(raw) for raw, _ in entries.values()) > 512 * 1024**2:
        raise ValueError("Combined dependency archive exceeds its preparation bound")
    with Path(target).open("xb") as output, gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, (raw, mode) in sorted(entries.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.uid, info.gid, info.mtime = len(raw), mode, 0, 0, 0
                archive.addfile(info, io.BytesIO(raw))
    return {"bytes": Path(target).stat().st_size, "sha256": sha(Path(target).read_bytes()),
            "expanded_bytes": sum(len(raw) for raw, _ in entries.values()), "files": len(entries)}


def build(workspace, packages, output):
    workspace, packages, output = Path(workspace), Path(packages), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    vendor = workspace / "organism/vendor"
    pinned = json.loads((workspace / "organism/evidence/runtime-sources-20261014-01/download-receipt.json").read_bytes())
    prior = json.loads((workspace / "organism/evidence/runtime-sources-20261014-01/package-audit.json").read_bytes())
    acquisition = json.loads((packages / "receipt.json").read_bytes())
    metadata = workspace / "organism" / acquisition["pins"]["metadata"]
    results = []
    for architecture, (platform, machine) in ARCHITECTURES.items():
        target = output / platform
        report = collect_packages(packages, architecture, target / "supplemental", metadata)
        base = vendor / ("dependencies-" + platform + ".tar.gz")
        expected = next(entry["archive_sha256"] for entry in prior["results"] if entry["architecture"] == platform)
        if sha(base.read_bytes()) != expected:
            raise ValueError("Original dependency archive changed")
        objects = report["elf_objects"] + archive_objects(base, machine)
        for entry in pinned["files"]:
            native = ("arm64" in entry["name"] or "aarch64" in entry["name"]) == (architecture == "arm64")
            if native:
                path = vendor / "downloads" / entry["name"]
                if path.stat().st_size != entry["bytes"] or sha(path.read_bytes()) != entry["sha256"]:
                    raise ValueError("Pinned native runtime archive changed")
                objects.extend(archive_objects(path, machine))
        closure = static_closure(objects)
        save(target / "static-closure.json", closure)
        archive = output / ("dependencies-" + platform + ".tar.gz")
        result = dependency_archive(base, target / "supplemental", archive)
        results.append({"platform": platform, "archive": archive.name, **result, "static_closure": closure,
                        "supplemental_libraries": len(report["carried_libraries"]),
                        "base_dependency_sha256": expected})
    report = {"scope": "Static complete dependency candidates for both platforms; native loading and OpenPGP checks pending",
              "results": results, "builder_sha256": sha(Path(__file__).read_bytes()),
              "packages_receipt_sha256": sha((packages / "receipt.json").read_bytes()),
              "native_execution_verified": False, "qualification_passed": False}
    save(output / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(Path(__file__).resolve().parents[1], args.packages, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
