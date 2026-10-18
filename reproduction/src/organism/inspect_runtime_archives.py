"""Static ELF inspection of pinned archives; does not establish native execution."""

import argparse
import hashlib
import io
import json
import posixpath
from pathlib import Path, PurePosixPath
import tarfile

from elftools.elf.elffile import ELFFile


def inspect_archive(path: Path, expected: dict) -> dict:
    if path.stat().st_size != expected["bytes"]:
        raise ValueError("Archive size changed")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError("Archive hash changed")
    binaries = []
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        names = {PurePosixPath(member.name).name for member in members}
        full_names = {member.name for member in members}
        for member in members:
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream.read(4) != b"\x7fELF":
                continue
            stream.seek(0)
            elf = ELFFile(io.BytesIO(stream.read()))
            needed, search_paths, versions = [], [], {}
            dynamic = elf.get_section_by_name(".dynamic")
            if dynamic is not None:
                for tag in dynamic.iter_tags():
                    if tag.entry.d_tag == "DT_NEEDED":
                        needed.append(tag.needed)
                    elif tag.entry.d_tag in {"DT_RPATH", "DT_RUNPATH"}:
                        search_paths.append(getattr(tag, "rpath", getattr(tag, "runpath", "")))
            requirements = elf.get_section_by_name(".gnu.version_r")
            if requirements is not None:
                for dependency, auxiliaries in requirements.iter_versions():
                    versions[dependency.name] = [aux.name for aux in auxiliaries]
            interpreters = [s.get_interp_name() for s in elf.iter_segments() if s.header.p_type == "PT_INTERP"]
            binaries.append({
                "path": member.name, "machine": elf.header.e_machine,
                "class": elf.elfclass, "little_endian": elf.little_endian,
                "needed": needed, "search_paths": search_paths,
                "interpreter": interpreters, "symbol_versions": versions,
            })
    machines = sorted({b["machine"] for b in binaries})
    expected_machine = "EM_AARCH64" if any(x in path.name for x in ("arm64", "aarch64")) else "EM_X86_64"
    if machines != [expected_machine]:
        raise ValueError(f"Unexpected ELF architectures: {machines}")
    external = set()
    for binary in binaries:
        for lib in binary["needed"]:
            if "$ORIGIN" in lib:
                resolved = posixpath.normpath(lib.replace("$ORIGIN", str(PurePosixPath(binary["path"]).parent)))
                if resolved not in full_names:
                    external.add(lib)
            elif lib not in names:
                external.add(lib)
    external = sorted(external)
    glibc = {
        name for binary in binaries for version_names in binary["symbol_versions"].values()
        for name in version_names if name.startswith("GLIBC_") and name[6:7].isdigit()
    }
    maximum_glibc = max(glibc, key=lambda s: tuple(int(n) for n in s[6:].split(".")), default=None)
    return {
        "archive": path.name, "sha256": expected["sha256"],
        "elf_files": binaries, "architectures": machines,
        "external_dynamic_libraries": external,
        "highest_required_glibc_symbol_version": maximum_glibc,
        "note": "Static direct dependencies only. Runtime-loaded libraries and actual OS compatibility need native tests.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    reports = [inspect_archive(args.downloads / item["name"], item) for item in receipt["files"]]
    with args.output.open("x") as output:
        json.dump({"native_execution_verified": False, "archives": reports}, output, indent=2)
        output.write("\n")
    print(json.dumps([{k: r[k] for k in ("archive", "architectures", "external_dynamic_libraries", "highest_required_glibc_symbol_version")} for r in reports], indent=2))
