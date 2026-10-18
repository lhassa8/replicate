"""Inventory cross-platform wheels and produce deterministic dependency archives."""

import argparse
from email.parser import Parser
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from elftools.elf.elffile import ELFFile


def package(source: Path, output: Path, architecture: str) -> dict:
    if output.exists():
        raise FileExistsError("Do not overwrite a retained dependency archive")
    expected = {"linux-x86_64": "EM_X86_64", "linux-arm64": "EM_AARCH64"}[architecture]
    distributions, native, entries = [], [], []
    for path in sorted(source.glob("*.dist-info/METADATA")):
        metadata = Parser().parsestr(path.read_text())
        distributions.append({
            "name": metadata["Name"], "version": metadata["Version"],
            "license_expression": metadata.get("License-Expression"),
            "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("Dependency payload contains an unexpected symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        with path.open("rb") as stream:
            if stream.read(4) == b"\x7fELF":
                stream.seek(0)
                elf = ELFFile(stream)
                if elf.header.e_machine != expected:
                    raise ValueError("Dependency wheel has the wrong native architecture")
                dynamic = elf.get_section_by_name(".dynamic")
                needed = [tag.needed for tag in dynamic.iter_tags() if tag.entry.d_tag == "DT_NEEDED"] if dynamic else []
                native.append({"path": relative, "machine": elf.header.e_machine, "needed": needed})
        entries.append({
            "path": relative, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    if not native or len(distributions) != 48:
        raise ValueError("Unexpected dependency inventory")
    with output.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for entry in entries:
                path = source / entry["path"]
                info = archive.gettarinfo(str(path), arcname="site-packages/" + entry["path"])
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                with path.open("rb") as payload:
                    archive.addfile(info, payload)
    return {
        "architecture": architecture, "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "unpacked_bytes": sum(e["bytes"] for e in entries),
        "distributions": distributions, "native_extensions": native, "files": entries,
        "execution_verified": False,
        "entrypoint_policy": "Invoke the bundled Python interpreter explicitly. Build-host console-script shebangs are not portable activation commands.",
        "license_review_complete": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = []
    for architecture in ("linux-x86_64", "linux-arm64"):
        source = args.vendor / architecture / "site-packages"
        target = args.vendor / f"dependencies-{architecture}.tar.gz"
        reports.append(package(source, target, architecture))
    args.output.write_text(json.dumps({"packages": reports}, indent=2) + "\n")
    print(json.dumps([{
        "architecture": r["architecture"], "archive_bytes": r["archive_bytes"],
        "unpacked_bytes": r["unpacked_bytes"], "distributions": len(r["distributions"]),
        "native_extensions": len(r["native_extensions"]),
    } for r in reports], indent=2))
