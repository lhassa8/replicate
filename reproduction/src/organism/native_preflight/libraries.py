"""Collect resolved non-substrate ELF libraries on the native preparation host.

This uses ldd only on hash-verified first-party runtime candidates and pinned
dependencies. It is not an untrusted-binary inspection service. Runtime-loaded
libraries can exceed DT_NEEDED; successful inference remains a separate check.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess

from .common import digest, write_json


BASE = frozenset({
    "libc.so.6", "libm.so.6", "libpthread.so.0", "librt.so.1", "libdl.so.2",
    "libutil.so.1", "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1",
})


def elf_files(roots):
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            with path.open("rb") as source:
                if source.read(4) == b"\x7fELF":
                    yield path


def parse_ldd(text):
    if re.search(r"=>\s+not found", text):
        raise ValueError("Unresolved native library: " + text.strip()[:2000])
    found = []
    for line in text.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(/\S+)\s+\(", line)
        if match:
            found.append((match.group(1), Path(match.group(2))))
            continue
        direct = re.match(r"\s*(/\S+)\s+\(", line)
        if direct:
            path = Path(direct.group(1))
            found.append((path.name, path))
    return found


def inside(path, roots):
    return any(path.resolve().is_relative_to(root.resolve()) for root in roots)


def resolve(binary, paths):
    result = subprocess.run(
        ["/usr/bin/ldd", str(binary)], text=True, capture_output=True, timeout=20,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LD_LIBRARY_PATH": ":".join(map(str, paths))},
    )
    text = result.stdout + result.stderr
    if result.returncode and ("not a dynamic executable" in text or "statically linked" in text):
        return []
    if result.returncode:
        raise RuntimeError("Native library inspection failed: " + str(binary))
    return parse_ldd(text)


def collect(root, output):
    root, output = Path(root), Path(output)
    library_dir = root / "supplemental/lib"
    notice_dir = root / "supplemental/notices"
    library_dir.mkdir(parents=True)
    notice_dir.mkdir()
    roots = [root / "python", root / "tools", root / "site-packages"]
    search = [library_dir, root / "python/lib", root / "tools/llama.cpp/b9850/llama-b9850"]
    objects = list(elf_files(roots))
    records, substrate = {}, {}
    for binary in objects:
        for name, path in resolve(binary, search):
            if name in BASE:
                substrate[name] = {"path": str(path), "sha256": digest(path)}
                continue
            if inside(path, roots + [library_dir]):
                continue
            if "/" in name or name in (".", ".."):
                raise ValueError("Invalid native library name")
            target = library_dir / name
            if target.exists() and digest(target) != digest(path):
                raise ValueError("Conflicting native library resolutions")
            shutil.copyfile(path.resolve(), target)
            target.chmod(0o644)
            owner = subprocess.run(
                ["dpkg-query", "-S", str(path)], text=True, capture_output=True, timeout=10,
            )
            # Ubuntu's merged /usr may spell the same path two ways.
            if owner.returncode:
                owner = subprocess.run(
                    ["dpkg-query", "-S", str(path.resolve())], text=True, capture_output=True, timeout=10,
                )
            if owner.returncode or not owner.stdout.strip():
                raise ValueError(f"No package provenance for {name}")
            package = owner.stdout.split(": ", 1)[0].splitlines()[0]
            package_version = subprocess.check_output(
                ["dpkg-query", "-W", "-f=${Version}", package], text=True, timeout=10,
            )
            copyright_path = Path("/usr/share/doc") / package.split(":")[0] / "copyright"
            if not copyright_path.is_file():
                raise ValueError(f"No copyright notice for {package}")
            shutil.copyfile(copyright_path, notice_dir / (package.replace(":", "_") + ".copyright"))
            records[name] = {
                "source_path": str(path), "sha256": digest(target), "bytes": target.stat().st_size,
                "package": package, "package_version": package_version,
            }
    # Re-resolve both original binaries and copied libraries. Non-base system
    # resolutions are a hard preparation failure, never an implicit dependency.
    for binary in objects + list(elf_files([library_dir])):
        for name, path in resolve(binary, search):
            if name not in BASE and not inside(path, roots + [library_dir]):
                raise ValueError(f"Unbundled native dependency remains: {name}")
    report = {
        "scope": "Native ldd resolution of pinned inputs; not exhaustive dlopen coverage",
        "elf_objects_checked": len(objects), "base_os_libraries": substrate,
        "carried_libraries": records, "search_paths": list(map(str, search)),
        "qualification_passed": False,
    }
    write_json(output / "library-resolution.json", report)
    return search
