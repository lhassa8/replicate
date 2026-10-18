"""Package the public carried smoke worker; no qualification assets or SDK."""

import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tarfile


def package(source, output):
    source, output = Path(source), Path(output)
    entry = source.lstat()
    if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1 or entry.st_size > 65536:
        raise ValueError("Self-test worker must be a bounded regular source file")
    raw = source.read_bytes()
    imports = set()
    for node in ast.walk(ast.parse(raw)):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("Self-test must run without project imports")
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            imports.update(item.name.split(".")[0] for item in node.names)
    if not imports <= sys.stdlib_module_names:
        raise ValueError("Self-test may depend only on the standard library")
    with output.open("xb") as target:
        with gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                info = tarfile.TarInfo("readiness.py")
                info.mode, info.size = 0o644, len(raw)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(raw))
    return {"schema": "organism-development-self-test-v1", "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "archive_bytes": output.stat().st_size, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_bytes": len(raw), "standard_library_imports": sorted(imports),
            "entrypoint": "readiness.py", "public_smoke_case": "challenge-bound Celsius measurement and range assessment",
            "qualification_cases_included": False, "starts_or_admits_descendants": False,
            "native_execution_verified": False, "qualification_passed": False,
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    result = package(Path(__file__).resolve().parent / "self_test/readiness.py", args.output_directory / "self-test.tar.gz")
    (args.output_directory / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
