"""Verify the release candidate's files using only the Python standard library."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(root, evidence_only=False):
    manifest = json.loads((root / "MANIFEST.json").read_text())
    checked, total = 0, 0
    for relative, expected in manifest["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Manifest path escapes the supplied package")
        if evidence_only and path.parts[0] == "assets":
            continue
        actual = root / path
        if not actual.is_file() or actual.is_symlink():
            raise ValueError("Missing or nonregular package file: " + relative)
        digest, count = hashlib.sha256(), 0
        with actual.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
                count += len(block)
        if {"bytes": count, "sha256": digest.hexdigest()} != expected:
            raise ValueError("Package identity differs: " + relative)
        checked += 1
        total += count
    result = {"files_verified": checked, "bytes_verified": total,
              "evidence_only": evidence_only, "identity_match": True,
              "new_experiment_performed": False}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--evidence-only", action="store_true")
    args = parser.parse_args()
    verify(args.package.resolve(), args.evidence_only)
