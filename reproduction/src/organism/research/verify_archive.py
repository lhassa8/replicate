"""Verify every archive member against its manifest without extracting models."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def verify(archive, output):
    with tarfile.open(archive) as stream:
        candidates = [m for m in stream.getmembers()
                      if len(PurePosixPath(m.name).parts) == 2
                      and PurePosixPath(m.name).name == "MANIFEST.json"]
        if len(candidates) != 1 or not candidates[0].isfile():
            raise ValueError("Expected exactly one regular top-level package manifest")
        manifest_member = candidates[0]
        prefix = PurePosixPath(manifest_member.name).parts[0]
        manifest = json.loads(stream.extractfile(manifest_member).read())
        seen, total = set(), 0
        for member in stream:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or name.parts[0] != prefix:
                raise ValueError("Archive path escapes its package")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Archive contains a link or special file: " + member.name)
            relative = str(name.relative_to(prefix))
            if relative in seen:
                raise ValueError("Archive contains a duplicate file: " + relative)
            seen.add(relative)
            if relative == "MANIFEST.json":
                continue
            if relative not in manifest["files"]:
                raise ValueError("Archive contains an unlisted file: " + relative)
            digest, count = hashlib.sha256(), 0
            with stream.extractfile(member) as data:
                for block in iter(lambda: data.read(1024**2), b""):
                    digest.update(block)
                    count += len(block)
            if {"bytes": count, "sha256": digest.hexdigest()} != manifest["files"][relative]:
                raise ValueError("Archive member differs: " + relative)
            total += count
        if seen != set(manifest["files"]) | {"MANIFEST.json"}:
            raise ValueError("Archive does not contain the complete manifest inventory")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    result = {"archive": archive.name, "bytes": archive.stat().st_size,
              "sha256": digest.hexdigest(), "member_identities_verified": len(manifest["files"]),
              "member_bytes_verified": total, "complete_inventory_match": True,
              "models_extracted": False}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.archive.resolve(), args.output.resolve())
