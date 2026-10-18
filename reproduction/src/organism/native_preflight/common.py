"""Standard-library helpers shared by the launch builder and native worker."""

import hashlib
import json
from pathlib import Path
import re
import tarfile


ARCHES = ("x86_64", "arm64")
MAX_OUTPUT_BYTES = 512 * 1024**2


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_file(path, specification):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular input: {path.name}")
    if path.stat().st_size != specification["bytes"] or digest(path) != specification["sha256"]:
        raise ValueError(f"Input identity differs: {path.name}")


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.+-]{0,159}", value):
        raise ValueError("Invalid artifact name")
    return value


def extract(path, destination, *, limit=512 * 1024**2):
    """Extract a pinned archive with Python's data filter and a size ceiling."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 30_000 or sum(max(0, m.size) for m in members) > limit:
            raise ValueError("Archive exceeds extraction bound")
        # The data filter rejects traversal, devices and links outside the root.
        # A new private destination is required by callers.
        archive.extractall(destination, members=members, filter="data")


def validate_catalog(value):
    if value.get("schema") != "organism-native-preflight-v2" or value.get("qualification_passed") is not False:
        raise ValueError("Wrong native experiment schema")
    if (value.get("supplemental_libraries_carried") is not True
            or value.get("publisher_signature_verification_required_before_execution") is not True):
        raise ValueError("Native input catalog must require carried and authenticated libraries")
    if set(value["platforms"]) != set(ARCHES):
        raise ValueError("Both native architectures are required")
    names = set()
    for entry in value["files"]:
        safe_name(entry["name"])
        if entry["name"] in names or not isinstance(entry["bytes"], int) or isinstance(entry["bytes"], bool):
            raise ValueError("Duplicate or malformed input")
        names.add(entry["name"])
        if not 0 < entry["bytes"] <= 3 * 1024**3 or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("Input outside size or hash contract")
    expected = {value["model"]}
    for entry in value["platforms"].values():
        if set(entry) != {"python", "inference", "dependencies"}:
            raise ValueError("Incomplete native platform")
        expected.update(entry.values())
    if names != expected or len(names) != 7:
        raise ValueError("Unexpected or missing native inputs")
    return value
