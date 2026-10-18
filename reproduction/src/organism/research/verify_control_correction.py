"""Compare the two frozen deployments without changing either experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def inventory(path):
    with tarfile.open(path) as archive:
        result = {}
        for member in archive:
            if member.isfile():
                if member.name in result:
                    raise ValueError("Duplicate archive member: " + member.name)
                result[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        return result


def verify(previous, current, output):
    before = json.loads((previous / "assets.json").read_text())
    after = json.loads((current / "assets.json").read_text())
    expected = {"organism/research/control_entry.py",
                "organism/research/fixed_lifecycle_candidate.py"}
    changes = {}
    for name in ("agent.tar.gz", "owner-source.tar.gz"):
        path = current / name
        if not path.exists() and name == "agent.tar.gz":
            path = current / "shared/common/agent.tar.gz"
        old, new = inventory(previous / name), inventory(path)
        changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
        if changed != expected:
            raise ValueError("Unexpected carried source changes: " + str(sorted(changed)))
        changes[name] = {"changed_members": sorted(changed),
                         "unchanged_members": len((old.keys() & new.keys()) - changed)}
    unchanged = (
        "models/trained.gguf", "models/stock.gguf", "smoke-cases.json",
        "host.py", "worker-template.json", "storage-template.json",
    )
    for name in unchanged:
        original = before["files"].get(name)
        if original is None:
            raw = (previous / name).read_bytes()
            original = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if original != after["files"][name]:
            raise ValueError("Matched experiment input changed: " + name)
    for name in before["shared"]:
        if name != "common/agent.tar.gz" and before["shared"][name] != after["shared"][name]:
            raise ValueError("Shared payload component changed: " + name)
    if before["order"] != after["order"]:
        raise ValueError("Arm order changed")
    result = {
        "schema": "replicate-frozen-correction-comparison-v1",
        "archive_comparison": changes,
        "unchanged_inputs": {name: after["files"][name] for name in unchanged},
        "unchanged_other_shared_payload_components": True,
        "arm_order_unchanged": True,
        "previous_releases": {name: row["release_id"] for name, row in before["arms"].items()},
        "corrected_releases": {name: row["release_id"] for name, row in after["arms"].items()},
        "observations_from_different_releases_pooled": False,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.previous, args.current, args.output), indent=2))
