"""Restore startup diagnostics while retaining the complete v3 pilot verbatim."""

import argparse
import asyncio
import base64
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

from strands.models import LlamaCppModel

from . import inheritance_dataset as inherited
from . import runtime_dataset as legacy
from .audit_runtime_dataset import canonical, digest, require
from .runtime_rows import STARTUP_CORPUS_SCHEMA


DATASET_ID = "equipment-runtime-development-v4"
PARENT_MANIFEST_SHA256 = "1287685e4fc24dbbc2aa69afa9990b5a4b7d57e29f91db564fba14ada2205475"
STARTUP_PROFILES = ("a", "b", "c", "d", "g", "h")
SOURCES = tuple(sorted(set(inherited.SOURCES) | {
    "startup_dataset.py", "audit_startup_dataset.py", "runtime_rows.py",
}))


def startup_cases():
    """Six existing development profiles; qualification cases are never imported."""
    for original in legacy.development_cases():
        if original["role"] != "diagnostic" or original["profile_id"][-1] not in STARTUP_PROFILES:
            continue
        # Each selected profile retains all six conditions at one numeric seed.
        if original["seed"] != 2:
            continue
        case = deepcopy(original)
        identity = "readiness-" + digest(canonical({
            "dataset": DATASET_ID, "profile": case["profile_id"],
            "variation": case["variation"], "seed": case["seed"],
        }))[:32]
        case["case_id"] = case["observation"]["device_id"] = identity
        case["observation"]["sensor_id"] = "temperature"
        yield case


def contracts(model):
    definitions = inherited.contracts(model)
    definitions["diagnostic"] = legacy.contracts(model)["diagnostic"]
    return definitions


def pinned_parent(manifest_raw, contracts_raw):
    require(digest(manifest_raw) == PARENT_MANIFEST_SHA256, "Expected the preserved v3 parent manifest")
    manifest = json.loads(manifest_raw)
    require(manifest["dataset_id"] == inherited.DATASET_ID and manifest["counts"]["train"]["trajectories"] == 348
            and manifest["counts"]["validation"]["trajectories"] == 116, "Parent pilot inventory differs")
    require(manifest["files"]["contracts.json"] == {"bytes": len(contracts_raw), "sha256": digest(contracts_raw)},
            "Parent contracts differ")
    return manifest, json.loads(contracts_raw)


def read_parent(parent):
    parent = Path(parent)
    manifest_raw, contracts_raw = (parent / "manifest.json").read_bytes(), (parent / "contracts.json").read_bytes()
    manifest, definitions = pinned_parent(manifest_raw, contracts_raw)
    files = {}
    for name, expected in manifest["files"].items():
        require(Path(name).name == name, "Parent filename escapes its directory")
        raw = (parent / name).read_bytes()
        require(expected == {"bytes": len(raw), "sha256": digest(raw)}, "Parent data changed: " + name)
        files[name] = raw
    for name, expected in manifest["sources"].items():
        require(Path(name).name == name, "Parent source filename escapes its directory")
        require(digest((parent / "sources" / name).read_bytes()) == expected, "Parent source snapshot changed: " + name)
    return manifest_raw, contracts_raw, manifest, definitions, files


def retained_prefix(raw, split, parent):
    """The exact original serialized records must prefix the new split."""
    count = parent["counts"][split]["trajectories"]
    lines = raw.splitlines(keepends=True)
    require(len(lines) >= count, "Missing preserved parent trajectories")
    prefix = b"".join(lines[:count])
    require(parent["files"][split + ".jsonl"] == {"bytes": len(prefix), "sha256": digest(prefix)},
            "Preserved parent trajectory bytes changed")
    return [json.loads(line) for line in lines[:count]], [json.loads(line) for line in lines[count:]]


async def build(parent, output):
    require(version("strands-agents") == "1.57.1", "Pinned Strands version required")
    manifest_raw, contracts_raw, parent_manifest, parent_definitions, parent_files = read_parent(parent)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    release = json.loads(parent_files["release.json"])
    model = LlamaCppModel(base_url="http://127.0.0.1:9", model_id="reference-formatting-only",
                         params={"temperature": 0, "max_tokens": 192})
    groups = {split: [json.loads(line) for line in parent_files[split + ".jsonl"].splitlines()]
              for split in ("train", "validation")}
    additions = {split: [] for split in groups}
    try:
        definitions = contracts(model)
        require({k: v for k, v in definitions.items() if k != "diagnostic"} == parent_definitions,
                "Current equipment, inheritance or reproduction contracts changed")
        for case in startup_cases():
            record = await legacy.trajectory(case, model, definitions["diagnostic"], release)
            await legacy.replay(record, model, definitions["diagnostic"], release)
            additions[case["split"]].append(record)
    finally:
        await model.client.aclose()
    require({k: len(v) for k, v in additions.items()} == {"train": 24, "validation": 12},
            "Startup supplement inventory differs")
    for split, extra in additions.items():
        raw = parent_files[split + ".jsonl"]
        require(raw.endswith(b"\n"), "Parent split has no final newline")
        raw += b"".join((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode() for row in extra)
        (output / (split + ".jsonl")).write_bytes(raw)
        groups[split].extend(extra)
    (output / "contracts.json").write_text(json.dumps(definitions, indent=2) + "\n")
    (output / "release.json").write_bytes(parent_files["release.json"])
    (output / "sources").mkdir()
    paths = {name: Path(__file__).parent / name for name in SOURCES}
    paths.update(legacy.provider_sources())
    sources = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        (output / "sources" / name).write_bytes(raw)
        sources[name] = digest(raw)
    manifest = {
        "dataset_id": DATASET_ID, "schema": STARTUP_CORPUS_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "development_references_through_current_runtime_tools",
        "training_admitted": False, "optimization_started": False,
        "qualification_data_present": False, "human_review_claimed": False,
        "strands_version": version("strands-agents"),
        "parent": {"manifest_base64": base64.b64encode(manifest_raw).decode(),
                   "contracts_base64": base64.b64encode(contracts_raw).decode(),
                   "manifest_sha256": PARENT_MANIFEST_SHA256, "preserved_trajectories": 464},
        "counts": {split: inherited.split_counts(rows, definitions) for split, rows in groups.items()},
        "sources": sources,
        "files": {name: {"bytes": (output / name).stat().st_size, "sha256": digest((output / name).read_bytes())}
                  for name in ("train.jsonl", "validation.jsonl", "contracts.json", "release.json")},
        "split_scope": "All parent splits unchanged; diagnostic profiles a-d train and g-h validation.",
        "limitations": [
            "All 464 parent trajectories, including eight context-only incorrect actions, remain byte-for-byte unchanged.",
            "The 36 diagnostic references use the public startup contract; they are not withheld quality cases.",
            "Opaque IDs are newly derived; native failure IDs and model-generated answers are not training targets.",
            "No model inference, optimizer, native execution or human review is invoked by dataset construction.",
            "Startup still requires a successful diagnostic within the unchanged four-call budget.",
            "The supplement teaches exact successful actions, not unknown-device recovery or a relaxed readiness check.",
        ],
    }
    require(sum(row["trajectories"] for row in manifest["counts"].values()) == 500, "Pilot exceeds its allocation")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(build(args.parent, args.output))
    print(json.dumps({"dataset_id": result["dataset_id"], "counts": result["counts"],
                      "training_admitted": False}, indent=2))
