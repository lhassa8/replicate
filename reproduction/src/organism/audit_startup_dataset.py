"""Audit the restored startup corpus, including retained references and replay."""

import argparse
import asyncio
import base64
from importlib.metadata import version
import json
from pathlib import Path

from .audit_runtime_dataset import canonical, digest, require
from .runtime_rows import STARTUP_CORPUS_SCHEMA, inspect_record


async def audit(source):
    from strands.models import LlamaCppModel
    from . import startup_dataset as dataset
    from . import inheritance_dataset as inherited
    from . import runtime_dataset as legacy

    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_bytes())
    require(manifest["schema"] == STARTUP_CORPUS_SCHEMA and manifest["dataset_id"] == dataset.DATASET_ID
            and manifest["scope"] == "development_references_through_current_runtime_tools"
            and manifest["strands_version"] == version("strands-agents") == "1.57.1",
            "Startup corpus or runtime version differs")
    require(all(manifest[field] is False for field in
                ("training_admitted", "optimization_started", "qualification_data_present", "human_review_claimed")),
            "Startup corpus overclaims admission or qualification")
    require(set(manifest["files"]) == {"train.jsonl", "validation.jsonl", "contracts.json", "release.json"},
            "Unexpected startup corpus inventory")
    files = {}
    for name, expected in manifest["files"].items():
        raw = (source / name).read_bytes()
        require(expected == {"bytes": len(raw), "sha256": digest(raw)}, "Startup corpus file hash differs")
        files[name] = raw
    parent, parent_contracts = dataset.pinned_parent(
        base64.b64decode(manifest["parent"]["manifest_base64"], validate=True),
        base64.b64decode(manifest["parent"]["contracts_base64"], validate=True),
    )
    require(manifest["parent"]["manifest_sha256"] == dataset.PARENT_MANIFEST_SHA256
            and manifest["parent"]["preserved_trajectories"] == 464, "Parent provenance differs")
    paths = {name: Path(dataset.__file__).parent / name for name in dataset.SOURCES}
    paths.update(legacy.provider_sources())
    require(set(paths) == set(manifest["sources"]), "Startup source inventory differs")
    for name, path in paths.items():
        require(digest(path.read_bytes()) == digest((source / "sources" / name).read_bytes()) == manifest["sources"][name],
                "Current or retained startup source changed: " + name)
    require(parent["files"]["release.json"] == manifest["files"]["release.json"], "Parent release changed")
    release = json.loads(files["release.json"])
    require(release == legacy.stock_release(Path(dataset.__file__).parent), "Release fixture changed")
    definitions = json.loads(files["contracts.json"])
    require({k: v for k, v in definitions.items() if k != "diagnostic"} == parent_contracts,
            "Preserved tool contract changed")
    cases = {case["case_id"]: case for case in inherited.development_cases()}
    extra_cases = {case["case_id"]: case for case in dataset.startup_cases()}
    require(len(cases) == 464 and len(extra_cases) == 36 and not set(cases) & set(extra_cases),
            "Expected startup case inventory differs")
    cases.update(extra_cases)
    model = LlamaCppModel(base_url="http://127.0.0.1:9", model_id="reference-formatting-only",
                         params={"temperature": 0, "max_tokens": 192})
    counts, seen, profiles = {}, set(), set()
    try:
        require(definitions == dataset.contracts(model), "Current runtime tool contract differs")
        for split in ("train", "validation"):
            preserved, additions = dataset.retained_prefix(files[split + ".jsonl"], split, parent)
            require(len(additions) == (24 if split == "train" else 12)
                    and all(row["case_id"] in extra_cases for row in additions), "Startup supplement changed")
            records = preserved + additions
            split_profiles = {row["profile_id"] for row in records}
            require(not profiles & split_profiles, "Development profiles overlap splits")
            profiles.update(split_profiles)
            for row in records:
                key = row["case_id"]
                require(key not in seen and row["case"] == cases.get(key) and row["split"] == split,
                        "Duplicate, moved or changed startup fixture")
                seen.add(key)
                require(inspect_record(row, definitions[row["role"]], release) == row["evaluation"],
                        "Startup reference semantic audit differs")
                replay = legacy.replay if row["role"] == "diagnostic" else inherited.replay
                await replay(row, model, definitions[row["role"]], release)
            counts[split] = inherited.split_counts(records, definitions)
        require(seen == set(cases) and counts == manifest["counts"], "Startup pilot coverage differs")
        require(counts["train"]["roles"] == {
            "equipment": 180, "reproduction": 60, "equipment_inherited": 108, "diagnostic": 24,
        } and counts["validation"]["roles"] == {
            "equipment": 60, "reproduction": 20, "equipment_inherited": 36, "diagnostic": 12,
        } and counts["train"]["assistant_targets"] == 1680
          and counts["validation"]["assistant_targets"] == 576, "Startup pilot balance or target count differs")
    finally:
        await model.client.aclose()
    return {
        "passed": True, "source_manifest_sha256": digest((source / "manifest.json").read_bytes()),
        "counts": counts, "current_runtime_replay_verified": True, "independent_semantics_verified": True,
        "provider_formatting_verified": True, "parent_trajectories_preserved_verbatim": 464,
        "startup_diagnostic_references_verified": 36, "context_only_bad_actions_verified": 8,
        "opaque_equipment_identifiers_verified": 240, "opaque_startup_identifiers_verified": 36,
        "signed_fixture_source_notes_verified": 144, "human_review_claimed": False,
        "serving_template_parity_verified": False, "training_admitted": False, "qualification_passed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as destination:
        try:
            result = asyncio.run(audit(args.source))
        except Exception as exc:
            result = {"passed": False, "failure": type(exc).__name__ + ": " + str(exc), "training_admitted": False}
        destination.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
