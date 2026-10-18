"""Reconstruct the paper supplement from portable copies of recorded evidence.

This does not start inference, contact a cloud service, or reproduce a live birth.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

from organism.research import auditor_mutations as experiment
from organism.research.heldout_study import audit as audit_holdout


def replay_mutations(source, output):
    source = Path(source)
    protocol = json.loads((source / "protocol.json").read_text())
    saved = json.loads((source / "results.json").read_text())
    if experiment.identity(source / "protocol.json") != saved["protocol"]:
        raise ValueError("Mutation protocol identity changed")
    for name, expected in protocol["source_files"].items():
        if (experiment.identity(source / "source" / name) != expected
                or experiment.identity(experiment.ROOT / name) != expected):
            raise ValueError("Frozen auditor or mutation implementation changed: " + name)
    rows = []
    with tempfile.TemporaryDirectory(prefix="replicate-record-replay-") as temporary:
        work = Path(temporary)
        for entry in saved["input_sets"]:
            birth = entry["birth"]
            original = source / "inputs" / birth
            for name in ("material", "identities"):
                if experiment.identity(original / (name + ".json")) != entry[name]:
                    raise ValueError("Input manifest identity changed: " + birth)
            identities = json.loads((original / "identities.json").read_text())
            for name, expected in identities.items():
                if experiment.identity(original / name) != expected:
                    raise ValueError("Input evidence identity changed: " + name)
            directory = work / birth
            shutil.copytree(original, directory)
            material = json.loads((directory / "material.json").read_text())
            for category in protocol["positive_checks_per_birth"]:
                result = experiment.observed(lambda: experiment.check(
                    material, category, directory / "parent", directory / "child"))
                rows.append({"birth": birth, "category": category, "variant": "valid_control",
                             "expected_acceptance": True, **result})
            database = directory / "child/seed-evidence/control/state_directory/peer-audit.sqlite3"
            for mutation in protocol["mutations_per_birth"]:
                category, name = mutation["category"], mutation["name"]
                changed = experiment.mutate(material, category, name)
                initial = database.read_bytes()
                try:
                    if category == "chain":
                        statement = {
                            "deleted_peer_record": "DELETE FROM records WHERE sequence=2",
                            "modified_peer_record": "UPDATE records SET body='{}' WHERE sequence=2",
                            "modified_history_head": "UPDATE head SET digest='" + "0" * 64 + "'",
                        }[name]
                        with sqlite3.connect(database) as db:
                            db.execute(statement)
                    result = experiment.observed(lambda: experiment.check(
                        changed, category, directory / "parent", directory / "child"))
                finally:
                    database.write_bytes(initial)
                rows.append({"birth": birth, "category": category, "variant": name,
                             "expected_acceptance": False,
                             "outer_job_hash_recomputed": category == "decision", **result})
            for name, expected in identities.items():
                if experiment.identity(directory / name) != expected:
                    raise ValueError("Replay failed to restore an evidence copy")
    if rows != saved["results"]:
        raise ValueError("Mutation decisions or rejection reasons did not reproduce exactly")
    result = {
        "replay_kind": "recorded-evidence reconstruction, not new native execution",
        "source_results": experiment.identity(source / "results.json"),
        "native_births": len(saved["input_sets"]),
        "valid_controls_accepted": sum(r["accepted"] for r in rows if r["expected_acceptance"]),
        "invalid_variants_rejected": sum(not r["accepted"] for r in rows if not r["expected_acceptance"]),
        "exact_results_match": True,
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def replay_holdout(source, output):
    """Run the original scorer on a copy; ignore only the replay timestamp."""
    source = Path(source)
    saved = json.loads((source / "results.json").read_text())
    freeze = json.loads((source / "agent-freeze.json").read_text())
    for name, expected in freeze["source_files"].items():
        if experiment.identity(experiment.ROOT / name) != expected:
            raise ValueError("Active reconstruction source differs from the experiment: " + name)
    with tempfile.TemporaryDirectory(prefix="replicate-holdout-replay-") as temporary:
        copied = Path(temporary) / "heldout"
        shutil.copytree(source, copied)
        reconstructed = audit_holdout(copied)
    comparable = deepcopy(saved)
    comparable.pop("reconstructed_at")
    reconstructed.pop("reconstructed_at")
    if reconstructed != comparable:
        raise ValueError("Held-out summary did not reconstruct exactly")
    result = {
        "replay_kind": "recorded-evidence reconstruction, not new inference",
        "source_results": experiment.identity(source / "results.json"),
        "exact_results_match": True,
        "successes": {name: value["successes"] for name, value in reconstructed["arms"].items()},
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("mutations", "holdout"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (replay_mutations if args.kind == "mutations" else replay_holdout)(
        args.source, args.output)
    print(json.dumps(result))
