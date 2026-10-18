"""Reproducible development-only synthetic demonstrations, with replay checks.

No qualification families or answers are generated here. Training admission
remains pending the frozen experiment contract. The generator is deterministic
code, not a teacher model, and is recorded as such in every trajectory.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .setup_reference import messages_from_events, solve
from .setup_simulator import DeviceCase, SCHEMA_VERSION, SetupSession, TOOLS, contract_hash, evaluate_setup


# Each complete API family belongs to one split. Tuple truth is independently
# specified: (raw value, canonical value, expected diagnostic).
FAMILIES = (
    {"id": "dev-api-a", "split": "train", "protocol": "gateway-json-v1", "selector": "/bearing/temp",
     "unit": "C", "canonical": "C", "range": (10, 40), "values": ((28, 28, "normal"), (80, 80, "high"), (0, 0, "low"))},
    {"id": "dev-api-b", "split": "train", "protocol": "gateway-register-v2", "selector": "holding:4102",
     "unit": "F", "canonical": "C", "range": (10, 40), "values": ((95, 35, "normal"), (176, 80, "high"), (32, 0, "low"))},
    {"id": "dev-api-c", "split": "train", "protocol": "condition-json-v1", "selector": "/spindle/rms",
     "unit": "mm/s", "canonical": "mm/s", "range": (0, 4.5), "values": ((2, 2, "normal"), (8, 8, "high"), (1, 1, "normal"))},
    {"id": "dev-api-d", "split": "train", "protocol": "condition-register-v3", "selector": "input:3025",
     "unit": "in/s", "canonical": "mm/s", "range": (0, 4.5), "values": ((0.1, 2.54, "normal"), (0.3, 7.62, "high"), (0.05, 1.27, "normal"))},
    {"id": "dev-api-e", "split": "train", "protocol": "inspection-status-v1", "selector": "/quality/reject_fraction",
     "unit": "ratio", "canonical": "percent", "range": (0, 5), "values": ((0.01, 1, "normal"), (0.12, 12, "high"), (0.02, 2, "normal"))},
    {"id": "dev-api-f", "split": "train", "protocol": "event-counter-v2", "selector": "machine.error.count",
     "unit": "count", "canonical": "count", "range": (0, 2), "values": ((0, 0, "normal"), (9, 9, "high"), (1, 1, "normal"))},
    {"id": "dev-api-g", "split": "validation", "protocol": "thermal-status-v4", "selector": "/probe/absolute",
     "unit": "K", "canonical": "C", "range": (10, 40), "values": ((312.15, 39, "normal"), (353.15, 80, "high"), (273.15, 0, "low"))},
    {"id": "dev-api-h", "split": "validation", "protocol": "inspection-register-v5", "selector": "holding:4199",
     "unit": "percent", "canonical": "percent", "range": (0, 5), "values": ((2, 2, "normal"), (15, 15, "high"), (1, 1, "normal"))},
)
VARIATIONS = (
    "new", "lost_ack", "stale", "wrong_adapter",
    "unsupported", "small_memory", "retired", "occupied",
)
SEEDS = tuple(range(5))


def development_cases():
    for family in FAMILIES:
        for variation in VARIATIONS:
            for seed in SEEDS:
                raw, normalized, diagnostic = family["values"][seed % len(family["values"])]
                key = f"setup-dev-v1/{family['id']}/{variation}/{seed}"
                identity = "d-" + hashlib.sha256(key.encode()).hexdigest()[:16]
                outcome = "rejected" if variation in {"unsupported", "small_memory", "retired"} else (
                    "deferred" if variation == "occupied" else "ready"
                )
                case = DeviceCase(
                    case_id=identity, family_id=family["id"],
                    architecture="riscv64" if variation == "unsupported" else ("arm64" if seed % 2 else "x86_64"),
                    protocol=family["protocol"], selector=family["selector"],
                    unit=family["unit"], value=raw, expected_normalized=normalized,
                    expected_diagnostic=diagnostic, expected_outcome=outcome,
                    canonical_unit=family["canonical"],
                    normal_low=family["range"][0], normal_high=family["range"][1],
                    memory_mib=4096 if variation == "small_memory" else 8192,
                    retired=variation == "retired", occupied=variation == "occupied",
                    age_seconds=600 if variation == "stale" else 2,
                    acknowledgement_lost_once=variation == "lost_ack",
                    existing_wrong_configuration=variation == "wrong_adapter",
                )
                yield family["split"], variation, seed, case


def trajectory(case: DeviceCase, variation: str, seed: int) -> dict:
    session = SetupSession(case)
    solve(session)
    evaluation = evaluate_setup(case, session)
    if not evaluation["passed"]:
        raise ValueError(f"Reference failed its independent checks: {case.case_id}")
    return {
        "case_id": case.case_id, "family_id": case.family_id,
        "provenance": {
            "kind": "locally_authored_deterministic_synthetic",
            "generator": "organism.setup_reference.solve",
            "teacher_model_used": False, "human_review_claimed": False,
            "variation": variation, "seed": seed,
            "admission": "pending_frozen_experiment_contract",
        },
        "case": asdict(case), "tool_contract_sha256": contract_hash(),
        "messages": messages_from_events(case.case_id, session.events),
        "events": session.events, "evaluation": evaluation,
        "final_state": session.evaluator_snapshot(),
    }


def replay_trajectory(record: dict, case: DeviceCase) -> dict:
    if record["case"] != asdict(case) or record["tool_contract_sha256"] != contract_hash():
        raise ValueError("Trajectory fixture or tool contract changed")
    session = SetupSession(case)
    for event in record["events"]:
        observed = session.call(event["tool"], event["arguments"])
        if observed != event["result"]:
            raise ValueError("Recorded observation differs from replay")
    if record["messages"] != messages_from_events(case.case_id, session.events):
        raise ValueError("Model messages differ from the verified trajectory")
    evaluation = evaluate_setup(case, session)
    if not evaluation["passed"] or evaluation != record["evaluation"]:
        raise ValueError("Independent trajectory evaluation failed")
    if record["final_state"] != session.evaluator_snapshot():
        raise ValueError("Recorded final state differs from replay")
    return evaluation


def _write_json(path: Path, value):
    # Preserve field order: the native chat template iterates tool definitions
    # and argument objects in insertion order, as the serving provider does.
    path.write_text(json.dumps(value, indent=2) + "\n")


def build(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    groups = {"train": [], "validation": []}
    for split, variation, seed, case in development_cases():
        record = trajectory(case, variation, seed)
        replay_trajectory(record, case)
        groups[split].append(record)
    counts = {}
    for split, records in groups.items():
        with (output / f"{split}.jsonl").open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        counts[split] = {
            "trajectories": len(records),
            "assistant_decisions": sum(len(r["events"]) for r in records),
            "families": sorted({r["family_id"] for r in records}),
            "maximum_tool_calls": max(len(r["events"]) for r in records),
        }
    assert not set(counts["train"]["families"]) & set(counts["validation"]["families"])
    _write_json(output / "tools.json", TOOLS)
    manifest = {
        "dataset_id": "setup-development-v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "development_only_in_memory_simulation",
        "status": "verified_synthetic_not_yet_admitted_for_training",
        "qualification_data_present": False, "optimization_started": False,
        "schema_version": SCHEMA_VERSION, "tool_contract_sha256": contract_hash(),
        "revision_note": "Same development cases as v1; per-property descriptions match the captured live Strands schema. Supersedes v1 omission and v2 aliased-description errors.",
        "counts": counts,
        "splitting": "whole_API_family; validation is development data, never qualification",
        "duplicate_scenarios": "Some seeds share physical values; IDs do not imply independent environments.",
        "limitations": [
            "No real device transport, package payload, identity cryptography, or process activation.",
            "Deterministic reference passes every case; this suite does not establish a model advantage.",
            "No sealed qualification set is included or claimed.",
        ],
        "files": {
            path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            for path in sorted(output.iterdir()) if path.is_file()
        },
        "sources": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("setup_simulator.py", "setup_reference.py", "setup_dataset.py")
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output)["counts"], indent=2))
