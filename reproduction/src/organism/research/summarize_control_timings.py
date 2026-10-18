"""Derive native timing rows with explicit roles, counts, and measurement bounds."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def identity(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def summarize(audit_path, description_path, output):
    audit = json.loads(audit_path.read_text())
    description = json.loads(description_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "replicate-native-control-timings-v1",
        "inputs": {"audit": identity(audit_path), "description": identity(description_path)},
        "diagnostic_timing_boundary": "application RPC and polling, identical owner code for all arms",
        "readiness_timing_boundary": "owner enable to useful readiness, including transfer and activation",
        "memory_measurement_boundary": "largest reported per-application service-group MemoryPeak",
        "whole_host_memory_claimed": False,
        "attempted_lineages_per_arm": 1,
        "shared_host": True,
        "inferential_statistics_claimed": False,
        "arms": {},
    }
    names = {"fixed-policy": "Fixed", "stock-control": "Stock", "trained-merged": "Trained"}
    latex = ["% Generated descriptive timings; seconds and GiB; one trial per arm."]
    for arm in audit["arms"]:
        name = arm["arm"]
        if arm.get("not_executed"):
            result["arms"][name] = {"not_executed": True}
            continue
        by_slot = {}
        for slot in ("seed", "child", "grandchild"):
            rows = [row for row in arm["tasks"] if row["slot"] == slot]
            by_slot[slot] = {
                "reconstructed_cases": len(rows), "passed": sum(row["passed"] for row in rows),
                "median_seconds": statistics.median(row["seconds"] for row in rows) if rows else None,
            }
        peak = arm["largest_reported_application_memory_peak_bytes"]
        timings = description["arms"][name]["owner_enable_to_useful_ready_seconds"]
        value = {
            "diagnostics_by_individual": by_slot,
            "owner_enable_to_useful_ready_seconds": timings,
            "largest_reported_application_memory_peak_bytes": peak,
            "largest_reported_application_memory_peak_GiB": peak / 1024**3 if peak is not None else None,
            "complete_lineage_reconstructed": arm["complete_lineage_reconstructed"],
        }
        result["arms"][name] = value
        median = by_slot["seed"]["median_seconds"]
        fields = [names[name], "---" if median is None else f"{median:.3f}",
                  str(by_slot["seed"]["reconstructed_cases"]),
                  *["---" if slot not in timings else f"{timings[slot]:.2f}"
                    for slot in ("child", "grandchild")],
                  "---" if peak is None else f"{peak / 1024**3:.3f}"]
        latex.append(" & ".join(fields) + r"\\")
    (output / "native-timings.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "native-timing-table-rows.tex").write_text("\n".join(latex) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--description", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.audit, args.description, args.output), indent=2))
