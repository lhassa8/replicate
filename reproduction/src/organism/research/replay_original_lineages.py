"""Replay original device/link consistency checks on their exact accessed files.

This reuses the unchanged original auditors. It does not rerun cloud admission,
create hardware, or observe a newly live lineage.
"""

import argparse
import base64
import json
from pathlib import Path

from organism.capsule import verify_manifest
from organism.native_preflight import audit_lineage as original
from organism.native_preflight import audit_lineage_transport as transport
from organism.native_preflight import first_lineage_tasks, first_lineage_profile
from organism.native_preflight.audit_startup import require
from organism.research.heldout_study import identity, ROOT


def replay(evidence, output):
    freeze = json.loads((evidence / "holdout/agent-freeze.json").read_text())
    for name, expected in freeze["source_files"].items():
        require(identity(ROOT / name) == expected, "Frozen original auditor source differs: " + name)
    directory = evidence / "original-lineages"
    release = json.loads((directory / "release-audit-inputs.json").read_text())
    rows = []
    for trial in ("08", "09", "10"):
        root = directory / trial
        saved = json.loads((root / "independent-lineage-audit.json").read_text())
        for slot, files in saved["read_file_sha256"].items():
            for name, sha256 in files.items():
                require(identity(root / slot / name)["sha256"] == sha256,
                        "Original evidence identity changed: " + trial + "/" + slot + "/" + name)
        catalog = json.loads((root / "seed/input-catalog.json").read_text())
        verify_manifest(base64.b64decode(catalog["manifest_base64"]),
                        {k: base64.b64decode(v) for k, v in catalog["release_keys"].items()})
        devices = {slot: original.inspect_device(root / slot, slot, catalog, release)
                   for slot in ("seed", "child", "grandchild")}
        require({slot: row[2] for slot, row in devices.items()} == saved["devices"],
                "Original per-device result differs: " + trial)
        links = []
        for parent, child in (("seed", "child"), ("child", "grandchild")):
            links.append(transport.link(
                devices[parent][0], devices[parent][1], devices[child][0], devices[child][1],
                catalog, child_enabled_at=devices[child][2]["controls"][0]["observed_at"]))
        require(links == saved["links"], "Original birth-link result differs: " + trial)
        rows.append({"trial": trial, "devices_reconstructed": 3, "births_reconstructed": len(links),
                     "diagnostics_reconstructed": sum(d[2]["temperature_tasks"]["successes"]
                                                      for d in devices.values()),
                     "endurance_diagnostics_reconstructed": sum(
                         d[2]["endurance"]["tasks"] if d[2]["endurance"] else 0 for d in devices.values()),
                     "exact_original_results_match": True})
    native = evidence / "original-native"
    saved_native = json.loads((native / "independent-native-audit.json").read_text())
    for name, sha256 in saved_native["read_file_sha256"].items():
        require(identity(native / "returned" / name)["sha256"] == sha256,
                "Original native evidence identity changed: " + name)
    drive = json.loads((native / "returned/drive-receipt.json").read_text())
    birth = drive["cold_starts"][-1]["birth"]
    binding = {key: birth[key] for key in ("device_id", "individual_id", "release_id")}
    binding["start_token"] = birth["process_token"]
    native_tasks = first_lineage_tasks.audit(
        native / "returned/temperature-tasks",
        cases_sha256=first_lineage_profile.CASES_SHA256, binding=binding)
    require(native_tasks == saved_native["temperature_tasks"],
            "Original single-device diagnostics differ")
    result = {"trials": rows,
              "original_native_diagnostics_reconstructed": native_tasks["successes"],
              "total_diagnostics_reconstructed": native_tasks["successes"] + sum(
                  row["diagnostics_reconstructed"] + row["endurance_diagnostics_reconstructed"]
                  for row in rows),
              "single_device_cold_start_admission_replayed": False,
              "new_native_execution": False,
              "cloud_admission_replayed": False,
              "scope": "Exact original per-device and parent-child artifact consistency"}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.evidence.resolve(), args.output.resolve())))
