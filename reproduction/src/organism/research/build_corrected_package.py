"""Assemble the corrected comparison while retaining the previous experiment."""

import argparse
import json
from pathlib import Path
import shutil

from organism.research.build_reproduction_package import ROOT, archive, prepare


def build(output):
    execution = ROOT / "organism/evidence/paper-controls-execution-20261018-01"
    preparation = ROOT / "organism/evidence/paper-controls-preparation-20261018-01"
    closure = json.loads((execution / "CLOSED.json").read_text())
    if not closure["cleanup_verified"] or not closure["temporary_storage_removed"]:
        raise ValueError("The corrected experiment is not closed")
    controls = execution / "returned/results"
    for name in ("independent-control-audit.json", "control-description.json"):
        if not (controls / name).is_file():
            raise ValueError("Missing corrected outcome reconstruction: " + name)
    prepare(output, controls, preparation)
    shutil.copyfile(ROOT / "docs/replicate-reproduction-README-20261018.md", output / "README.md")
    shutil.copyfile(ROOT / "docs/replicate-corrected-native-plan-20261018.md",
                    output / "docs/replicate-corrected-native-plan-20261018.md")

    history = output / "evidence/history"
    previous = ROOT / "organism/evidence/paper-controls-execution-20261017-02/returned/results"
    shutil.copytree(previous, history / "native-controls-20261017",
                    ignore=shutil.ignore_patterns("__pycache__", "*.sock"))
    shutil.copyfile(preparation / "smoke-cases.json",
                    history / "native-controls-20261017/smoke-cases.json")
    previous_preparation = ROOT / "organism/evidence/paper-controls-preparation-20261017-04"
    frozen_previous = history / "native-controls-20261017-inputs"
    frozen_previous.mkdir()
    for name in ("agent.tar.gz", "owner-source.tar.gz", "assets.json",
                 "fixed-policy-manifest.json", "stock-control-manifest.json",
                 "trained-merged-manifest.json", "release-trust.json",
                 "worker-template.json", "storage-template.json"):
        shutil.copyfile(previous_preparation / name, frozen_previous / name)

    frozen = output / "evidence/control-protocol-20261018"
    frozen.mkdir()
    for name in ("protocol.json", "tests.log"):
        shutil.copyfile(preparation / name, frozen / name)
    shutil.copytree(preparation / "source", frozen / "source")
    operations = output / "evidence/control-allocation-20261018"
    operations.mkdir()
    for name in ("ledger.json", "resource-admission.json", "input-admission.json",
                 "upload-receipts.json", "permit-publication.json",
                 "output-preservation.json", "host-output-receipt.json",
                 "cleanup-readback.json", "storage-cleanup-readback.json", "CLOSED.json"):
        shutil.copyfile(execution / name, operations / name)
    correction = ROOT / "organism/evidence/paper-controls-correction-20261018-01"
    retained_correction = history / "lifecycle-correction-20261018"
    retained_correction.mkdir()
    for name in ("before-identities.json", "description-checks.json",
                 "frozen-comparison.json", "paper-before.tex"):
        shutil.copyfile(correction / name, retained_correction / name)
    shutil.copytree(correction / "before", retained_correction / "before")
    shutil.copyfile(ROOT / "output/replicate-reproduction-20261017-receipt.json",
                    history / "previous-package-receipt.json")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "archive"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.output.resolve()) if args.action == "prepare"
          else archive(args.output.resolve()))
