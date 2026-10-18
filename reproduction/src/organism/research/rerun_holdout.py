"""Repeat the published synthetic cases on a Mac using packaged frozen inputs.

This is a replication attempt on known cases, not a new held-out evaluation.
"""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil

from organism.research import heldout_study as study


def run(package, output):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("The diagnostic replication entry point requires an Apple Silicon Mac")
    original = package / "evidence/holdout"
    frozen, protocol = study.verify_freeze(original)
    for name, expected in frozen["versions"].items():
        if version(name) != expected:
            raise ValueError("Dependency version differs: " + name)
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in frozen["source_files"].items():
        source = study.ROOT / name
        if study.identity(source) != expected:
            raise ValueError("Frozen agent or scorer source changed: " + name)
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    revised = deepcopy(frozen)
    for name, model in revised["models"].items():
        path = package / "assets/models" / ("stock.gguf" if name == "stock-control" else "trained.gguf")
        if study.identity(path) != {key: model[key] for key in ("bytes", "sha256")}:
            raise ValueError("Packaged model identity differs")
        model["path"] = str(path.resolve())
    for name, expected in frozen["runtime_inputs"].items():
        if study.identity(package / "assets/local-runtime" / name) != expected:
            raise ValueError("Packaged Mac runtime differs: " + name)
    binary_suffix = next(name for name in frozen["runtime_inputs"] if name.endswith("/llama-server"))
    revised["binary_path"] = str((package / "assets/local-runtime" / binary_suffix).resolve())
    revised["replication_of"] = study.identity(original / "agent-freeze.json")
    revised["replication_started_at"] = datetime.now(timezone.utc).isoformat()
    revised["scope"] = "New execution of published synthetic cases; not a new blind holdout"
    study.write(output / "agent-freeze.json", revised)
    shutil.copyfile(original / "cases.json", output / "cases.json")
    changed_protocol = deepcopy(protocol)
    changed_protocol["agent_freeze"] = study.identity(output / "agent-freeze.json")
    changed_protocol["replication_of"] = study.identity(original / "protocol.json")
    study.write(output / "protocol.json", changed_protocol)
    asyncio.run(study.run(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.package.resolve(), args.output.resolve())
