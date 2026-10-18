"""Repeat the frozen controls offline inside a fresh, disposable Linux guest."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def run(package, work):
    if os.geteuid() != 0 or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Use root in a fresh x86-64 Linux guest with systemd")
    if Path("/var/lib/replicate-paper").exists():
        raise ValueError("The comparison's guest state already exists; use a fresh guest")
    assets = package / "assets"
    catalog = json.loads((assets / "assets.json").read_text())
    for relative, expected in catalog["files"].items():
        path = assets / relative
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("Frozen control input differs: " + relative)
    work.mkdir(parents=True, exist_ok=False, mode=0o700)
    for arm, specification in catalog["arms"].items():
        capsule = work / "inputs/capsules" / arm
        capsule.mkdir(parents=True)
        manifest_path = assets / specification["manifest_asset"]
        manifest = json.loads(manifest_path.read_text())
        shutil.copyfile(manifest_path, capsule / ".manifest.json")
        for name in manifest["body"]["files"]:
            source = (assets / specification["model_asset"] if name == manifest["body"]["model"]["path"]
                      else assets / catalog["shared"][name]["asset"])
            target = capsule / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for name in ("release-trust.json", "smoke-cases.json"):
        shutil.copyfile(assets / name, work / "inputs" / name)
    for name, archive in (
        ("owner", "owner-source.tar.gz"),
        ("python", "shared/linux-x86_64/python.tar.gz"),
        ("dependencies", "shared/linux-x86_64/dependencies.tar.gz"),
    ):
        target = work / name
        target.mkdir()
        with tarfile.open(assets / archive) as stream:
            stream.extractall(target, filter="data")
    Path("/var/lib/replicate-paper").mkdir(mode=0o755)
    environment = {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/nonexistent", "LC_ALL": "C.UTF-8",
        "PYTHONPATH": str(work / "owner") + ":" + str(work / "dependencies/site-packages"),
        "LD_LIBRARY_PATH": str(work / "python/python/lib") + ":" + str(work / "dependencies/supplemental/lib"),
        "OTEL_SDK_DISABLED": "true", "AWS_EC2_METADATA_DISABLED": "true",
    }
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "inputs_verified": True,
              "failure": None, "scope": "New offline execution of published cohosted controls",
              "worker_returncode": None, "cloud_admission_claimed": False}
    try:
        report["worker_returncode"] = subprocess.run(
            [str(work / "python/python/bin/python3.12"), "-m", "organism.research.control_worker",
             "--inputs", str(work / "inputs"), "--output", str(work / "results")],
            env=environment, timeout=3 * 3600, check=False).returncode
        if report["worker_returncode"]:
            report["failure"] = "Worker returned " + str(report["worker_returncode"])
        return report["worker_returncode"]
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        subprocess.run(["systemctl", "stop", "replicate-paper-*.service"], check=False)
        (work / "results").mkdir(exist_ok=True)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        (work / "results/host-report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.package.resolve(), args.work.resolve()))
