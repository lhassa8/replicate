"""Prepare comparison input identities and carried application; no cloud calls."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from organism.capsule import sign_manifest
from organism.enrollment import public_bytes, key_id
from organism.research.heldout_study import identity, write


ROOT = Path(__file__).resolve().parents[2]


def archive(files, target):
    with target.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for name, content in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(content), 0o644, 0
                tar.addfile(info, io.BytesIO(content))


def build(output, *, experiment_id="replicate-paper-controls-20261017"):
    output.mkdir(parents=True, exist_ok=False)
    capsule = ROOT / "organism/evidence/first-lineage-native-execution-20261017-06/trained-capsule/capsule"
    original = json.loads((capsule / ".manifest.json").read_text())
    model_pair = json.loads((ROOT / "organism/evidence/paper-heldout-20261017-03/agent-freeze.json").read_text())["models"]
    with tarfile.open(capsule / original["body"]["agent"]) as tar:
        agent = {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}
    additions = (
        "organism/research/__init__.py", "organism/research/control_policy.py",
        "organism/research/control_entry.py", "organism/equipment_baseline.py",
        "organism/research/fixed_lifecycle_candidate.py",
    )
    for name in additions:
        agent[name] = (ROOT / name).read_bytes()
    archive(agent, output / "agent.tar.gz")
    owner = {str(p.relative_to(ROOT)): p.read_bytes() for p in (ROOT / "organism").glob("*.py")}
    for name in ("__init__", "control_worker", "control_policy", "control_entry",
                 "fixed_lifecycle_candidate"):
        p = ROOT / "organism/research" / (name + ".py")
        owner[str(p.relative_to(ROOT))] = p.read_bytes()
    archive(owner, output / "owner-source.tar.gz")
    key = Ed25519PrivateKey.generate()
    public = public_bytes(key.public_key())
    import base64
    write(output / "release-trust.json", {key_id(public): base64.b64encode(public).decode()})
    shared, uploads, arms = {}, {}, {}
    for name, expected in original["body"]["files"].items():
        if name == original["body"]["model"]["path"]:
            continue
        source = output / "agent.tar.gz" if name == original["body"]["agent"] else capsule / name
        observed = identity(source)
        if name != original["body"]["agent"] and observed != expected:
            raise ValueError("Original carried runtime changed: " + name)
        asset = "shared/" + name
        shared[name] = {"asset": asset, **observed}
        uploads[asset] = {"path": str(source), **observed}
    for arm in ("fixed-policy", "stock-control", "trained-merged"):
        selected = model_pair["stock-control" if arm == "stock-control" else "trained-merged"]
        source = Path(selected["path"])
        if identity(source) != {k: selected[k] for k in ("bytes", "sha256")}:
            raise ValueError("Matched model identity changed")
        asset = "models/" + ("stock.gguf" if arm == "stock-control" else "trained.gguf")
        uploads[asset] = selected
        body = deepcopy(original["body"])
        body["experiment_id"] = experiment_id
        body["files"][body["agent"]] = identity(output / "agent.tar.gz")
        body["files"][body["model"]["path"]] = {k: selected[k] for k in ("bytes", "sha256")}
        body["model"]["trained_for_organism"] = arm != "stock-control"
        signed = sign_manifest(body, key, require_trained=False)
        path = output / (arm + "-manifest.json")
        path.write_bytes(signed)
        uploads[path.name] = {"path": str(path), **identity(path)}
        arms[arm] = {"model_asset": asset, "manifest_asset": path.name,
                     "release_id": json.loads(signed)["release_id"], "model_executed": arm != "fixed-policy"}
    old_cases = json.loads((ROOT / "organism/evidence/first-lineage-temperature-evaluation-20261016-05/batch-01/plan.json").read_text())["cases"]
    cases = [next(row for row in old_cases if row["variation"] == variation)
             for variation in ("nominal", "fault", "stale", "invalid", "missing")]
    write(output / "smoke-cases.json", {"cases": cases, "development_cases": True,
                                      "heldout_cases_included": False})
    for name in ("owner-source.tar.gz", "release-trust.json", "smoke-cases.json"):
        uploads[name] = {"path": str(output / name), **identity(output / name)}
    host = ROOT / "organism/research/control_host.py"
    uploads["host.py"] = {"path": str(host), **identity(host)}
    assets = {name: {k: row[k] for k in ("bytes", "sha256")} for name, row in uploads.items()}
    write(output / "assets.json", {
        "schema": "replicate-paper-control-inputs-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "files": assets, "shared": shared, "arms": arms,
        "source_files": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                         for name, raw in owner.items()},
        "scope": "Three policy arms on one native Linux host, separate application users and 8 GiB service limits",
        "order": ["fixed-policy", "stock-control", "trained-merged"],
        "original_separate_machine_qualification": False,
    })
    uploads["assets.json"] = {"path": str(output / "assets.json"), **identity(output / "assets.json")}
    write(output / "upload-review.json", {"files": uploads, "total_bytes": sum(row["bytes"] for row in uploads.values())})
    print(json.dumps({"files": len(uploads), "bytes": sum(row["bytes"] for row in uploads.values()),
                      "assets": identity(output / "assets.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default="replicate-paper-controls-20261017")
    args = parser.parse_args()
    build(args.output.resolve(), experiment_id=args.experiment_id)
