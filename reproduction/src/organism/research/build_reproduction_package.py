"""Assemble an explicit, inspectable release candidate; never publish it."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import distributions
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[2]


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepare_originals(output):
    originals = output / "evidence/original-lineages"
    for trial in ("08", "09", "10"):
        source = ROOT / ("organism/evidence/first-lineage-trial-20261017-" + trial)
        destination = originals / trial
        destination.mkdir(parents=True)
        audit = json.loads((source / "independent-lineage-audit.json").read_text())
        for slot, read_files in audit["read_file_sha256"].items():
            for name, expected in read_files.items():
                path = source / "returned" / slot / name
                if identity(path)["sha256"] != expected:
                    raise ValueError("Original admitted evidence changed: " + str(path))
                target = destination / slot / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            # The nested diagnostic auditor opens its own recorded inputs rather
            # than going through the outer auditor's Evidence reader.
            shutil.copytree(source / "returned" / slot / "observe/temperature-tasks",
                            destination / slot / "observe/temperature-tasks", dirs_exist_ok=True)
            if (source / "returned" / slot / "endurance").exists():
                shutil.copytree(source / "returned" / slot / "endurance",
                                destination / slot / "endurance", dirs_exist_ok=True)
        for name in ("independent-lineage-audit.json", "external-lineage-admission.json",
                     "CLOSED.json", "independent-endurance-audit.json"):
            if (source / name).exists():
                shutil.copyfile(source / name, destination / name)
    # The original source and receipts remain unchanged. This small descriptor
    # records the exact release inputs needed by the original device auditors.
    import base64
    from organism.native_preflight.audit_first_lineage_profile import archive_member_hash
    catalog = json.loads((originals / "08/seed/input-catalog.json").read_text())
    raw = base64.b64decode(catalog["manifest_base64"], validate=True)
    body = json.loads(raw)["body"]
    capsule = ROOT / "organism/evidence/first-lineage-native-execution-20261017-06/trained-capsule/capsule"
    release = {
        "release_id": catalog["release_id"], "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "checker_sha256": archive_member_hash(capsule / body["self_test"], "readiness.py"),
        "runtime_sha256": archive_member_hash(
            capsule / body["platforms"]["linux-x86_64"]["inference"], "llama-b9850/llama-server"),
        "model_sha256": body["files"][body["model"]["path"]]["sha256"],
        "model_bytes": body["files"][body["model"]["path"]]["bytes"],
    }
    (originals / "release-audit-inputs.json").write_text(json.dumps(release, indent=2) + "\n")
    native_source = ROOT / "organism/evidence/first-lineage-native-execution-20261017-07"
    native = output / "evidence/original-native"
    native.mkdir(parents=True)
    saved = json.loads((native_source / "independent-native-audit.json").read_text())
    for name, expected in saved["read_file_sha256"].items():
        source = native_source / "returned" / name
        if identity(source)["sha256"] != expected:
            raise ValueError("Original native evidence changed: " + name)
        target = native / "returned" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copytree(native_source / "returned/temperature-tasks",
                    native / "returned/temperature-tasks", dirs_exist_ok=True)
    for name in ("independent-native-audit.json", "external-native-admission.json", "CLOSED.json"):
        shutil.copyfile(native_source / name, native / name)


def prepare(output, controls, preparation):
    """Small review tree plus an explicit inventory of large local assets."""
    heldout = ROOT / "organism/evidence/paper-heldout-20261017-03"
    mutations = ROOT / "organism/evidence/paper-auditor-mutations-20261017-01"
    if not (heldout / "execution-complete.json").exists():
        raise ValueError("The frozen holdout is not complete")
    if not (controls / "host-report.json").exists():
        raise ValueError("The native control attempt has not returned")
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("organism", "organism/native_preflight", "organism/research"):
        for path in sorted((ROOT / directory).glob("*.py")):
            # The cloud transfer helpers are private orchestration conveniences,
            # not needed for offline scientific replay.
            if path.name in ("transfer_controls.py",):
                continue
            target = output / "src" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    for source, name in ((heldout, "holdout"), (mutations, "mutations"), (controls, "controls")):
        shutil.copytree(source, output / "evidence" / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.sock"))
    history = output / "evidence/history"
    history.mkdir()
    for suffix in ("01", "02"):
        source = ROOT / ("organism/evidence/paper-heldout-20261017-" + suffix)
        if source.exists():
            shutil.copytree(source, history / ("holdout-" + suffix),
                            ignore=shutil.ignore_patterns("__pycache__", "*.sock"))
    original = ROOT / "organism/evidence/first-lineage-final-report-20261017-01"
    for name in ("lineage-measurements.json", "completed-runtime-gates.json"):
        if (original / name).exists():
            shutil.copyfile(original / name, history / name)
    prepare_originals(output)
    for name in ("unsuccessful-lineage-index.json", "native-history-index.json"):
        source = ROOT / "organism/evidence/first-lineage-milestone-20261016-01" / name
        shutil.copyfile(source, history / name)
    docs = output / "docs"
    docs.mkdir()
    for source in ("organism/FIRST_LINEAGE_OPERATIONS.md",
                   "docs/self-replicating-organism-plan-20261013.md",
                   "docs/replicate-first-lineage-plan-20261016.md",
                   "docs/replicate-paper-improvement-plan-20261017.md",
                   "docs/replicate-fixed-lifecycle-followup-20261017.md"):
        shutil.copyfile(ROOT / source, docs / Path(source).name)
    shutil.copyfile(ROOT / "docs/replicate-reproduction-README-20261017.md", output / "README.md")
    shutil.copyfile(ROOT / "output/pdf/replicate-whitepaper-20261017.tex", output / "paper.tex")
    shutil.copyfile(preparation / "smoke-cases.json", output / "evidence/controls/smoke-cases.json")
    tests = output / "src/organism/tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")
    for name in ("test_paper_lifecycle_candidate.py", "test_reproduction.py",
                 "test_receiver_state.py", "test_application.py", "test_preparation.py",
                 "test_supervision.py"):
        shutil.copyfile(ROOT / "organism/tests" / name, tests / name)
    for suffix in ("01", "02"):
        source = ROOT / ("organism/evidence/paper-controls-execution-20261017-" + suffix)
        destination = history / ("native-control-allocation-" + suffix)
        destination.mkdir()
        for name in ("ledger.json", "cleanup-readback.json", "failure-console-readback.json",
                     "resource-admission.json", "storage-prefix-cleanup-readback.json",
                     "output-preservation.json"):
            if (source / name).exists():
                shutil.copyfile(source / name, destination / name)
    freeze = json.loads((heldout / "agent-freeze.json").read_text())
    (output / "requirements.txt").write_text(
        "# Python 3.12; top-level versions from the frozen experiment.\n" +
        "".join(name + "==" + ver + "\n" for name, ver in freeze["versions"].items()))
    (output / "requirements-lock.txt").write_text(
        "# Installed Python dependency versions in the author's replay environment.\n" +
        "".join(name + "==" + ver + "\n" for name, ver in sorted(
            (dist.metadata["Name"], dist.version) for dist in distributions())))
    review = json.loads((preparation / "upload-review.json").read_text())
    for name, entry in review["files"].items():
        if identity(Path(entry["path"])) != {key: entry[key] for key in ("bytes", "sha256")}:
            raise ValueError("Selected frozen control input changed: " + name)
    assets = {("assets/" + name): Path(entry["path"]) for name, entry in review["files"].items()}
    for name in freeze["runtime_inputs"]:
        assets["assets/local-runtime/" + name] = ROOT / name
    capsule = ROOT / "organism/evidence/first-lineage-native-execution-20261017-06/trained-capsule"
    assets["assets/original-release/manifest.json"] = capsule / "capsule/.manifest.json"
    assets["assets/original-release/agent.tar.gz"] = capsule / "capsule/common/agent.tar.gz"
    assets["assets/original-release/release-trust.json"] = capsule / "release-trust.json"
    # The private assembly inventory stays outside the distributable tree.
    index = {name: {"path": str(path), **identity(path)} for name, path in assets.items()}
    (output.parent / (output.name + "-assembly.json")).write_text(json.dumps(index, indent=2) + "\n")
    return output


def archive(output):
    assembly = output.parent / (output.name + "-assembly.json")
    assets = json.loads(assembly.read_text())
    manifest = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            if path.is_symlink():
                raise ValueError("Unexpected symlink in release tree: " + str(path))
            manifest[str(path.relative_to(output))] = identity(path)
    for name, expected in assets.items():
        actual = identity(Path(expected["path"]))
        if actual != {key: expected[key] for key in ("bytes", "sha256")}:
            raise ValueError("Large asset changed before packaging: " + name)
        manifest[name] = actual
    (output / "MANIFEST.json").write_text(json.dumps({
        "schema": "replicate-paper-reproduction-package-v1",
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "publicly_released": False, "external_replication": False,
        "files": manifest,
    }, indent=2) + "\n")
    destination = output.parent / (output.name + ".tar")
    with tarfile.open(destination, "x", dereference=True) as tar:
        tar.add(output, arcname=output.name)
        for name, expected in assets.items():
            tar.add(expected["path"], arcname=output.name + "/" + name, recursive=False)
    receipt = {"archive": destination.name, **identity(destination),
               "files": len(manifest), "publicly_released": False}
    (output.parent / (output.name + "-receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "archive"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--preparation", type=Path)
    args = parser.parse_args()
    result = (prepare(args.output.resolve(), args.controls.resolve(), args.preparation.resolve())
              if args.action == "prepare" else archive(args.output.resolve()))
    print(result)
