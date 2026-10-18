"""Build a deterministic source kit and a verified local upload inventory.

No AWS operations, credential reads, model copies, or deployment side effects.
"""

import argparse
import gzip
import io
import json
from pathlib import Path
import tarfile

from .common import check_file, digest, validate_catalog, write_json
from .infrastructure import storage_template, worker_template


SOURCE_NAMES = (
    "__init__.py", "diagnostics.py", "probe_local.py", "probe_setup.py", "audit_setup.py",
    "setup_dataset.py", "setup_reference.py", "setup_simulator.py", "setup_tools.py",
    "native_preflight/__init__.py", "native_preflight/common.py",
    "native_preflight/libraries.py", "native_preflight/host.py", "native_preflight/runner.py",
    "native_preflight/verify_libraries.py", "ubuntu_supplemental.py",
)


def source_archive(workspace, target):
    source_hashes = {}
    with target.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for name in sorted(SOURCE_NAMES):
                source = workspace / "organism" / name
                if source.is_symlink():
                    raise ValueError("Source kit cannot contain symlinks")
                body = source.read_bytes()
                item = tarfile.TarInfo("organism/" + name)
                item.mode, item.size, item.mtime = 0o644, len(body), 0
                archive.addfile(item, io.BytesIO(body))
                source_hashes[name] = digest(source)
    return source_hashes


def build(workspace, target):
    workspace, target = Path(workspace), Path(target)
    target.mkdir(parents=True, exist_ok=False)
    evidence = workspace / "organism/evidence/runtime-sources-20261014-01"
    downloads = json.loads((evidence / "download-receipt.json").read_text())["files"]
    dependencies = workspace / "organism/artifacts/supplemental-development-20261014-01"
    package_audit = json.loads((dependencies / "receipt.json").read_text())
    files, inventory = [], []
    for specification in downloads:
        name = specification["name"]
        path = workspace / "organism/vendor/downloads" / name
        check_file(path, specification)
        files.append({key: specification[key] for key in ("name", "bytes", "sha256")})
        inventory.append({"path": str(path), "key_suffix": "inputs/" + name})
    for entry in package_audit["results"]:
        name = entry["archive"]
        path = dependencies / name
        specification = {"name": name, "bytes": entry["bytes"], "sha256": entry["sha256"]}
        check_file(path, specification)
        files.append(specification)
        inventory.append({"path": str(path), "key_suffix": "inputs/" + name})
    model_path = workspace / "models/granite-3b/granite-base-Q4_K_M.gguf"
    # The retained manifest also describes an unrelated adapter. Admit only the
    # explicitly selected base file; this kit never copies or loads the adapter.
    model_spec = next(entry for entry in json.loads((model_path.parent / "manifest.json").read_text())["files"]
                      if entry["name"] == model_path.name)
    check_file(model_path, model_spec)
    files.append(model_spec)
    inventory.append({"path": str(model_path), "key_suffix": "inputs/" + model_path.name})
    sources = source_archive(workspace, target / "kit.tar.gz")
    catalog = {
        "schema": "organism-native-preflight-v2", "qualification_passed": False,
        "model": model_path.name, "trained_for_organism": False,
        "supplemental_libraries_carried": True,
        "publisher_signature_verification_required_before_execution": True,
        "source_hashes": sources, "files": sorted(files, key=lambda entry: entry["name"]),
        "platforms": {
            "x86_64": {
                "python": "cpython-3.12.14+20260901-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
                "inference": "llama-b9850-bin-ubuntu-x64.tar.gz",
                "dependencies": "dependencies-linux-x86_64.tar.gz"},
            "arm64": {
                "python": "cpython-3.12.14+20260901-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz",
                "inference": "llama-b9850-bin-ubuntu-arm64.tar.gz",
                "dependencies": "dependencies-linux-arm64.tar.gz"},
        },
    }
    validate_catalog(catalog)
    write_json(target / "catalog.json", catalog)
    write_json(target / "storage-template.json", storage_template())
    write_json(target / "worker-template.json", worker_template())
    inventory += [{"path": str(target / name), "key_suffix": "inputs/" + name} for name in ("catalog.json", "kit.tar.gz")]
    for entry in inventory:
        path = Path(entry["path"])
        entry.update({"bytes": path.stat().st_size, "sha256": digest(path)})
    write_json(target / "upload-inventory.json", {
        "scope": "Review only; uploading and launching require explicit cloud-spending approval",
        "resources_created": False, "spending_authorized": False, "files": inventory,
        "total_upload_bytes": sum(entry["bytes"] for entry in inventory),
    })
    write_json(target / "build-receipt.json", {
        "kit_sha256": digest(target / "kit.tar.gz"), "catalog_sha256": digest(target / "catalog.json"),
        "source_hashes": sources, "qualification_passed": False, "native_execution_verified": False,
        "worker_template_sha256": digest(target / "worker-template.json"),
        "storage_template_sha256": digest(target / "storage-template.json"),
    })
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(Path(__file__).resolve().parents[2], args.output.resolve()))
