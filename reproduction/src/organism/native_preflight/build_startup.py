"""Build the full-capsule native seed kit without copying its large payloads."""

import argparse
import base64
import gzip
import io
import json
from pathlib import Path
import tarfile

from ..capsule import verify_capsule_directory
from .common import digest, write_json
from .infrastructure import storage_template, worker_template
from .startup_common import SCHEMA, validate_catalog


SOURCE_NAMES = (
    "__init__.py", "capsule.py", "device_resources.py", "device_storage.py", "discovery_book.py", "enrollment.py", "node_context.py",
    "peer_audit.py", "peer_identity.py", "peer_transport.py", "preparation.py",
    "readiness_supervision.py", "receiver.py", "receiver_service.py", "receiver_state.py",
    "runtime_environment.py", "runtime_storage.py", "safe_archive.py", "supervision.py", "supervision_records.py",
    "systemd_supervision.py", "network_namespace.py", "network_budget.py", "network_enforcement.py",
    "start_guard.py", "ubuntu_supplemental.py",
    "native_preflight/__init__.py", "native_preflight/common.py", "native_preflight/host.py",
    "native_preflight/libraries.py", "native_preflight/verify_libraries.py",
    "native_preflight/startup_common.py", "native_preflight/startup_host.py",
    "native_preflight/startup_runner.py", "native_preflight/startup_seed.py",
)


def source_archive(workspace, target):
    hashes = {}
    with target.open("xb") as output, gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name in sorted(SOURCE_NAMES):
                path = Path(workspace) / "organism" / name
                if path.is_symlink():
                    raise ValueError("Native startup source must not be a symlink")
                raw = path.read_bytes()
                if len(raw) > 1024**2:
                    raise ValueError("Native startup source exceeds its bound")
                compile(raw, name, "exec")
                entry = tarfile.TarInfo("organism/" + name)
                entry.size, entry.mode = len(raw), 0o644
                entry.uid = entry.gid = entry.mtime = 0
                entry.uname = entry.gname = ""
                archive.addfile(entry, io.BytesIO(raw))
                hashes[name] = digest(path)
    return hashes


def build(workspace, artifact, output):
    workspace, artifact, output = Path(workspace).resolve(), Path(artifact).resolve(), Path(output).absolute()
    manifest = (artifact / "capsule/.manifest.json").read_bytes()
    trust = json.loads((artifact / "release-trust.json").read_bytes())
    keys = {name: base64.b64decode(value, validate=True) for name, value in trust["public_keys"].items()}
    envelope = verify_capsule_directory(artifact / "capsule", manifest, keys, require_trained=False)
    if envelope["body"]["model"]["trained_for_organism"] is not False:
        raise ValueError("Native seed compatibility baseline requires the declared stock release")
    output.mkdir(parents=True, exist_ok=False)
    sources = source_archive(workspace, output / "kit.tar.gz")
    catalog = {
        "schema": SCHEMA, "trained_for_organism": False, "qualification_passed": False,
        "release_id": envelope["release_id"], "manifest_base64": base64.b64encode(manifest).decode(),
        "release_keys": trust["public_keys"], "platforms": envelope["body"]["platforms"],
        "files": envelope["body"]["files"], "source_hashes": sources,
        "scope": "Owner-provisioned native seed through receiver, application and carried readiness checker",
    }
    validate_catalog(catalog)
    write_json(output / "catalog.json", catalog)
    write_json(output / "storage-template.json", storage_template())
    write_json(output / "worker-template.json", worker_template(startup=True))
    inventory = [{"path": str(artifact / "capsule" / name), "key_suffix": "inputs/capsule/" + name, **spec}
                 for name, spec in sorted(catalog["files"].items())]
    inventory += [{"path": str(output / name), "key_suffix": "inputs/" + name,
                   "bytes": (output / name).stat().st_size, "sha256": digest(output / name)}
                  for name in ("catalog.json", "kit.tar.gz")]
    write_json(output / "upload-inventory.json", {
        "scope": "Review only; no AWS resources or spending authorized by this file",
        "resources_created": False, "spending_authorized": False, "files": inventory,
        "total_upload_bytes": sum(item["bytes"] for item in inventory),
    })
    write_json(output / "build-receipt.json", {
        "schema": SCHEMA, "kit_sha256": digest(output / "kit.tar.gz"), "catalog_sha256": digest(output / "catalog.json"),
        "source_hashes": sources, "release_id": envelope["release_id"],
        "worker_template_sha256": digest(output / "worker-template.json"),
        "storage_template_sha256": digest(output / "storage-template.json"),
        "current_receiver_application_checker_integration_in_kit": True,
        "seed_is_owner_provisioned": True, "native_execution_verified": False,
        "autonomous_reproduction_verified": False, "qualification_passed": False,
    })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(Path(__file__).resolve().parents[2], args.artifact, args.output))


if __name__ == "__main__":
    main()
