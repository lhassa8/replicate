"""Build the complete owner-host kit from an admitted trained capsule/export."""

import argparse
import gzip
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from .build_startup import SOURCE_NAMES as LEGACY_SOURCES
from .common import digest, write_json
from .infrastructure import storage_template
from .profile_infrastructure import selected_architectures, selected_x86_instance, worker_template
from .common import ARCHES
from .trained_inputs import prepare as prepare_candidate
from . import first_lineage_profile


SOURCE_NAMES = tuple(sorted(set(LEGACY_SOURCES) | {
    "training_inputs.py", "quantization_precision.py", "process_storage.py",
    "native_preflight/trained_inputs.py", "native_preflight/device_capacity.py",
    "native_preflight/receiver_unit.py", "native_preflight/receiver_entry.py",
    "native_preflight/profile_seed.py", "native_preflight/profile_setup.py",
    "native_preflight/profile_permit.py", "native_preflight/profile_entry.py",
    "native_preflight/profile_host.py",
    "native_preflight/profile_census.py", "native_preflight/freeze_guard.py",
    "native_preflight/first_lineage_profile.py",
    "native_preflight/owner_mount_namespace.py",
}))
FIRST_LINEAGE_SOURCES = tuple(sorted(set(SOURCE_NAMES) | {
    "audit_application.py", "audit_equipment_diagnostics.py", "audit_trained_tasks.py",
    "equipment_development.py", "first_lineage_cases.py",
    "native_preflight/first_lineage_tasks.py", "native_preflight/first_lineage_seed.py",
}))


def source_archive(workspace, target, *, first_lineage=False):
    sources = {}
    with Path(target).open("xb") as output, gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name in FIRST_LINEAGE_SOURCES if first_lineage else SOURCE_NAMES:
                path = Path(workspace) / "organism" / name
                if path.is_symlink():
                    raise ValueError("Native owner source cannot be a link")
                raw = path.read_bytes()
                if len(raw) > 1024**2:
                    raise ValueError("Native owner source exceeds its individual bound")
                compile(raw, name, "exec")
                entry = tarfile.TarInfo("organism/" + name)
                entry.size, entry.mode = len(raw), 0o644
                entry.uid = entry.gid = entry.mtime = 0
                entry.uname = entry.gname = ""
                archive.addfile(entry, io.BytesIO(raw))
                sources[name] = {"bytes": len(raw), "sha256": digest(path)}
    if len(sources) > 100 or sum(item["bytes"] for item in sources.values()) > 8 * 1024**2:
        raise ValueError("Native kit exceeds the reviewed bootstrap extraction bound")
    return sources


def build(workspace, output, *, artifact, export_directory, admission_path, training_recipe,
          architectures=ARCHES, cases=None, x86_instance_type=None):
    workspace, output = Path(workspace).resolve(), Path(output).absolute()
    architectures = selected_architectures(architectures)
    selected_x86_instance(x86_instance_type, first_lineage=cases is not None)
    if cases is not None:
        if architectures != ("x86_64",):
            raise ValueError("First-lineage native kit requires one x86-64 device")
        first_lineage_profile.validate_cases(cases)
    # The actual export is reaudited before candidate publication. No synthetic
    # catalog, family name or unlinked training claim can replace this call.
    if output.exists() or output.is_symlink():
        raise FileExistsError("Native launch kit output must be new")
    with tempfile.TemporaryDirectory(prefix="organism-native-candidate-") as temporary:
        candidate = Path(temporary) / "candidate"
        prepare_candidate(
            workspace, candidate, artifact=artifact, export_directory=export_directory,
            admission_path=admission_path, training_recipe=training_recipe)
        output.mkdir(parents=True, exist_ok=False)
        shutil.copytree(candidate, output / "candidate")
    catalog_path = output / "candidate/trained-catalog.json"
    catalog = json.loads(catalog_path.read_bytes())
    (output / "catalog.json").write_bytes(catalog_path.read_bytes())
    sources = source_archive(workspace, output / "kit.tar.gz", first_lineage=cases is not None)
    write_json(output / "worker-template.json", worker_template(
        architectures=architectures, first_lineage=cases is not None,
        x86_instance_type=x86_instance_type))
    write_json(output / "storage-template.json", storage_template())
    inventory = [
        {"path": str(Path(artifact).resolve() / "capsule" / name), "key_suffix": "inputs/capsule/" + name, **item}
        for name, item in sorted(catalog["files"].items())]
    owner_inputs = ["catalog.json", "kit.tar.gz"]
    if cases is not None:
        shutil.copyfile(cases, output / first_lineage_profile.CASES_NAME)
        first_lineage_profile.validate_cases(output / first_lineage_profile.CASES_NAME)
        owner_inputs.append(first_lineage_profile.CASES_NAME)
    inventory += [{"path": str(output / name), "key_suffix": "inputs/" + name,
                   "bytes": (output / name).stat().st_size, "sha256": digest(output / name)}
                  for name in owner_inputs]
    write_json(output / "upload-inventory.json", {"files": inventory,
                                                 "total_upload_bytes": sum(item["bytes"] for item in inventory)})
    receipt = {
        "schema": "organism-native-profile-kit-v1", "sources": sources,
        "kit_sha256": digest(output / "kit.tar.gz"), "catalog_sha256": digest(output / "catalog.json"),
        "worker_template_sha256": digest(output / "worker-template.json"),
        "storage_template_sha256": digest(output / "storage-template.json"),
        "release_id": catalog["release_id"], "whole_host_storage_max_gib": 32,
        "independent_instance_permits_required": True, "seed_is_owner_provisioned": True,
        "native_execution_verified": False, "resources_created": False, "qualification_passed": False,
    }
    if architectures != ARCHES:
        receipt.update(schema="organism-native-profile-kit-v2", architectures=list(architectures))
    if cases is not None:
        receipt.update(schema="organism-native-profile-kit-v3",
                       first_lineage=first_lineage_profile.descriptor())
    if x86_instance_type is not None:
        receipt["x86_instance_type"] = x86_instance_type
    write_json(output / "build-receipt.json", receipt)
    return output


def main():
    parser = argparse.ArgumentParser()
    for name in ("artifact", "export-directory", "admission-path", "training-recipe", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--x86-only", action="store_true",
                        help="Prepare one x86-64 device with the same per-device controls")
    parser.add_argument("--cases", type=Path,
                        help="Opt into three cold starts and the exact frozen temperature cases")
    parser.add_argument("--x86-instance-type", choices=("c7i.xlarge", "c7a.xlarge"),
                        help="Explicit 8 GiB four-vCPU choice for the first-lineage profile")
    args = parser.parse_args()
    print(build(Path(__file__).resolve().parents[2], args.output, artifact=args.artifact,
                export_directory=args.export_directory, admission_path=args.admission_path,
                training_recipe=args.training_recipe,
                architectures=("x86_64",) if args.x86_only else ARCHES, cases=args.cases,
                x86_instance_type=args.x86_instance_type))


if __name__ == "__main__":
    main()
