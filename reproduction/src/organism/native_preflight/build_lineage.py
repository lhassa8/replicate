"""Prepare one common lineage kit from the admitted single-device release."""

import argparse
import gzip
import io
import json
from pathlib import Path
import shutil
import tarfile

from ..training_inputs import require
from .build_profile import FIRST_LINEAGE_SOURCES
from .common import check_file, digest, write_json
from .first_lineage_profile import CASES_NAME, descriptor, validate_cases
from .infrastructure import storage_template
from .lineage_host import receiver_inputs
from .lineage_infrastructure import worker_template
from .profile_operations import checked_kit
from .trained_inputs import validate_catalog


SOURCES = tuple(sorted(set(FIRST_LINEAGE_SOURCES) | {
    "native_preflight/lineage_network.py", "native_preflight/lineage_enrollment.py",
    "native_preflight/lineage_provision.py", "native_preflight/lineage_observe.py",
    "native_preflight/lineage_endurance.py", "native_preflight/lineage_control.py",
    "native_preflight/lineage_permit.py", "native_preflight/lineage_host.py",
    "native_preflight/lineage_entry.py",
}))


def checked(directory, single_kit):
    """Recheck carried source and release bytes before native launch or auditing."""
    directory, single_kit = Path(directory).resolve(), Path(single_kit).resolve()
    original, original_inventory = checked_kit(single_kit)
    build = json.loads((directory / "build-receipt.json").read_bytes())
    require(build["schema"] == "organism-native-lineage-kit-v1"
            and build["single_device_kit_receipt_sha256"] == digest(single_kit / "build-receipt.json")
            and build["release_id"] == original["release_id"]
            and build.get("x86_instance_type") == original.get("x86_instance_type")
            and build["cases"] == descriptor()["cases"]
            and build["execution_slots"] == ["seed", "child", "grandchild"]
            and build["original_birth_seconds"] == 1200 and build["maximum_lease_seconds"] == 14400
            and build["owner_full_package_slot"] == "seed" and build["fresh_private_bundles_required"] is True
            and build["single_device_native_gate_required"] is True
            and build["all_three_trials_require_same_release"] is True
            and all(build[key] is False for key in
                    ("launch_ready", "native_execution_verified", "resources_created", "qualification_passed")),
            "Lineage kit changed its original release or execution scope")
    for name, field in (("kit.tar.gz", "kit_sha256"), ("catalog.json", "catalog_sha256"),
                        ("worker-template.json", "worker_template_sha256"),
                        ("storage-template.json", "storage_template_sha256")):
        require(digest(directory / name) == build[field], "Lineage kit artifact identity changed")
    require(json.loads((directory / "worker-template.json").read_bytes()) == worker_template(
                x86_instance_type=build.get("x86_instance_type"))
            and json.loads((directory / "storage-template.json").read_bytes()) == storage_template()
            and (directory / "catalog.json").read_bytes() == (single_kit / "catalog.json").read_bytes(),
            "Lineage launch template or catalog differs")
    validate_cases(directory / CASES_NAME)
    catalog = validate_catalog(json.loads((directory / "catalog.json").read_bytes()))
    workspace = Path(__file__).resolve().parents[2]
    require(set(build["sources"]) == set(SOURCES)
            and (directory / "kit.tar.gz").stat().st_size <= 2 * 1024**2,
            "Lineage source inventory exceeds the bootstrap bound")
    seen = set()
    with tarfile.open(directory / "kit.tar.gz") as archive:
        for member in archive:
            name = member.name.removeprefix("organism/")
            require(member.name == "organism/" + name and name in SOURCES and name not in seen
                    and member.isfile() and 0 < member.size <= 1024**2, "Unexpected lineage source member")
            with archive.extractfile(member) as stream:
                raw = stream.read(member.size + 1)
            import hashlib
            require({"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} == build["sources"][name],
                    "Carried lineage source differs from its receipt")
            check_file(workspace / "organism" / name, build["sources"][name])
            seen.add(name)
    require(seen == set(SOURCES) and sum(row["bytes"] for row in build["sources"].values()) <= 8 * 1024**2,
            "Carried lineage sources are incomplete")
    inventory = json.loads((directory / "upload-inventory.json").read_bytes())
    expected = {
        "inputs/capsule/" + name: {"path": original_inventory["inputs/capsule/" + name]["path"], **entry}
        for name, entry in catalog["files"].items()}
    expected.update({
        "inputs/receiver/" + name: {"path": original_inventory["inputs/capsule/" + name]["path"], **entry}
        for name, entry in receiver_inputs(catalog).items()})
    expected.update({"inputs/" + name: {"path": str(directory / name),
                                      "bytes": (directory / name).stat().st_size,
                                      "sha256": digest(directory / name)}
                     for name in ("kit.tar.gz", "catalog.json", CASES_NAME)})
    actual = {row["key_suffix"]: {key: row[key] for key in ("path", "bytes", "sha256")}
              for row in inventory["files"]}
    require(len(actual) == len(inventory["files"]) and actual == expected
            and inventory["total_upload_bytes"] == sum(row["bytes"] for row in expected.values())
            and inventory["private_enrollment_bundles_added_separately"] is True,
            "Lineage upload inventory adds or omits a component")
    return build, catalog, actual


def source_archive(workspace, target):
    observed = {}
    with Path(target).open("xb") as raw, gzip.GzipFile(
            fileobj=raw, filename="", mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name in SOURCES:
                path = Path(workspace) / "organism" / name
                require(path.is_file() and not path.is_symlink()
                        and path.stat().st_nlink == 1, "Owner source is missing, linked or not regular")
                payload = path.read_bytes()
                require(len(payload) <= 1024**2, "Owner source exceeds its individual bound")
                compile(payload, name, "exec")
                entry = tarfile.TarInfo("organism/" + name)
                entry.size, entry.mode, entry.mtime = len(payload), 0o644, 0
                archive.addfile(entry, io.BytesIO(payload))
                observed[name] = {"bytes": len(payload), "sha256": digest(path)}
    require(len(observed) <= 100 and sum(row["bytes"] for row in observed.values()) <= 8 * 1024**2,
            "Lineage source kit exceeds the original bootstrap extraction limit")
    return observed


def build(workspace, single_kit, output):
    """No keys or compute are created; external native admission remains required."""
    workspace, single_kit, output = Path(workspace).resolve(), Path(single_kit).resolve(), Path(output).absolute()
    original, original_inventory = checked_kit(single_kit)
    require(original["schema"] == "organism-native-profile-kit-v3"
            and original["first_lineage"] == descriptor(), "Lineage requires the exact single-device release")
    catalog = validate_catalog(json.loads((single_kit / "catalog.json").read_bytes()))
    validate_cases(single_kit / CASES_NAME)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("catalog.json", CASES_NAME):
        shutil.copyfile(single_kit / name, output / name)
    sources = source_archive(workspace, output / "kit.tar.gz")
    require(all(sources[name] == identity for name, identity in original["sources"].items()),
            "The single-device owner sources changed before lineage preparation")
    write_json(output / "worker-template.json", worker_template(
        x86_instance_type=original.get("x86_instance_type")))
    write_json(output / "storage-template.json", storage_template())
    inventory = []
    for name, entry in sorted(catalog["files"].items()):
        source = original_inventory["inputs/capsule/" + name]
        inventory.append({"path": source["path"], "key_suffix": "inputs/capsule/" + name, **entry})
    for name, entry in receiver_inputs(catalog).items():
        source = original_inventory["inputs/capsule/" + name]
        inventory.append({"path": source["path"], "key_suffix": "inputs/receiver/" + name, **entry})
    for name in ("kit.tar.gz", "catalog.json", CASES_NAME):
        path = output / name
        inventory.append({"path": str(path), "key_suffix": "inputs/" + name,
                          "bytes": path.stat().st_size, "sha256": digest(path)})
    write_json(output / "upload-inventory.json", {
        "files": inventory, "total_upload_bytes": sum(row["bytes"] for row in inventory),
        "private_enrollment_bundles_added_separately": True,
    })
    write_json(output / "build-receipt.json", {
        "schema": "organism-native-lineage-kit-v1", "sources": sources,
        "release_id": catalog["release_id"], "kit_sha256": digest(output / "kit.tar.gz"),
        "catalog_sha256": digest(output / "catalog.json"),
        "worker_template_sha256": digest(output / "worker-template.json"),
        "storage_template_sha256": digest(output / "storage-template.json"),
        "single_device_kit_receipt_sha256": digest(single_kit / "build-receipt.json"),
        "cases": descriptor()["cases"], "execution_slots": ["seed", "child", "grandchild"],
        "all_three_trials_require_same_release": True, "original_birth_seconds": 1200,
        "maximum_lease_seconds": 14400, "owner_full_package_slot": "seed",
        "fresh_private_bundles_required": True, "single_device_native_gate_required": True,
        "launch_ready": False, "native_execution_verified": False, "resources_created": False,
        "qualification_passed": False,
        **({"x86_instance_type": original["x86_instance_type"]}
           if "x86_instance_type" in original else {}),
    })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-kit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(Path(__file__).resolve().parents[2], args.single_kit, args.output))


if __name__ == "__main__":
    main()
