"""Bind an actual completed GPU result into a separate CPU export source kit."""

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

from .export_host import checked_input, extract_archive
from .export_infrastructure import worker_template, storage_template
from .export_trained_model import SOURCES
from .export_worker import STANDARD_PROFILE, prepare_recipe, profile_for_job
from .gguf_export import canonical, identity
from .training_inputs import load, require


def tool_inventory(directory):
    directory = Path(directory).resolve()
    result = {}
    for path in sorted(directory.rglob("*")):
        name = str(path.relative_to(directory))
        if path.is_symlink():
            require(path.resolve().is_relative_to(directory), "Export tool alias escapes its bundle")
            result[name] = {"target": os.readlink(path), **identity(path)}
        elif path.is_file():
            result[name] = identity(path)
    return result


def build(root, training_kit, training_results, training_identity, tools_root, tools_archive, output, *,
          calibration=None, precision_profile=None, managed_evidence=None):
    root, training_kit, training_results, training_identity, tools_root, tools_archive, output = (
        Path(value).resolve() for value in
        (root, training_kit, training_results, training_identity, tools_root, tools_archive, output))
    require(not output.exists() and output.parent.is_dir(), "CPU export package output must be new")
    expected_results = load(training_identity)
    require(set(expected_results) == {"bytes", "sha256"}
            and identity(training_results) == expected_results
            and 0 < expected_results["bytes"] <= 2 * 1024**3, "Returned GPU archive differs from its saved identity")
    require(identity(tools_archive)["bytes"] <= 200 * 1024**2, "Export tools archive exceeds its bound")
    with tempfile.TemporaryDirectory(prefix="organism-cpu-export-build-") as scratch:
        temporary = Path(scratch)
        workspace, returned, extracted_tools = (temporary / name for name in ("workspace", "training", "tools"))
        require(training_kit.stat().st_size <= 64 * 1024**2, "Original training kit exceeds its bound")
        extract_archive(training_kit, workspace, source=True)
        require(sum(path.stat().st_size for path in workspace.rglob("*") if path.is_file()) <= 256 * 1024**2,
                "Original training source exceeds its extraction bound")
        extract_archive(training_results, returned)
        managed_provenance = None
        if managed_evidence is not None:
            from .training_worker.managed_export import verify_transfer
            managed_provenance = verify_transfer(workspace, returned, managed_evidence)
        else:
            host = load(returned / "host-receipt.json")
            require(host.get("failure") is None and host.get("mode") == "train"
                and host.get("preparation_passed") is True and host.get("service_stopped_confirmed") is True
                and host.get("native_source_files_unchanged") is True
                and host.get("service_result", {}).get("ActiveState") == "inactive"
                and host["service_result"].get("Result") == "success"
                and host["service_result"].get("ExecMainStatus") == "0",
                    "A completed, stopped and successful GPU host is required")
            permit = host["launch_permit"]
            require(permit["kit_sha256"] == identity(training_kit)["sha256"]
                and permit["instance_id"] == host["instance_id"] and permit["deadline"] == host["deadline"],
                    "GPU host receipt belongs to another input kit, instance or deadline")
        extract_archive(tools_archive, extracted_tools, tools=True)
        require(tool_inventory(extracted_tools) == tool_inventory(tools_root),
                "Tools archive and prepared profile differ")
        for name in (*SOURCES, "export_worker.py", "export_host.py"):
            source, destination = root / "organism" / name, workspace / "organism" / name
            if destination.exists():
                require(identity(source) == identity(destination), "Export would replace a frozen training source: " + name)
            else:
                shutil.copyfile(source, destination)
        if calibration is not None:
            from transformers import AutoTokenizer
            from .adapter_export import admit_training_export
            from .quantization_calibration import verify_calibration
            admitted = admit_training_export(workspace, returned / "workload")
            tokenizer = AutoTokenizer.from_pretrained(admitted["paths"]["tokenizer_receipt"].parent,
                                                      local_files_only=True)
            verify_calibration(admitted, tokenizer, calibration)
            shutil.copytree(calibration, workspace / "export-calibration")
        job = prepare_recipe(workspace, returned / "workload", extracted_tools, workspace / "export-job.json",
                             calibrated=calibration is not None, precision_profile=precision_profile)
        original = load(workspace / "worker-input.json")
        training_recipe = load(workspace / original["recipe"])
        if managed_provenance is not None:
            require(training_recipe["corpus"] == "first_lineage_runtime",
                    "Managed export is restricted to the separately admitted first-lineage recipe")
            (workspace / "managed-training-provenance.json").write_bytes(canonical(managed_provenance))
        files = {str(path.relative_to(workspace)): path for path in workspace.rglob("*") if path.is_file()}
        require(len(files) <= 128 and sum(path.stat().st_size for path in files.values()) <= 256 * 1024**2,
                "CPU export source package exceeds its bound")
        manifest = {
            "schema": "organism-cpu-export-input-v1", "limits": job["limits"],
            "files": [{"path": name, **identity(path)} for name, path in sorted(files.items())],
            "base_metadata": training_recipe["inputs"]["base_metadata"]["path"],
            "archives": {"training_results": identity(training_results), "tools": identity(tools_archive)},
            "qualification_passed": False,
        }
        profile = profile_for_job(job)
        if profile != STANDARD_PROFILE:
            manifest["worker_profile"] = profile
        (workspace / "export-input.json").write_bytes(canonical(manifest))
        checked_input(workspace)
        output.mkdir()
        kit = output / "kit.tar.gz"
        with kit.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for name, path in sorted({**files, "export-input.json": workspace / "export-input.json"}.items()):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
        require(kit.stat().st_size <= 64 * 1024**2, "Compressed CPU source kit exceeds its bootstrap bound")
        for name, value in (("export-input.json", manifest), ("export-job.json", job),
                            ("worker-template.json", worker_template()), ("storage-template.json", storage_template())):
            (output / name).write_text(json.dumps(value, indent=2) + "\n")
        receipt = {
            "schema": "organism-cpu-export-build-v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "original_training_kit": identity(training_kit), "original_gpu_results": identity(training_results),
            "input_files": len(files), "input_bytes": sum(path.stat().st_size for path in files.values()),
            "external_inputs": {"tools.tar.gz": identity(tools_archive), "training-results.tar.gz": identity(training_results)},
            "artifacts": {name: identity(output / name) for name in
                          ("kit.tar.gz", "export-input.json", "export-job.json", "worker-template.json", "storage-template.json")},
            "source_hashes": {name: identity(root / "organism" / name) for name in
                              ("export_build.py", "export_host.py", "export_infrastructure.py", "export_worker.py")},
            "completed_training_bound": True, "spending_authorized": False,
            "cloud_resources_created": False, "native_export_execution_verified": False, "qualification_passed": False,
        }
        if profile != STANDARD_PROFILE:
            receipt["worker_profile"] = profile
        if managed_provenance is not None:
            receipt["managed_training"] = {
                "provenance": identity(workspace / "managed-training-provenance.json"),
                "job_readback": managed_provenance["job_readback"],
                "original_managed_output": managed_provenance["original_managed_output"],
                "actual_managed_controls_reconstructed": True,
                "ec2_host_receipt_synthesized": False,
            }
        (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "training-kit", "training-results", "training-identity", "tools-root", "tools-archive", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--precision-profile")
    parser.add_argument("--managed-evidence", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.training_kit, args.training_results, args.training_identity,
                           args.tools_root, args.tools_archive, args.output, calibration=args.calibration,
                           precision_profile=args.precision_profile,
                           managed_evidence=args.managed_evidence), indent=2))
