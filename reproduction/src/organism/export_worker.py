"""Offline CPU merge/export stages and bounded result retention.

This module does not provision a host, authorize spending, download inputs,
upload results, select a checkpoint, or qualify a deployment.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time

from .adapter_export import admit_training_export
from .gguf_export import Toolchain, canonical, converter_command, identity
from .runtime_environment import library_environment
from .training_inputs import load, offline_environment, require, verified_path


LIMITS = {
    "workload_seconds": 14400, "merge_seconds": 7200, "export_seconds": 7200,
    "maximum_memory_bytes": 56 * 1024**3, "maximum_cpus": 8, "maximum_swap_bytes": 0,
    "minimum_initial_free_bytes": 96 * 1024**3, "maximum_workload_bytes": 96 * 1024**3,
    "maximum_log_bytes": 16 * 1024**2, "maximum_result_bytes": 6 * 1024**3,
    "maximum_metadata_bytes": 256 * 1024**2, "maximum_metadata_file_bytes": 32 * 1024**2,
    "maximum_q4_file_bytes": 3 * 1024**3, "maximum_result_files": 512,
}
MODEL_RESULTS = {"workload/export/stock-q4.gguf", "workload/export/trained-q4.gguf"}
STANDARD_PROFILE = "standard-v1"
FIRST_LINEAGE_PROFILE = "first-lineage-v1"


def limits_for_profile(profile=STANDARD_PROFILE):
    require(profile in (STANDARD_PROFILE, FIRST_LINEAGE_PROFILE), "Unknown CPU export worker profile")
    limits = dict(LIMITS)
    if profile == FIRST_LINEAGE_PROFILE:
        limits.update(workload_seconds=7200, merge_seconds=2700, export_seconds=4500)
    return limits


def lease_envelope(profile=STANDARD_PROFILE):
    limits = limits_for_profile(profile)
    return {"workload_seconds": limits["workload_seconds"], "cleanup_seconds": 900,
            "preparation_seconds": 600,
            "maximum_lease_seconds": 14400 if profile == FIRST_LINEAGE_PROFILE else 21600}


def profile_for_training(recipe):
    return FIRST_LINEAGE_PROFILE if recipe.get("corpus") == "first_lineage_runtime" else STANDARD_PROFILE


def profile_for_job(recipe):
    profile = recipe.get("worker_profile", STANDARD_PROFILE)
    limits_for_profile(profile)
    return profile


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_identities(workspace):
    from .export_trained_model import SOURCES
    return {name: identity(Path(workspace) / "organism" / name) for name in (*SOURCES, "export_worker.py")}


def numerical_limits():
    from .export_trained_model import LIMITS
    return LIMITS


def calibration_receipt(workspace, admitted, tools_root):
    from transformers import AutoTokenizer
    from .quantization_calibration import verify_calibration

    directory = Path(workspace) / "export-calibration"
    manifest = load(Path(tools_root) / "toolchain.json")
    require("imatrix" in manifest and manifest["imatrix"] in manifest["files"],
            "Calibrated job requires the recorded calibration executable")
    tokenizer = AutoTokenizer.from_pretrained(admitted["paths"]["tokenizer_receipt"].parent, local_files_only=True)
    verify_calibration(admitted, tokenizer, directory)
    return {"directory": "export-calibration", "plan": identity(directory / "plan.json"),
            "text": identity(directory / "calibration.txt")}


def prepare_recipe(workspace, worker_output, tools_root, output, *, calibrated=False, precision_profile=None):
    """Bind a completed real pilot; missing or failed training cannot make a job."""
    workspace, worker_output, tools_root, output = (
        Path(path).resolve() for path in (workspace, worker_output, tools_root, output))
    require(not output.exists(), "Export recipe output must be new")
    admitted = admit_training_export(workspace, worker_output)
    profile = profile_for_training(admitted["recipe"])
    require(type(calibrated) is bool, "Invalid calibrated-job selection")
    manifest = load(tools_root / "toolchain.json")
    require((manifest["system"], manifest["machine"]) == ("Linux", "x86_64"),
            "CPU export requires the prepared Linux x86-64 tools")
    tools = Toolchain(tools_root, manifest)
    for name, entry in manifest["files"].items():
        path = tools.path(name)
        require(identity(path) == entry, "Prepared export tool changed")
        if name in manifest["symlinks"]:
            require(path.is_symlink() and os.readlink(path) == manifest["symlinks"][name],
                    "Prepared export alias changed")
        else:
            verified_path(tools_root, {"path": name, **entry})
    recipe = {
        "schema": "organism-cpu-export-job-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "training_admission": admitted["receipt"], "toolchain": identity(tools_root / "toolchain.json"),
        "sources": source_identities(workspace), "limits": limits_for_profile(profile),
        "numerical_limits": numerical_limits(),
        "device": "cpu", "optimizer_steps_this_stage": 0, "quality_selection_performed": False,
        "spending_authorized": False, "qualification_passed": False,
    }
    if profile != STANDARD_PROFILE:
        recipe["worker_profile"] = profile
    if calibrated:
        recipe["calibration"] = calibration_receipt(workspace, admitted, tools_root)
    else:
        require(not (workspace / "export-calibration").exists(), "Unused calibration input in ordinary export")
    if precision_profile is not None:
        from .quantization_precision import specification
        require(calibrated, "Precision experiment requires bound calibration")
        recipe["schema"] = "organism-cpu-export-job-v2"
        recipe["quantization_precision"] = specification(precision_profile)
    write(output, recipe)
    return recipe


def validate_precision_job(recipe):
    if recipe.get("schema") == "organism-cpu-export-job-v2":
        from .quantization_precision import validate_specification
        require("calibration" in recipe, "Precision experiment requires bound calibration")
        return validate_specification(recipe.get("quantization_precision"))
    require(recipe.get("schema") == "organism-cpu-export-job-v1"
            and "quantization_precision" not in recipe, "Export precision version differs")
    return None


def validate_recipe(workspace, worker_output, tools_root, recipe):
    validate_precision_job(recipe)
    profile = profile_for_job(recipe)
    require(recipe.get("device") == "cpu"
            and canonical(recipe.get("limits")) == canonical(limits_for_profile(profile))
            and canonical(recipe.get("numerical_limits")) == canonical(numerical_limits()),
            "CPU export job changed its declared limits or device")
    require(recipe.get("optimizer_steps_this_stage") == 0
            and type(recipe["optimizer_steps_this_stage"]) is int
            and recipe.get("quality_selection_performed") is False
            and recipe.get("qualification_passed") is False, "Export job implies optimization or qualification")
    require(recipe["sources"] == source_identities(workspace), "Export implementation changed")
    require(recipe["toolchain"] == identity(Path(tools_root) / "toolchain.json"), "Export tool profile changed")
    admitted = admit_training_export(workspace, worker_output)
    require(recipe["training_admission"] == admitted["receipt"], "Export job belongs to another training run")
    require(profile == profile_for_training(admitted["recipe"]),
            "CPU export worker profile differs from the admitted training corpus")
    if "calibration" in recipe:
        require(recipe["calibration"] == calibration_receipt(workspace, admitted, tools_root),
                "Calibrated export job changed its training-only input")
    else:
        require(not (Path(workspace) / "export-calibration").exists(), "Unbound calibration input")
    return admitted


def process_environment(cache):
    environment = {key: value for key, value in os.environ.items() if key in ("PATH", "LANG", "TMPDIR")}
    environment.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        AWS_EC2_METADATA_DISABLED="true", OTEL_SDK_DISABLED="true", WANDB_DISABLED="true",
        TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
        XDG_CACHE_HOME=str(cache), HOME=str(cache))
    return environment


def run_stage(argv, log, *, seconds, environment, maximum_log_bytes=LIMITS["maximum_log_bytes"]):
    require(0 < seconds <= 7200 and 0 < maximum_log_bytes <= LIMITS["maximum_log_bytes"],
            "Invalid export stage bound")
    started = time.monotonic()
    record = {"argv": list(map(str, argv)), "seconds_limit": seconds, "passed": False, "failure": None}
    process = None
    try:
        with Path(log).open("xb") as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=environment,
                                       start_new_session=True)
            record["pid"] = process.pid
            while True:
                require(Path(log).stat().st_size <= maximum_log_bytes, "Export stage log exceeds its bound")
                if time.monotonic() - started > seconds:
                    raise TimeoutError("Export stage deadline exhausted")
                if process.poll() is not None:
                    break
                time.sleep(min(0.1, seconds))
            require(process.returncode == 0, "Export stage failed; see " + str(log))
            record["passed"] = True
    except BaseException as exc:
        record["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        if process is not None:
            try:
                # Reap completed children before signaling: an exited process
                # no longer establishes ownership of a live process group.
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        if process.poll() is None:
                            process.kill()
                            process.wait(timeout=10)
                            raise RuntimeError("Could not stop the full owned export process group")
                process.wait(timeout=10)
            except BaseException as exc:
                record["passed"] = False
                record["cleanup_failure"] = type(exc).__name__ + ": " + str(exc)
                raise
            finally:
                record["return_code"] = process.returncode
                record["seconds"] = round(time.monotonic() - started, 3)
                write(Path(log).with_suffix(".stage.json"), record)
        else:
            record["seconds"] = round(time.monotonic() - started, 3)
            write(Path(log).with_suffix(".stage.json"), record)
    return record


def module_command(workspace, module, arguments):
    require(module in ("organism.merge_trained_adapter", "organism.export_trained_model"),
            "Unrecognized CPU export stage")
    wrapper = "import runpy,sys;sys.path.insert(0,sys.argv[1]);m=sys.argv[2];sys.argv=sys.argv[2:];runpy.run_module(m,run_name='__main__')"
    return [sys.executable, "-I", "-c", wrapper, str(workspace), module, *map(str, arguments)]


def converter_precheck(tools, log, environment, *, seconds=60):
    """Exercise the real guarded imports before loading the full model."""
    require(0 < seconds <= 60, "Invalid converter startup allowance")
    environment = dict(environment)
    environment.update(library_environment(tuple(
        tools.path(name) for name in tools.manifest["library_paths"])))
    return run_stage(converter_command(tools, ["--help"]), log, seconds=seconds, environment=environment)


def resource_limits(memory, swap, cpu):
    quota, period = cpu.split()
    require(memory.isdecimal() and 0 < int(memory) <= LIMITS["maximum_memory_bytes"]
            and swap == "0" and quota.isdecimal() and period.isdecimal() and int(period) > 0
            and 0 < int(quota) <= LIMITS["maximum_cpus"] * int(period),
            "CPU export resource limits are absent or too broad")
    return {"memory_max": int(memory), "memory_swap_max": int(swap),
            "cpu_quota": int(quota), "cpu_period": int(period)}


def runtime_context(preparation, *, cgroup_root=Path("/sys/fs/cgroup")):
    require(platform.system() == "Linux" and platform.machine() == "x86_64" and os.geteuid() != 0,
            "Export workload requires an unprivileged native Linux x86-64 service")
    path = Path(preparation)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_uid == 0 and info.st_mode & 0o022 == 0,
            "Export host preparation must be root-owned and read-only to the worker")
    prepared = load(path)
    namespace = os.readlink("/proc/self/ns/net")
    require(namespace != prepared["host_network_namespace"], "Export workload shares the online host namespace")
    groups = Path("/proc/self/cgroup").read_text().splitlines()
    require(len(groups) == 1 and groups[0].startswith("0::/"), "CPU export requires a unified cgroup")
    relative = groups[0][4:]
    require(".." not in Path(relative).parts, "Invalid workload cgroup path")
    group = cgroup_root / relative
    memory, swap = (group / "memory.max").read_text().strip(), (group / "memory.swap.max").read_text().strip()
    limits = resource_limits(memory, swap, (group / "cpu.max").read_text())
    return {"network_namespace": namespace, "network_isolation_observed": True,
            "cgroup": relative, **limits,
            "host_preparation": identity(path), "instance_id": prepared["instance_id"]}


def run(workspace, worker_output, base, tools_root, output, recipe_path, preparation):
    offline_environment()
    started = time.monotonic()
    workspace, worker_output, base, tools_root, output, recipe_path = (
        Path(path).resolve() for path in (workspace, worker_output, base, tools_root, output, recipe_path))
    context = runtime_context(preparation)
    recipe = load(recipe_path)
    admitted = validate_recipe(workspace, worker_output, tools_root, recipe)
    limits = recipe["limits"]
    require(not output.exists() and output.parent.is_dir()
            and all(not output.is_relative_to(path) and not path.is_relative_to(output)
                    for path in (workspace, worker_output, base, tools_root)),
            "CPU export output overlaps an input")
    require(shutil.disk_usage(output.parent).free >= LIMITS["minimum_initial_free_bytes"],
            "CPU export requires 96 GiB free before merging")
    tools = Toolchain(tools_root, load(tools_root / "toolchain.json"))
    verified_tools = tools.verify()
    output.mkdir()
    cache = output / "cache"
    cache.mkdir()
    environment = process_environment(cache)
    report = {"schema": "organism-cpu-export-workload-v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "context": context, "job_recipe": identity(recipe_path), "training_admission": admitted["receipt"],
              "tools": verified_tools, "stages": {}, "passed": False, "failure": None,
              "optimizer_steps_this_stage": 0, "quality_selection_performed": False,
              "export_pair_verified": False, "deployment_export_complete": False, "qualification_passed": False}
    write(output / "job.json", recipe)

    def stage(name, module, arguments, limit):
        remaining = limits["workload_seconds"] - (time.monotonic() - started)
        require(remaining > 0, "CPU export workload deadline exhausted")
        count = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        require(count <= LIMITS["maximum_workload_bytes"], "CPU export scratch allowance exhausted")
        report["stages"][name] = {"record": name + ".stage.json", "log": name + ".log"}
        report["stages"][name] = run_stage(module_command(workspace, module, arguments), output / (name + ".log"),
                                         seconds=min(limit, remaining), environment=environment)

    try:
        report["stages"]["converter_precheck"] = {"record": "converter-precheck.stage.json",
                                                  "log": "converter-precheck.log"}
        report["stages"]["converter_precheck"] = converter_precheck(
            tools, output / "converter-precheck.log", environment,
            seconds=min(60, limits["workload_seconds"] - (time.monotonic() - started)))
        if "calibration" in recipe:
            calibration_environment = dict(environment)
            calibration_environment.update(library_environment(tuple(
                tools.path(name) for name in tools.manifest["library_paths"])))
            report["stages"]["calibration_precheck"] = run_stage(
                [str(tools.path(tools.manifest["imatrix"])), "--help"],
                output / "calibration-precheck.log", seconds=60, environment=calibration_environment)
        stage("merge", "organism.merge_trained_adapter",
              ["--workspace", workspace, "--worker-output", worker_output, "--base", base,
               "--output", output / "merge", "--device", "cpu"], limits["merge_seconds"])
        export_arguments = [
            "--workspace", workspace, "--worker-output", worker_output, "--merge-output", output / "merge",
            "--output", output / "export", "--tools-root", tools_root,
            "--tools-manifest", tools_root / "toolchain.json"]
        if "calibration" in recipe:
            export_arguments += ["--calibration", workspace / "export-calibration"]
        precision_profile = validate_precision_job(recipe)
        if precision_profile is not None:
            export_arguments += ["--precision-profile", precision_profile]
        stage("export", "organism.export_trained_model", export_arguments, limits["export_seconds"])
        exported = load(output / "export/report.json")
        require(exported.get("passed") is True and exported.get("export_pair_verified") is True
                and exported.get("qualification_passed") is False, "Full-size export did not verify the matched pair")
        if precision_profile is not None:
            require(exported.get("schema") == "organism-trained-gguf-pair-export-v2"
                    and exported.get("quantization_precision", {}).get("specification")
                    == recipe["quantization_precision"], "Export precision differs from the bound job")
        require(validate_recipe(workspace, worker_output, tools_root, recipe)["receipt"] == admitted["receipt"]
                and tools.verify() == verified_tools, "CPU export inputs changed")
        require(time.monotonic() - started <= limits["workload_seconds"], "CPU export workload deadline exhausted")
        report.update(passed=True, export_pair_verified=True)
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["seconds"] = round(time.monotonic() - started, 3)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        write(output / "workload.json", report)
    return report


def package_results(results, target):
    """Retain Q4 artifacts, reports, references and logs; inventory intermediates."""
    results, target = Path(results).resolve(), Path(target).resolve()
    require(not target.exists() and not target.is_relative_to(results), "Result archive overlaps its inputs")
    selected, omitted, metadata_bytes, total = {}, {}, 0, 0
    for path in sorted(results.rglob("*")):
        relative = path.relative_to(results)
        if "cache" in relative.parts:
            continue
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), "Export result contains a symbolic link")
        if stat.S_ISDIR(info.st_mode):
            continue
        require(stat.S_ISREG(info.st_mode), "Export result contains a special file")
        name = str(relative)
        skip = ({"base-loading-view", "stock-loading-view", "merged"} & set(relative.parts)
                or path.name in ("stock-f32.gguf", "trained-f32.gguf"))
        # Small loader indexes are needed to diagnose a failed export.
        if path.name == "model.safetensors.index.json":
            skip = False
        if skip:
            require(len(omitted) < 256 and info.st_size <= 20 * 1024**3, "Intermediate inventory exceeds its bound")
            omitted[name] = {**identity(path), "retained_as": "identity only; reproducible intermediate"}
            continue
        require(info.st_nlink == 1, "Export result unexpectedly shares a hard link")
        if name in MODEL_RESULTS:
            require(info.st_size <= LIMITS["maximum_q4_file_bytes"], "Q4 result exceeds its transfer bound")
        else:
            require(info.st_size <= LIMITS["maximum_metadata_file_bytes"], "Export metadata file exceeds its bound")
            metadata_bytes += info.st_size
        total += info.st_size
        require(metadata_bytes <= LIMITS["maximum_metadata_bytes"]
                and total <= LIMITS["maximum_result_bytes"]
                and len(selected) < LIMITS["maximum_result_files"] - 1, "Export results exceed transfer bounds")
        selected[name] = identity(path)
    inventory = {"schema": "organism-cpu-export-retention-v1", "files": selected,
                 "intermediates_not_transferred": omitted, "metadata_bytes": metadata_bytes,
                 "payload_bytes": total, "limits": LIMITS, "qualification_passed": False}
    receipt_path = results / "retention.json"
    write(receipt_path, inventory)
    require(total + receipt_path.stat().st_size <= LIMITS["maximum_result_bytes"], "Retention receipt exceeds result bound")
    with tarfile.open(target, "w:gz") as archive:
        for name, expected in selected.items():
            path = verified_path(results, {"path": name, **expected})
            archive.add(path, arcname=name, recursive=False)
        archive.add(receipt_path, arcname=receipt_path.name, recursive=False)
    require(target.stat().st_size <= LIMITS["maximum_result_bytes"], "Compressed export result exceeds its bound")
    return {**identity(target), "retention": identity(receipt_path), "regular_files": len(selected) + 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-recipe", action="store_true")
    parser.add_argument("--calibrated", action="store_true")
    parser.add_argument("--precision-profile")
    for name in ("workspace", "worker-output", "tools-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("base", "recipe", "host-preparation"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.prepare_recipe:
        result = prepare_recipe(args.workspace, args.worker_output, args.tools_root, args.output,
                                calibrated=args.calibrated, precision_profile=args.precision_profile)
    else:
        if any(value is None for value in (args.base, args.recipe, args.host_preparation)):
            parser.error("--base, --recipe and --host-preparation are required")
        result = run(args.workspace, args.worker_output, args.base, args.tools_root, args.output,
                     args.recipe, args.host_preparation)
    print(json.dumps(result, indent=2))
