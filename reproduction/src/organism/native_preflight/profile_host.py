"""Bounded dedicated-host controller for the separate-service trained seed.

Online S3 preparation and evidence upload stay in this owner process. The
receiver and bootstrap driver enter private namespaces and clear credentials.
The fixed guest/AWS expiry is installed independently before this controller.
"""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time

from .. import device_resources, device_storage
from ..runtime_environment import library_environment
from .common import ARCHES, digest, extract, write_json
from .host import bounded_results, download
from .profile_permit import instance_identity, remaining, wait as wait_permit
from .profile_setup import HostProfile, NAMESPACES
from . import profile_census
from . import first_lineage_profile
from .startup_host import owned_units
from .trained_inputs import validate_catalog
from .verify_libraries import verify_carried


OUTPUT = Path("/var/lib/organism-profile-output")
INPUT = device_storage.ROOT / "data/seed-input"
BOOT = device_storage.ROOT / "data/bootstrap"
CONFIG = device_storage.ROOT / "control/receiver.json"
DRIVER = "organism-native-seed.service"
MAX_CATALOG = 1024**2


def runner(libraries, operation, architecture, *, cases_sha256=None):
    first_lineage = first_lineage_profile.validate_selection(cases_sha256, architecture)
    environment = {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/nonexistent", "LC_ALL": "C.UTF-8",
        "AWS_EC2_METADATA_DISABLED": "true", "OTEL_SDK_DISABLED": "true",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        **library_environment(libraries),
    }
    command = [
        "/usr/bin/env", "-i", *(key + "=" + value for key, value in environment.items()),
        str(BOOT / "python/bin/python3.12"), "-I", "-B",
        str(Path(__file__).with_name("profile_entry.py")),
        "--operation", operation, "--architecture", architecture,
        "--catalog", str(INPUT / "catalog.json"), "--capsule", str(INPUT / "capsule"),
        "--output", str(OUTPUT / (operation + "-receipt.json")),
    ]
    if first_lineage:
        command += ["--cases", str(OUTPUT / first_lineage_profile.CASES_NAME),
                    "--cases-sha256", cases_sha256]
    return command


def driver_command(libraries, architecture, *, cases_sha256=None):
    first_lineage = first_lineage_profile.validate_selection(cases_sha256, architecture)
    properties = {
        "User": "root", "Group": "root", "NetworkNamespacePath": "/run/netns/" + NAMESPACES[1],
        "UMask": "0077", "KillMode": "control-group", "MemoryAccounting": "yes",
        "MemoryMax": "512M", "MemorySwapMax": "0", "CPUQuota": "100%",
        "TasksMax": "64", "RuntimeMaxSec": "1500", "TimeoutStopSec": "20",
    }
    if first_lineage:
        properties["RuntimeMaxSec"] = str(first_lineage_profile.DRIVER_SECONDS)
    return ["systemd-run", "--unit=" + DRIVER, "--wait", "--pipe", "--service-type=exec",
            *["--property=" + key + "=" + value for key, value in properties.items()],
            *runner(libraries, "drive", architecture, cases_sha256=cases_sha256)]


def prepare(client, args):
    INPUT.mkdir(mode=0o755)
    response = client.get_object(Bucket=args.bucket, Key=args.prefix + "/inputs/catalog.json")
    try:
        if not 0 < response["ContentLength"] <= MAX_CATALOG:
            raise ValueError("Trained native catalog exceeds its bound")
        raw = response["Body"].read(MAX_CATALOG + 1)
    finally:
        response["Body"].close()
    if len(raw) != response["ContentLength"] or hashlib.sha256(raw).hexdigest() != args.catalog_sha256:
        raise ValueError("Trained native catalog differs from the approved input")
    catalog = validate_catalog(json.loads(raw))
    (INPUT / "catalog.json").write_bytes(raw)
    (OUTPUT / "input-catalog.json").write_bytes(raw)
    if getattr(args, "cases_sha256", None) is not None:
        first_lineage_profile.validate_selection(args.cases_sha256, args.architecture)
        target = OUTPUT / first_lineage_profile.CASES_NAME
        download(client, args.bucket, args.prefix + "/inputs/" + target.name, target,
                 first_lineage_profile.descriptor()["cases"])
        target.chmod(0o600)
        first_lineage_profile.validate_cases(target)
    capsule = INPUT / "capsule"
    capsule.mkdir()
    for name, entry in catalog["files"].items():
        if remaining(args.deadline) <= 300:
            raise TimeoutError("Insufficient time remains for the native seed and cleanup")
        target = capsule / name
        target.parent.mkdir(parents=True, exist_ok=True)
        download(client, args.bucket, args.prefix + "/inputs/capsule/" + name, target, entry)
    (capsule / ".manifest.json").write_bytes(base64.b64decode(catalog["manifest_base64"], validate=True))
    BOOT.mkdir()
    native = catalog["platforms"]["linux-" + args.architecture]
    extract(capsule / native["python"], BOOT)
    extract(capsule / native["dependencies"], BOOT)
    extract(capsule / native["inference"], BOOT / "tools/llama.cpp/b9850")
    libraries = verify_carried(BOOT, OUTPUT, args.architecture)
    return catalog, libraries


def owner_config():
    entry = CONFIG.lstat()
    if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != 0 or entry.st_mode & 0o077
            or entry.st_nlink != 1 or entry.st_size > 16384):
        raise ValueError("Native cleanup configuration is not a private owner file")
    value = json.loads(CONFIG.read_bytes())
    if not re.fullmatch("[0-9a-f]{64}", value.get("device_id", "")):
        raise ValueError("Native cleanup lacks a valid device identity")
    return value


def cleanup_workloads(profile, libraries, *, driver_started, maximum_starts=1):
    result = {"application_units": [], "passed": False}
    if driver_started:
        result["driver"] = profile.stop_unit(DRIVER, shared=False)
    if CONFIG.exists():
        config = owner_config()
        command = [
            "/usr/bin/env", "-i", "PATH=/usr/sbin:/usr/bin:/sbin:/bin", "PYTHONDONTWRITEBYTECODE=1",
            *(key + "=" + value for key, value in library_environment(libraries).items()),
            str(BOOT / "python/bin/python3.12"), "-I", "-B",
            str(Path(__file__).with_name("receiver_entry.py")),
            "--config", str(CONFIG), "control", "--action", "retire",
            "--reason", "Independent combined native host cleanup",
        ]
        try:
            reply = profile.command(command, check=False, timeout=60)
            result["retirement_return_code"] = reply.returncode
            result["retirement"] = reply.stdout
        except Exception as exc:
            result["retirement_failure"] = type(exc).__name__ + ": " + str(exc)
        names = owned_units(config, maximum_starts=maximum_starts)
        # The receiver must stop before taking a final stable unit inventory.
        # Retirement fences any delayed start even if the listener was busy.
        if device_resources.RECEIVER_UNIT in profile.units:
            result["receiver"] = profile.stop_unit(device_resources.RECEIVER_UNIT)
        names = sorted(set(names + owned_units(config, maximum_starts=maximum_starts)))
        for name in names:
            result["application_units"].append(profile.stop_unit(name))
        result["owned_unit_names"] = names
    elif device_resources.RECEIVER_UNIT in profile.units:
        result["receiver"] = profile.stop_unit(device_resources.RECEIVER_UNIT)
    result["passed"] = (
        all(item["absence_verified"] for item in result["application_units"])
        and result.get("driver", {"absence_verified": True})["absence_verified"]
        and result.get("receiver", {"absence_verified": True})["absence_verified"])
    return result


def copy_evidence():
    """Retain public inputs and bounded work/checker records, excluding keys/state DB."""
    target = OUTPUT / "seed-evidence"
    target.mkdir()
    control = device_storage.ROOT / "control"
    work = device_storage.ROOT / "data/work_root"
    state = control / "state_directory"
    files = [CONFIG, control / "enrollment_file", control / "discovery_file"]
    files += [state / name for name in ("receiver-status.json", "supervision.sqlite3",
                                        "supervision.enrolled", "peer-audit.sqlite3", "peer-audit.enrolled")]
    roots = [work, device_storage.ROOT / "data/logs"]
    if state.exists():
        roots.extend(path for path in state.iterdir() if re.fullmatch("[0-9a-f]{32}-check", path.name))
    for root in roots:
        if not root.exists():
            continue
        if root.is_symlink():
            raise ValueError("Native evidence root is a link")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("Native evidence contains a link")
            if path.is_file():
                files.append(path)
    files = sorted(set(path for path in files if path.exists()))
    if len(files) > 900 or sum(path.stat().st_size for path in files) > 384 * 1024**2:
        raise ValueError("Combined native evidence exceeds its bounded allowance")
    for path in files:
        entry = path.lstat()
        if (not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1
                or path.suffix in (".pem", ".key") or path.name == "state.sqlite3"):
            raise ValueError("Unexpected private or non-regular native evidence")
        destination = target / path.relative_to(device_storage.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return {"files": len(files), "device_private_keys_copied": False,
            "admission_state_database_copied": False}


def sample():
    group = Path("/sys/fs/cgroup") / device_resources.CGROUP.lstrip("/")
    if not group.exists():
        return {"aggregate_present": False}
    return {"aggregate_present": True, "observed_at": datetime.now(timezone.utc).isoformat(),
            **{name: (group / name).read_text().strip()
               for name in ("memory.current", "memory.peak", "memory.swap.current", "memory.events",
                            "cpu.stat", "pids.current")},
            "host_memory": Path("/proc/meminfo").read_text()}


def receiver_progress(expected_device_id, *, path=None):
    """Read only the public owner status; telemetry cannot authorize a start."""
    path = Path(path) if path is not None else (
        device_storage.ROOT / "control/state_directory/receiver-status.json")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            entry = os.fstat(stream.fileno())
            if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.geteuid()
                    or entry.st_nlink != 1 or stat.S_IMODE(entry.st_mode) != 0o600
                    or not 0 < entry.st_size <= 65536):
                raise ValueError("Status is not a bounded private owner file")
            raw = stream.read(65537)
        if len(raw) != entry.st_size:
            raise ValueError("Status size changed")
        value = json.loads(raw)
        if (value["schema"] != "organism-receiver-service-v1"
                or value["qualification_passed"] is not False
                or value["device_id"] != expected_device_id
                or not re.fullmatch("[0-9a-f]{64}", expected_device_id)
                or not isinstance(value["result"], dict)):
            raise ValueError("Status belongs to another receiver or scope")
        written = datetime.fromisoformat(value["observed_at"])
        if written.tzinfo is None:
            raise ValueError("Status timestamp lacks a timezone")
        result = {}
        for key in ("state", "reason", "submission_error", "owner_mode", "birth_status",
                    "operation_id", "process_exit_verified", "restart_required"):
            if key not in value["result"]:
                continue
            item = value["result"][key]
            if not (item is None or type(item) is bool or isinstance(item, str) and len(item) <= 1024):
                raise ValueError("Status summary exceeds its scalar bounds")
            result[key] = item
        if not isinstance(result.get("state"), str) or not result["state"]:
            raise ValueError("Status has no state")
        return {"available": True, "device_id": value["device_id"], "result": result,
                "status_written_at": value["observed_at"], "status_bytes": len(raw),
                "status_sha256": hashlib.sha256(raw).hexdigest(),
                "qualification_passed": False}
    except FileNotFoundError:
        return {"available": False, "reason": "not_yet_published", "qualification_passed": False}
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
        return {"available": False, "reason": "invalid_or_unreadable",
                "error_type": type(exc).__name__, "qualification_passed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--architecture", choices=ARCHES, required=True)
    parser.add_argument("--kit-sha256", required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--cases-sha256")
    args = parser.parse_args()
    first_lineage = first_lineage_profile.validate_selection(args.cases_sha256, args.architecture)
    import boto3
    client = boto3.client("s3", region_name=args.region)
    OUTPUT.mkdir(mode=0o700, exist_ok=False)
    prefix = args.prefix + "/outputs/" + args.architecture
    receipt = {"schema": "organism-native-profile-host-v1",
               "started_at": datetime.now(timezone.utc).isoformat(), "architecture": args.architecture,
               "deadline": args.deadline, "qualification_passed": False, "passed": False}
    profile = None
    libraries = []
    process = None
    driver_started = False
    samples = []
    if first_lineage:
        receipt["first_lineage"] = first_lineage_profile.descriptor()
    minimum_remaining = first_lineage_profile.MINIMUM_REMAINING_SECONDS if first_lineage else 1700
    observer_seconds = first_lineage_profile.OBSERVER_SECONDS if first_lineage else 1550
    maximum_samples = first_lineage_profile.MAXIMUM_SAMPLES if first_lineage else 400

    def save():
        write_json(OUTPUT / "host-receipt.json", receipt)

    try:
        receipt["instance_id"] = instance_identity()
        receipt["permit_wait"] = {}
        receipt["launch_permit"] = wait_permit(client, args, receipt["instance_id"], receipt["permit_wait"])
        profile = HostProfile(receipt["launch_permit"]["volumes"], receipt=receipt, save=save)
        profile.prepare_storage()
        if first_lineage:
            receipt["owner_evaluator_sources"] = first_lineage_profile.restrict_owner_sources(
                Path(__file__).resolve().parents[1])
        catalog, libraries = prepare(client, args)
        receipt["release_id"] = catalog["release_id"]
        profile.prepare_network()
        profile.command(runner(libraries, "provision", args.architecture,
                               cases_sha256=args.cases_sha256), timeout=180)
        config = owner_config()
        receiver_options = {"runtime_seconds": first_lineage_profile.DRIVER_SECONDS + 60} if first_lineage else {}
        profile.start_receiver(
            config, config_path=CONFIG, python=BOOT / "python/bin/python3.12",
            entrypoint=Path(__file__).with_name("receiver_entry.py"), libraries=libraries, **receiver_options)
        if remaining(args.deadline) < minimum_remaining:
            raise TimeoutError("Insufficient fixed lease time remains for the complete native test")
        with (OUTPUT / "driver.log").open("x") as stream:
            process = subprocess.Popen(driver_command(libraries, args.architecture,
                                                      cases_sha256=args.cases_sha256),
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            driver_started = True
            launched = time.monotonic()
            last_upload = 0
            while process.poll() is None:
                if time.monotonic() - launched > observer_seconds or remaining(args.deadline) <= 120:
                    raise TimeoutError("Native observer reached its fixed execution bound")
                samples.append(sample())
                if "ready_census" not in receipt:
                    census = profile_census.serve_ready(
                        profile.host(), OUTPUT, release_id=catalog["release_id"],
                        allowed_devices=[receipt["disk_plan"][role]["device"] for role in ("data", "control")])
                    if census is not None:
                        receipt["ready_census"] = {
                            "passed": census["passed"],
                            "guard_exit_verified": census.get("guard_exit_verified", False),
                            "thaw_verified": census.get("thaw_verified", False),
                            "all_process_writes_contained": False,
                        }
                        save()
                if first_lineage:
                    for ordinal in (2, 3):
                        label = "cold-start-" + str(ordinal).zfill(2)
                        if label not in receipt:
                            census = profile_census.serve_ready(
                                profile.host(), OUTPUT / label, release_id=catalog["release_id"],
                                allowed_devices=[receipt["disk_plan"][role]["device"]
                                                 for role in ("data", "control")])
                            if census is not None:
                                receipt[label] = {"passed": census["passed"],
                                                  "guard_exit_verified": census.get("guard_exit_verified", False),
                                                  "thaw_verified": census.get("thaw_verified", False)}
                                save()
                if len(samples) > maximum_samples:
                    raise ValueError("Native resource sample count exceeded its bound")
                if time.monotonic() - last_upload >= 60:
                    client.put_object(Bucket=args.bucket, Key=prefix + "/progress.json",
                                      Body=json.dumps({"instance_id": receipt["instance_id"],
                                                       "sample": samples[-1],
                                                       "receiver_status": receiver_progress(config["device_id"]),
                                                       "qualification_passed": False}).encode())
                    last_upload = time.monotonic()
                time.sleep(5)
            receipt["driver_return_code"] = process.returncode
        result = json.loads((OUTPUT / "drive-receipt.json").read_bytes())
        receipt["passed"] = (process.returncode == 0 and result.get("passed") is True
                             and receipt.get("ready_census", {}).get("passed") is True
                             and result.get("owner_census", {}).get("passed") is True)
        if first_lineage:
            receipt["passed"] = bool(
                receipt["passed"] and result.get("cold_starts_verified") is True
                and result.get("temperature_tasks", {}).get("tasks_passed") is True
                and all(receipt.get("cold-start-" + str(n).zfill(2), {}).get("passed") is True for n in (2, 3)))
    except BaseException as exc:
        receipt["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        if profile is not None:
            try:
                receipt["workload_cleanup"] = cleanup_workloads(
                    profile, libraries, driver_started=driver_started,
                    maximum_starts=first_lineage_profile.COLD_STARTS if first_lineage else 1)
                if not receipt["workload_cleanup"]["passed"]:
                    raise RuntimeError("An owned workload still has a process")
                receipt["evidence_copy"] = copy_evidence()
            except Exception as exc:
                receipt["cleanup_or_evidence_failure"] = type(exc).__name__ + ": " + str(exc)
                receipt["passed"] = False
            try:
                receipt["profile_cleanup"] = profile.remove()
                if receipt["profile_cleanup"].get("passed") is not True:
                    raise RuntimeError("Native profile cleanup was not verified")
            except Exception as exc:
                receipt["profile_cleanup_failure"] = type(exc).__name__ + ": " + str(exc)
                receipt["passed"] = False
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                    receipt["observer_forced_stop"] = True
                    receipt["passed"] = False
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(OUTPUT / "resource-samples.json", samples)
        save()
        # Publish the small terminal receipt before the bounded archive.
        client.put_object(Bucket=args.bucket, Key=prefix + "/host-receipt.json",
                          Body=(OUTPUT / "host-receipt.json").read_bytes())
        target = OUTPUT.parent / "organism-profile-results.tar.gz"
        bounded_results(OUTPUT, target)
        client.upload_file(str(target), args.bucket, prefix + "/results.tar.gz")
        client.put_object(Bucket=args.bucket, Key=prefix + "/results-identity.json",
                          Body=json.dumps({"bytes": target.stat().st_size, "sha256": digest(target)}).encode())
    return 0 if receipt["passed"] and not receipt.get("failure") else 1


if __name__ == "__main__":
    raise SystemExit(main())
