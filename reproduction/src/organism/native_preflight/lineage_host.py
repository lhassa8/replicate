"""Dedicated owner host for availability, observation, and fixed-lease cleanup.

Only the seed receives a full package from owner storage. Other receivers are
given their private enrollment and the signed portable receiving substrate.
Their application can be established only through the existing peer service.
"""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .. import device_storage
from ..runtime_environment import library_environment
from ..training_inputs import require
from .common import digest, extract, write_json
from .host import bounded_results, download
from . import first_lineage_profile, lineage_control, lineage_enrollment, lineage_network
from . import lineage_permit, profile_census
from .profile_host import (BOOT, CONFIG, INPUT, OUTPUT, cleanup_workloads, copy_evidence,
                           owner_config, receiver_progress, sample)
from .profile_permit import instance_identity, remaining
from .profile_setup import HostProfile
from .trained_inputs import validate_catalog
from .verify_libraries import verify_carried


def read_small(client, bucket, key, maximum):
    response = client.get_object(Bucket=bucket, Key=key)
    try:
        require(0 < response["ContentLength"] <= maximum, "Owner input exceeds its bound")
        raw = response["Body"].read(maximum + 1)
        require(len(raw) == response["ContentLength"], "Owner input was truncated")
        return raw
    finally:
        response["Body"].close()


def receiver_inputs(catalog):
    native = catalog["platforms"]["linux-x86_64"]
    names = {native[role] for role in ("python", "dependencies", "inference")}
    require(len(names) == 3, "Portable receiving substrate must contain three separate archives")
    return {name: catalog["files"][name] for name in sorted(names)}


def prepare(client, args):
    INPUT.mkdir(mode=0o700)
    raw = read_small(client, args.bucket, args.prefix + "/inputs/catalog.json", 1024**2)
    require(hashlib.sha256(raw).hexdigest() == args.catalog_sha256, "Lineage catalog changed")
    catalog = validate_catalog(json.loads(raw))
    (INPUT / "catalog.json").write_bytes(raw)
    (OUTPUT / "input-catalog.json").write_bytes(raw)
    download(client, args.bucket, args.prefix + "/inputs/" + first_lineage_profile.CASES_NAME,
             OUTPUT / first_lineage_profile.CASES_NAME, first_lineage_profile.descriptor()["cases"])
    (OUTPUT / first_lineage_profile.CASES_NAME).chmod(0o600)
    first_lineage_profile.validate_cases(OUTPUT / first_lineage_profile.CASES_NAME)
    bundle_raw = read_small(client, args.bucket, args.prefix + "/inputs/nodes/" + args.slot + ".tar.gz",
                            lineage_enrollment.MAX_BUNDLE)
    require(hashlib.sha256(bundle_raw).hexdigest() == args.bundle_sha256,
            "This device's private enrollment bundle changed")
    archive = INPUT / "node.tar.gz"
    lineage_enrollment.private_write(archive, bundle_raw)
    lineage_enrollment.extract_bundle(archive, INPUT / "node")
    lineage_enrollment.verify(INPUT / "node", args.slot, catalog)
    archive.unlink()
    receiver = INPUT / "receiver"
    receiver.mkdir()
    entries = receiver_inputs(catalog)
    for name, entry in entries.items():
        target = receiver / name
        target.parent.mkdir(parents=True, exist_ok=True)
        download(client, args.bucket, args.prefix + "/inputs/receiver/" + name, target, entry)
    BOOT.mkdir()
    native = catalog["platforms"]["linux-x86_64"]
    extract(receiver / native["python"], BOOT)
    extract(receiver / native["dependencies"], BOOT)
    extract(receiver / native["inference"], BOOT / "tools/llama.cpp/b9850")
    libraries = verify_carried(BOOT, OUTPUT, "x86_64")
    if args.slot == "seed":
        capsule = INPUT / "capsule"
        capsule.mkdir()
        for name, entry in catalog["files"].items():
            require(remaining(args.deadline) > 300, "Insufficient time for seed preparation")
            target = capsule / name
            target.parent.mkdir(parents=True, exist_ok=True)
            download(client, args.bucket, args.prefix + "/inputs/capsule/" + name, target, entry)
        (capsule / ".manifest.json").write_bytes(base64.b64decode(catalog["manifest_base64"], validate=True))
    return catalog, libraries, entries


def runner(libraries, args, operation, *, start_epoch=None):
    require(operation in ("provision", "enable", "seed", "observe", "endurance"),
            "Unexpected owner operation")
    environment = {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/nonexistent", "LC_ALL": "C.UTF-8",
        "AWS_EC2_METADATA_DISABLED": "true", "OTEL_SDK_DISABLED": "true",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        **library_environment(libraries),
    }
    destination = OUTPUT / operation
    if operation in ("provision", "enable", "seed"):
        destination = OUTPUT / (operation + "-receipt.json")
    command = [
        "/usr/bin/env", "-i", *(key + "=" + value for key, value in environment.items()),
        str(BOOT / "python/bin/python3.12"), "-I", "-B",
        str(Path(__file__).with_name("lineage_entry.py")), "--operation", operation,
        "--slot", args.slot, "--catalog", str(INPUT / "catalog.json"), "--output", str(destination),
    ]
    if operation == "provision":
        command += ["--bundle", str(INPUT / "node")]
    elif operation == "seed":
        require(args.slot == "seed", "Owner package delivery is restricted to the seed")
        command += ["--capsule", str(INPUT / "capsule")]
    elif operation in ("observe", "endurance"):
        command += ["--cases", str(OUTPUT / first_lineage_profile.CASES_NAME)]
    if operation == "endurance":
        require(args.slot != "seed" and start_epoch is not None,
                "Endurance requires a descendant and a common clock")
        command += ["--ready", str(OUTPUT / "observe"), "--start-epoch", str(start_epoch)]
    return command


def driver_command(libraries, args, operation, *, start_epoch=None):
    seconds = {"seed": 1230, "observe": 5160, "endurance": 7530}[operation]
    require(remaining(args.deadline) > seconds + 120, "Original lease cannot contain this observation")
    properties = {
        "User": "root", "Group": "root", "UMask": "0077", "KillMode": "control-group",
        "MemoryAccounting": "yes", "MemoryMax": "512M", "MemorySwapMax": "0",
        "CPUQuota": "100%", "TasksMax": "64", "RuntimeMaxSec": str(seconds), "TimeoutStopSec": "20",
        "NetworkNamespacePath": "/run/netns/" + (
            lineage_network.BOOTSTRAP_NAMESPACE if operation == "seed" else lineage_network.DEVICE_NAMESPACE),
    }
    return [
        "systemd-run", "--unit=organism-lineage-" + operation + ".service",
        "--wait", "--pipe", "--service-type=exec",
        *["--property=" + key + "=" + value for key, value in properties.items()],
        *runner(libraries, args, operation, start_epoch=start_epoch),
    ]


def main():
    parser = argparse.ArgumentParser()
    for name in ("bucket", "prefix", "region", "architecture", "slot", "kit-sha256",
                 "catalog-sha256", "cases-sha256", "bundle-sha256", "deadline"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    require(os.geteuid() == 0, "Only the dedicated native owner may run this controller")
    first_lineage_profile.validate_selection(args.cases_sha256, args.architecture)
    lineage_network.slot_value(args.slot)
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    client = boto3.client("s3", region_name=args.region,
                          config=Config(connect_timeout=5, read_timeout=15, retries={"total_max_attempts": 2}))
    OUTPUT.mkdir(mode=0o700, exist_ok=False)
    prefix = args.prefix + "/outputs/" + args.slot + "/"
    receipt = {
        "schema": "organism-native-lineage-host-v1", "slot": args.slot, "architecture": args.architecture,
        "deadline": args.deadline, "started_at": datetime.now(timezone.utc).isoformat(),
        "controls": [], "operations": [], "failure": None, "passed": False,
        "descendant_launch_performed": False, "qualification_passed": False,
    }
    profile, libraries, config = None, [], None
    samples, drivers = [], []
    process = None
    last_upload = 0

    def save():
        write_json(OUTPUT / "host-receipt.json", receipt)

    def tick():
        nonlocal last_upload
        require(remaining(args.deadline) > 120, "Fixed lineage observation lease ended")
        samples.append(sample())
        require(len(samples) <= 3000, "Lineage resource sample count exceeds its four-hour bound")
        directory = OUTPUT / "observe"
        if "ready_census" not in receipt and config is not None:
            census = profile_census.serve_ready(
                profile.host(), directory, release_id=receipt["release_id"],
                allowed_devices=[receipt["disk_plan"][role]["device"] for role in ("data", "control")])
            if census is not None:
                receipt["ready_census"] = {
                    "passed": census["passed"], "guard_exit_verified": census.get("guard_exit_verified", False),
                    "thaw_verified": census.get("thaw_verified", False),
                }
                save()
        if time.monotonic() - last_upload >= 30:
            value = {
                "observed_at": datetime.now(timezone.utc).isoformat(), "instance_id": receipt["instance_id"],
                "slot": args.slot, "stage": receipt.get("stage"), "sample": samples[-1],
                "receiver_status": receiver_progress(config["device_id"]) if config else None,
                "qualification_passed": False,
            }
            client.put_object(Bucket=args.bucket, Key=prefix + "progress.json",
                              Body=json.dumps(value).encode(), ServerSideEncryption="AES256")
            last_upload = time.monotonic()

    def drive(operation, *, start_epoch=None):
        nonlocal process
        receipt["stage"] = operation
        argv = driver_command(libraries, args, operation, start_epoch=start_epoch)
        unit = "organism-lineage-" + operation + ".service"
        drivers.append(unit)
        record = {"operation": operation, "argv": argv, "started_at": datetime.now(timezone.utc).isoformat()}
        receipt["operations"].append(record)
        save()
        with (OUTPUT / (operation + "-driver.log")).open("x") as stream:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                tick()
                time.sleep(5)
            record.update(return_code=process.returncode, completed_at=datetime.now(timezone.utc).isoformat())
        require(process.returncode == 0, "Lineage " + operation + " failed")
        process = None
        save()

    def wait_control(sequence, *, enabled, observed):
        receipt["stage"] = "awaiting-control-" + str(sequence)
        save()
        while True:
            tick()
            try:
                raw = read_small(client, args.bucket, args.prefix + "/inputs/controls/"
                                 + args.slot + "/" + str(sequence).zfill(2) + ".json", 16384)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404", "AccessDenied", "403"):
                    raise
                time.sleep(5)
                continue
            value = lineage_control.validate(
                json.loads(raw), args, receipt["launch_permit"], receipt["release_id"],
                sequence=sequence, enabled=enabled, observed=observed)
            record = {"value": value, "sha256": hashlib.sha256(raw).hexdigest(),
                      "observed_at": datetime.now(timezone.utc).isoformat()}
            receipt["controls"].append(record)
            write_json(OUTPUT / ("control-" + str(sequence).zfill(2) + ".json"), record)
            save()
            return value

    try:
        receipt["instance_id"] = instance_identity()
        receipt["permit_wait"] = {}
        receipt["launch_permit"] = lineage_permit.wait(
            client, args, receipt["instance_id"], receipt["permit_wait"])
        profile = HostProfile(receipt["launch_permit"]["volumes"], receipt=receipt, save=save)
        profile.prepare_storage()
        receipt["owner_evaluator_sources"] = first_lineage_profile.restrict_owner_sources(
            Path(__file__).resolve().parents[1])
        catalog, libraries, entries = prepare(client, args)
        receipt["release_id"] = catalog["release_id"]
        receipt["receiving_substrate"] = entries
        receipt["owner_full_package_provided"] = args.slot == "seed"
        lineage_network.configure(profile, args.slot)
        profile.command(runner(libraries, args, "provision"), timeout=180)
        # The receiving service now owns its one private identity in control storage.
        shutil.rmtree(INPUT / "node")
        config = owner_config()
        profile.start_receiver(
            config, config_path=CONFIG, python=BOOT / "python/bin/python3.12",
            entrypoint=Path(__file__).with_name("receiver_entry.py"), libraries=libraries,
            runtime_seconds=max(60, int(remaining(args.deadline) - 120)))
        wait_control(1, enabled=False, observed=False)
        profile.command(runner(libraries, args, "enable"), timeout=60)
        if args.slot == "seed":
            drive("seed")
        drive("observe")
        observed = json.loads((OUTPUT / "observe/observation.json").read_bytes())
        require(observed["tasks_passed"] is True and receipt["ready_census"]["passed"] is True,
                "The individual did not pass its task or owner census checks")
        ready_bytes = (OUTPUT / "observe/observation.json").read_bytes()
        client.put_object(Bucket=args.bucket, Key=prefix + "observation.json", Body=ready_bytes,
                          ServerSideEncryption="AES256", ChecksumAlgorithm="SHA256")
        control = wait_control(2, enabled=True, observed=True)
        if control["action"] == "endurance":
            drive("endurance", start_epoch=control["start_epoch"])
            from .lineage_endurance import audit
            receipt["endurance"] = audit(
                OUTPUT / "endurance", binding=observed["ready"]["binding"], start_epoch=control["start_epoch"])
        receipt["passed"] = True
    except BaseException as exc:
        receipt["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        if profile is not None:
            try:
                receipt["driver_cleanup"] = [profile.stop_unit(unit, shared=False) for unit in reversed(drivers)]
                require(all(row["absence_verified"] for row in receipt["driver_cleanup"]),
                        "An owner observation driver remains alive")
                receipt["workload_cleanup"] = cleanup_workloads(profile, libraries, driver_started=False)
                require(receipt["workload_cleanup"]["passed"], "An individual remains alive after retirement")
                receipt["evidence_copy"] = copy_evidence()
            except Exception as exc:
                receipt["cleanup_or_evidence_failure"] = type(exc).__name__ + ": " + str(exc)
                receipt["passed"] = False
            try:
                receipt["profile_cleanup"] = profile.remove()
                receipt["root_link_cleanup"] = lineage_network.remove_root_links(profile)
                require(receipt["profile_cleanup"]["passed"], "Native profile cleanup failed")
            except Exception as exc:
                receipt["profile_cleanup_failure"] = type(exc).__name__ + ": " + str(exc)
                receipt["passed"] = False
        if process is not None:
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
        client.put_object(Bucket=args.bucket, Key=prefix + "host-receipt.json",
                          Body=(OUTPUT / "host-receipt.json").read_bytes(), ServerSideEncryption="AES256")
        target = OUTPUT.parent / "organism-lineage-results.tar.gz"
        bounded_results(OUTPUT, target)
        client.upload_file(str(target), args.bucket, prefix + "results.tar.gz",
                           ExtraArgs={"ServerSideEncryption": "AES256"})
        client.put_object(Bucket=args.bucket, Key=prefix + "results-identity.json",
                          Body=json.dumps({"bytes": target.stat().st_size, "sha256": digest(target)}).encode(),
                          ServerSideEncryption="AES256")
    return 0 if receipt["passed"] and receipt["failure"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
