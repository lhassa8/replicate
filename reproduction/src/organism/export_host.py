"""Root-owned preparation and result transfer for one disposable CPU export host."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import stat
import subprocess
import time
import tarfile

from .export_worker import (LIMITS, STANDARD_PROFILE, limits_for_profile, lease_envelope,
                            profile_for_job, profile_for_training, package_results, run_stage)
from .gguf_export import canonical, identity
from .training_inputs import load, require, verified_path
from .training_worker.host import download_base, instance_identity, read_small, remaining


ROOT = Path("/srv/organism-export")
WORKSPACE, RESULTS, VENV = ROOT / "workspace", ROOT / "results", ROOT / "venv"
BASE, TOOLS, TRAINING = ROOT / "base", ROOT / "tools", ROOT / "training-results"
UNIT = "organism-export-work.service"
CLEANUP_SECONDS = 900


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checked_input(workspace):
    workspace = Path(workspace)
    manifest = load(workspace / "export-input.json")
    profile = profile_for_job(manifest)
    require(manifest.get("schema") == "organism-cpu-export-input-v1"
            and canonical(manifest.get("limits")) == canonical(limits_for_profile(profile)),
            "Unknown or changed CPU export input")
    entries = manifest["files"]
    require(isinstance(entries, list) and 1 <= len(entries) <= 128
            and len({entry["path"] for entry in entries}) == len(entries), "Invalid CPU export source inventory")
    files = {entry["path"]: verified_path(workspace, entry) for entry in entries}
    require({"worker-input.json", "export-job.json", manifest["base_metadata"]} <= set(files),
            "Export control inputs are not bound")
    require(sum(entry["bytes"] for entry in entries) <= 256 * 1024**2, "Export source inventory exceeds its bound")
    training = load(files["worker-input.json"])
    require(training.get("schema") == "organism-training-worker-input-v1" and training.get("mode") == "train"
            and {training["recipe"], training["contract"]} <= set(files),
            "The original admitted training controls are missing")
    by_name = {entry["path"]: entry for entry in entries}
    require(all(by_name.get(entry["path"]) == entry for entry in training["files"]),
            "An original training input was replaced")
    recipe = load(files[training["recipe"]])
    require(profile == profile_for_training(recipe),
            "CPU export input profile differs from its bound training corpus")
    require(manifest["base_metadata"] == recipe["inputs"]["base_metadata"]["path"]
            and recipe["inputs"]["dependency_lock"]["path"] in files,
            "CPU preparation differs from the original model and dependency inputs")
    for key, maximum in (("tools", 200 * 1024**2), ("training_results", 2 * 1024**3)):
        entry = manifest["archives"][key]
        require(set(entry) == {"bytes", "sha256"} and type(entry["bytes"]) is int
                and 0 < entry["bytes"] <= maximum and re.fullmatch("[0-9a-f]{64}", entry["sha256"]),
                "Invalid bound export input archive")
    return manifest


def validate_permit(permit, args, instance_id, *, worker_profile=STANDARD_PROFILE):
    limits_for_profile(worker_profile)
    require(permit.get("schema") == "organism-cpu-export-permit-v1"
            and permit.get("instance_id") == instance_id and permit.get("prefix") == args.prefix
            and permit.get("deadline") == args.deadline and permit.get("kit_sha256") == args.kit_sha256,
            "CPU export permit does not bind this exact instance, kit and expiry")
    require(permit.get("independent_expiry_configuration_verified") is True
            and isinstance(permit.get("schedule_arn"), str)
            and re.fullmatch(r"arn:aws:scheduler:us-east-1:[0-9]{12}:schedule/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
                             permit["schedule_arn"])
            and isinstance(permit.get("approval_reference"), str) and permit["approval_reference"].strip(),
            "CPU export scope approval and independent expiry are required")
    require(permit.get("worker_profile", STANDARD_PROFILE) == worker_profile,
            "CPU export permit changed its worker profile")


def wait_permit(client, args, instance_id, evidence, *, worker_profile=STANDARD_PROFILE):
    limits = limits_for_profile(worker_profile)
    started = time.monotonic()
    evidence.update(attempts=0, unavailable_responses=0, verified=False)
    while time.monotonic() - started < 600 and remaining(args.deadline) > limits["workload_seconds"] + CLEANUP_SECONDS:
        evidence["attempts"] += 1
        try:
            permit = read_small(client, args.bucket, args.prefix + "/inputs/launch-permit.json")
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code not in ("NoSuchKey", "404", "AccessDenied", "403"):
                raise
            evidence["unavailable_responses"] += 1
            time.sleep(min(5, max(0, 600 - (time.monotonic() - started))))
            continue
        require(time.monotonic() - started < 600
                and remaining(args.deadline) > limits["workload_seconds"] + CLEANUP_SECONDS,
                "CPU export permit arrived after its admission window")
        validate_permit(permit, args, instance_id, worker_profile=worker_profile)
        evidence.update(verified=True, seconds=round(time.monotonic() - started, 3))
        return permit
    raise TimeoutError("CPU export launch permit was unavailable within its admission window")


def download_archive(client, bucket, key, path, expected, deadline, *, worker_profile=STANDARD_PROFILE):
    limits = limits_for_profile(worker_profile)
    response = client.get_object(Bucket=bucket, Key=key)
    digest, total = hashlib.sha256(), 0
    try:
        require(response["ContentLength"] == expected["bytes"], "Export archive length differs")
        with Path(path).open("xb") as stream:
            while block := response["Body"].read(1024**2):
                require(remaining(deadline) > limits["workload_seconds"] + CLEANUP_SECONDS,
                        "Export input download exhausted the preparation window")
                total += len(block)
                require(total <= expected["bytes"], "Export archive exceeds declared size")
                stream.write(block)
                digest.update(block)
        require(total == expected["bytes"] and digest.hexdigest() == expected["sha256"],
                "Export archive identity differs")
    finally:
        response["Body"].close()
    return {"bytes": total, "sha256": digest.hexdigest()}


def extract_archive(path, target, *, tools=False, source=False):
    require(not (tools and source), "Ambiguous export archive profile")
    maximum = 200 * 1024**2 if tools else 256 * 1024**2 if source else 2 * 1024**3
    member_limit = 256 if source else 512
    require(not Path(target).exists(), "Export extraction target must be new")
    with tarfile.open(path, "r:gz") as archive:
        members, names, total = [], set(), 0
        for item in archive:
            name = PurePosixPath(item.name)
            require(name.parts and not name.is_absolute() and ".." not in name.parts
                    and "\\" not in item.name and str(name) == item.name.rstrip("/")
                    and str(name) not in names, "Invalid or duplicate export archive path")
            names.add(str(name))
            require(item.isfile() or tools and (item.isdir() or item.issym()),
                    "Export archive contains an unexpected file type")
            if item.issym():
                link = PurePosixPath(item.linkname)
                require(len(link.parts) == 1 and link.name not in (".", "..")
                        and not link.is_absolute() and "\\" not in item.linkname,
                        "Export tool alias escapes its directory")
            total += item.size
            require(0 <= item.size <= maximum and total <= maximum and len(members) < member_limit,
                    "Export archive extraction exceeds its bound")
            members.append(item)
        Path(target).mkdir()
        archive.extractall(target, members=members, filter="data")
    return {"members": len(members), "expanded_bytes": total}


def service_text(*, worker_profile=STANDARD_PROFILE):
    limits = limits_for_profile(worker_profile)
    wrapper = ("import sys,runpy;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[1:];"
               "runpy.run_module('organism.export_worker',run_name='__main__')")
    command = [
        "/usr/bin/env", "-i", "PATH=" + str(VENV / "bin") + ":/usr/bin:/bin",
        "LC_ALL=C.UTF-8", "PYTHONDONTWRITEBYTECODE=1", "AWS_EC2_METADATA_DISABLED=true",
        "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "CUDA_VISIBLE_DEVICES=",
        "XDG_CACHE_HOME=" + str(RESULTS / "cache"), "HOME=" + str(RESULTS / "cache"),
        str(VENV / "bin/python"), "-I", "-c",
        '"' + wrapper + '"',
        str(WORKSPACE), "--workspace", str(WORKSPACE), "--worker-output", str(TRAINING / "workload"),
        "--base", str(BASE), "--tools-root", str(TOOLS), "--output", str(RESULTS / "workload"),
        "--recipe", str(WORKSPACE / "export-job.json"), "--host-preparation", str(ROOT / "host-preparation.json"),
    ]
    return "\n".join([
        "[Unit]", "Description=Bounded offline organism CPU export", "[Service]", "Type=exec",
        "User=organism-export", "Group=organism-export", "PrivateNetwork=yes", "PrivateDevices=yes",
        "NoNewPrivileges=yes", "ProtectSystem=strict", "ProtectHome=yes", "PrivateTmp=yes",
        "ProtectKernelTunables=yes", "ProtectKernelModules=yes", "ProtectControlGroups=yes",
        "RestrictSUIDSGID=yes", "CapabilityBoundingSet=", "UMask=0077", "TasksMax=512",
        "ReadWritePaths=" + str(ROOT), "MemoryAccounting=yes", "MemoryMax=56G",
        "MemorySwapMax=0", "CPUQuota=800%", "RuntimeMaxSec=" + str(limits["workload_seconds"]), "TimeoutStopSec=30",
        "KillMode=control-group", "StandardOutput=append:" + str(RESULTS / "service.log"),
        "StandardError=inherit", "ExecStart=" + " ".join(command), "",
    ])


def service_state():
    result = subprocess.run(
        ["systemctl", "show", UNIT,
         "--property=LoadState,ActiveState,SubState,Result,ExecMainStatus,MainPID,ControlPID,ControlGroup,MemoryPeak,MemoryMax,MemorySwapMax"],
        capture_output=True, text=True, timeout=15)
    return result.returncode, dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def stop_service():
    stopped = subprocess.run(["systemctl", "stop", UNIT], capture_output=True, text=True, timeout=45)
    code, fields = service_state()
    absent = fields.get("LoadState") == "not-found"
    confirmed = ((absent or code == 0 and stopped.returncode == 0)
                 and fields.get("ActiveState") in ("inactive", "failed")
                 and fields.get("MainPID") == "0" and fields.get("ControlPID") == "0")
    group = fields.get("ControlGroup", "")
    if group:
        relative = PurePosixPath(group.lstrip("/"))
        require(".." not in relative.parts, "Invalid export service cgroup")
        events = Path("/sys/fs/cgroup") / relative / "cgroup.events"
        if events.exists():
            populated = dict(line.split() for line in events.read_text().splitlines()).get("populated")
            confirmed = confirmed and populated == "0"
    return {"stop_return_code": stopped.returncode, "service_state": fields,
            "service_stopped_confirmed": confirmed}


def run_service(deadline, progress, *, worker_profile=STANDARD_PROFILE):
    limits = limits_for_profile(worker_profile)
    require(remaining(deadline) >= limits["workload_seconds"] + CLEANUP_SECONDS,
            "Insufficient lifetime remains for CPU export and result transfer")
    (Path("/etc/systemd/system") / UNIT).write_text(service_text(worker_profile=worker_profile))
    subprocess.run(["systemctl", "daemon-reload"], check=True, timeout=30)
    subprocess.run(["systemctl", "start", "--no-block", UNIT], check=True, timeout=30)
    started, last_progress = time.monotonic(), 0
    while time.monotonic() - started < limits["workload_seconds"] + 60:
        try:
            code, fields = service_state()
        except Exception as exc:
            code, fields = -1, {"observation_failure": type(exc).__name__ + ": " + str(exc)}
        if time.monotonic() - last_progress >= 60:
            progress(fields)
            last_progress = time.monotonic()
        if code == 0 and fields.get("ActiveState") in ("inactive", "failed"):
            return fields
        time.sleep(5)
    raise TimeoutError("CPU export service exceeded its observation deadline")


def freeze_tree(directory):
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            continue
        path.chmod(0o555 if path.is_dir() or path.stat().st_mode & 0o111 else 0o444)
    Path(directory).chmod(0o555)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bucket", "prefix", "deadline", "kit-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    require(os.geteuid() == 0 and re.fullmatch(r"export-[0-9]{8}-[a-z0-9]{4,12}", args.prefix)
            and re.fullmatch("[0-9a-f]{64}", args.kit_sha256)
            and 0 < remaining(args.deadline) <= 21600, "Invalid disposable CPU export context")
    import boto3
    from botocore.config import Config
    client = boto3.client("s3", region_name="us-east-1",
                          config=Config(connect_timeout=10, read_timeout=30, retries={"total_max_attempts": 3}))
    RESULTS.mkdir(exist_ok=False)
    receipt = {"schema": "organism-cpu-export-host-v1", "started_at": datetime.now(timezone.utc).isoformat(),
               "deadline": args.deadline, "kit_sha256": args.kit_sha256,
               "failure": None, "preparation_passed": False, "qualification_passed": False}
    prefix = args.prefix + "/outputs/"

    def progress(fields):
        value = {"observed_at": datetime.now(timezone.utc).isoformat(),
                 "instance_id": receipt["instance_id"], "service": fields, "qualification_passed": False}
        save(ROOT / "progress.json", value)
        try:
            client.put_object(Bucket=args.bucket, Key=prefix + "progress.json", Body=canonical(value))
        except Exception as exc:
            receipt["progress_upload_failures"] = receipt.get("progress_upload_failures", 0) + 1
            receipt["last_progress_upload_failure"] = type(exc).__name__ + ": " + str(exc)
    try:
        manifest = checked_input(WORKSPACE)
        profile = profile_for_job(manifest)
        limits = limits_for_profile(profile)
        require(remaining(args.deadline) <= lease_envelope(profile)["maximum_lease_seconds"],
                "CPU export expiry exceeds the selected worker profile")
        receipt["worker_profile"] = profile
        receipt["instance_id"] = instance_identity()
        receipt["permit_wait"] = {}
        receipt["launch_permit"] = wait_permit(
            client, args, receipt["instance_id"], receipt["permit_wait"], worker_profile=profile)
        receipt["input_archives"] = {}
        for key, filename, target in (("tools", "tools.tar.gz", TOOLS),
                                      ("training_results", "training-results.tar.gz", TRAINING)):
            path = ROOT / filename
            receipt["input_archives"][key] = download_archive(
                client, args.bucket, args.prefix + "/inputs/" + filename, path, manifest["archives"][key],
                args.deadline, worker_profile=profile)
            receipt["input_archives"][key]["extraction"] = extract_archive(path, target, tools=key == "tools")
        subprocess.run(["python3", "-m", "venv", str(VENV)], check=True, timeout=120)
        training_input = load(WORKSPACE / "worker-input.json")
        recipe = load(WORKSPACE / training_input["recipe"])
        lock = WORKSPACE / recipe["inputs"]["dependency_lock"]["path"]
        install_seconds = min(1800, remaining(args.deadline) - limits["workload_seconds"] - CLEANUP_SECONDS)
        require(install_seconds > 0, "Insufficient CPU export preparation time remains")
        receipt["dependency_install"] = run_stage(
            [str(VENV / "bin/python"), "-m", "pip", "install", "--no-cache-dir", "--only-binary=:all:",
             "--require-hashes", "-r", str(lock)], RESULTS / "dependency-install.log", seconds=install_seconds,
            environment={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
        metadata = load(WORKSPACE / manifest["base_metadata"])
        receipt["native_downloads"] = download_base(metadata, BASE, args.deadline)
        account = pwd.getpwnam("organism-export")
        os.chown(RESULTS, account.pw_uid, account.pw_gid)
        RESULTS.chmod(0o700)
        cache = RESULTS / "cache"
        cache.mkdir()
        os.chown(cache, account.pw_uid, account.pw_gid)
        for path in BASE.iterdir():
            os.chown(path, account.pw_uid, account.pw_gid)
            path.chmod(0o444)
        BASE.chmod(0o555)
        for path in (WORKSPACE, TOOLS, TRAINING):
            freeze_tree(path)
        save(ROOT / "host-preparation.json", {"host_network_namespace": os.readlink("/proc/self/ns/net"),
                                             "instance_id": receipt["instance_id"], "kit_sha256": args.kit_sha256})
        receipt["preparation_passed"] = True
        receipt["service_result"] = run_service(args.deadline, progress, worker_profile=profile)
        require(receipt["service_result"].get("ActiveState") == "inactive"
                and receipt["service_result"].get("Result") == "success"
                and receipt["service_result"].get("ExecMainStatus") == "0", "CPU export workload failed")
        for name, entry in metadata["files"].items():
            verified_path(BASE, {"path": name, "bytes": entry["bytes"], "sha256": entry["sha256"]})
        checked_input(WORKSPACE)
        receipt["source_inputs_unchanged"] = True
    except Exception as exc:
        receipt["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        try:
            receipt.update(stop_service())
            require(receipt["service_stopped_confirmed"], "CPU export service stop was not confirmed")
        except Exception as exc:
            receipt.update(service_stopped_confirmed=False, stop_failure=type(exc).__name__ + ": " + str(exc))
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        execution = ROOT / "host-execution.json"
        save(execution, receipt)
        client.put_object(Bucket=args.bucket, Key=prefix + "host-execution.json", Body=execution.read_bytes())
        transfer = {"schema": "organism-cpu-export-transfer-v1", "execution": identity(execution),
                    "failure": None, "qualification_passed": False}
        try:
            require(receipt.get("service_stopped_confirmed"), "Cannot package outputs of an unconfirmed live service")
            target = ROOT / "results.tar.gz"
            transfer["archive"] = package_results(RESULTS, target)
            client.upload_file(str(target), args.bucket, prefix + "results.tar.gz")
            client.put_object(Bucket=args.bucket, Key=prefix + "results-identity.json",
                              Body=canonical(transfer["archive"]))
        except Exception as exc:
            transfer["failure"] = type(exc).__name__ + ": " + str(exc)
        save(ROOT / "host-transfer.json", transfer)
        client.put_object(Bucket=args.bucket, Key=prefix + "host-transfer.json",
                          Body=(ROOT / "host-transfer.json").read_bytes())
    return 0 if receipt["failure"] is None and receipt.get("service_stopped_confirmed") and transfer["failure"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
