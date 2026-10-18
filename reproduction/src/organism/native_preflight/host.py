"""Prepare pinned artifacts, run the isolated service, and preserve receipts.

Executed as root only on disposable, explicitly authorized EC2 workers.
Instance credentials are confined to preparation/output transfer; the model
service has a private network namespace and cannot reach instance metadata.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import tarfile
import time

from .common import ARCHES, MAX_OUTPUT_BYTES, check_file, digest, extract, validate_catalog, write_json
from .verify_libraries import verify_carried


ROOT = Path("/opt/organism-native")
OUTPUT = Path("/var/lib/organism-native/output")
UNIT = "organism-native-probe"


def download(client, bucket, key, path, specification):
    response = client.get_object(Bucket=bucket, Key=key)
    try:
        if response["ContentLength"] != specification["bytes"]:
            raise ValueError("S3 input length differs from the pinned catalog")
        remaining = specification["bytes"]
        with path.open("xb") as target:
            while remaining:
                block = response["Body"].read(min(8 * 1024**2, remaining))
                if not block:
                    raise ValueError("Truncated input transfer")
                target.write(block)
                remaining -= len(block)
            if response["Body"].read(1):
                raise ValueError("Input exceeds declared size")
    finally:
        response["Body"].close()
    check_file(path, specification)


def prepare(client, args):
    ROOT.mkdir()
    incoming = ROOT / "inputs"
    incoming.mkdir()
    response = client.get_object(Bucket=args.bucket, Key=f"{args.prefix}/inputs/catalog.json")
    try:
        if response["ContentLength"] > 1024 * 1024:
            raise ValueError("Catalog exceeds bound")
        body = response["Body"].read(1024 * 1024 + 1)
    finally:
        response["Body"].close()
    import hashlib
    if hashlib.sha256(body).hexdigest() != args.catalog_sha256:
        raise ValueError("Catalog identity differs from the approved launch")
    catalog = validate_catalog(json.loads(body))
    (ROOT / "catalog.json").write_bytes(body)
    shutil.copyfile(ROOT / "catalog.json", OUTPUT / "input-catalog.json")
    # Carry both platform archives even though only the native one is executed.
    for specification in catalog["files"]:
        name = specification["name"]
        download(client, args.bucket, f"{args.prefix}/inputs/{name}", incoming / name, specification)
    native = catalog["platforms"][args.architecture]
    extract(incoming / native["python"], ROOT)
    extract(incoming / native["dependencies"], ROOT)
    runtime_parent = ROOT / "tools/llama.cpp/b9850"
    extract(incoming / native["inference"], runtime_parent)
    model_dir = ROOT / "models/granite-3b"
    model_dir.mkdir(parents=True)
    # Move the one verified model instead of creating another multi-GB copy.
    os.rename(incoming / catalog["model"], model_dir / "granite-base-Q4_K_M.gguf")
    model = next(entry for entry in catalog["files"] if entry["name"] == catalog["model"])
    write_json(model_dir / "manifest.json", {"files": [model]})
    search = verify_carried(ROOT, OUTPUT, args.architecture)
    write_json(ROOT / "preparation.json", {
        "host_network_namespace": os.readlink("/proc/self/ns/net"),
        "input_catalog_sha256": args.catalog_sha256,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "online_preparation_used": True, "qualification_passed": False,
    })
    # Preserve carried libraries and notices for review and later release assembly.
    with tarfile.open(OUTPUT / "supplemental-libraries.tar.gz", "w:gz") as archive:
        archive.add(ROOT / "supplemental", arcname="supplemental")
    return search


def service_command(architecture, library_paths):
    properties = {
        "User": "organism", "Group": "organism", "PrivateNetwork": "yes",
        "NoNewPrivileges": "yes", "ProtectHome": "yes", "ProtectSystem": "strict",
        "PrivateTmp": "yes", "PrivateDevices": "yes", "RestrictSUIDSGID": "yes",
        "CapabilityBoundingSet": "", "LockPersonality": "yes",
        "ReadWritePaths": str(OUTPUT), "UMask": "0077", "TasksMax": "128",
        "MemoryAccounting": "yes", "MemoryMax": "6G", "MemorySwapMax": "0",
        "CPUQuota": "400%", "RuntimeMaxSec": "1800", "TimeoutStopSec": "20",
    }
    environment = {
        "PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LC_ALL": "C.UTF-8",
        "LD_LIBRARY_PATH": ":".join(map(str, library_paths)),
        "OPENSSL_CONF": "/dev/null",
        "OPENSSL_MODULES": str(ROOT / "supplemental/lib"),
        "OPENSSL_ENGINES": str(ROOT / "supplemental/lib"),
        "AWS_EC2_METADATA_DISABLED": "true", "OTEL_SDK_DISABLED": "true",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    }
    kit = Path(__file__).resolve().parent
    return [
        "systemd-run", f"--unit={UNIT}", "--wait", "--pipe", "--service-type=exec",
        *[f"--property={key}={value}" for key, value in properties.items()],
        "/usr/bin/env", "-i", *[f"{key}={value}" for key, value in environment.items()],
        str(ROOT / "python/bin/python3.12"), "-I", str(kit / "runner.py"),
        "--architecture", architecture,
    ]


def bounded_results(output, target):
    files = sorted(path for path in output.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("Output contains a symbolic link")
    if len(files) > 1000 or sum(path.stat().st_size for path in files) > MAX_OUTPUT_BYTES:
        raise ValueError("Output exceeds native-check transfer allowance")
    with tarfile.open(target, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(output)), recursive=False)
    if target.stat().st_size > MAX_OUTPUT_BYTES:
        raise ValueError("Compressed output exceeds transfer allowance")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--architecture", choices=ARCHES, required=True)
    parser.add_argument("--catalog-sha256", required=True)
    args = parser.parse_args()
    import boto3  # Host preparation dependency, not used by the isolated service.
    client = boto3.client("s3", region_name="us-east-1")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    receipt = {"started_at": datetime.now(timezone.utc).isoformat(), "architecture": args.architecture,
               "qualification_passed": False, "preparation_passed": False}
    started = time.monotonic()
    try:
        search = prepare(client, args)
        account = pwd.getpwnam("organism")
        os.chown(OUTPUT, account.pw_uid, account.pw_gid)
        OUTPUT.chmod(0o700)
        receipt["preparation_passed"] = True
        with (OUTPUT / "service.log").open("w") as log:
            result = subprocess.run(
                service_command(args.architecture, search), stdout=log, stderr=subprocess.STDOUT,
                timeout=1900, check=False,
            )
        receipt["service_return_code"] = result.returncode
    except Exception as exc:
        receipt["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        # A timeout in the observer does not establish the service has stopped.
        stop = subprocess.run(["systemctl", "stop", UNIT], capture_output=True, text=True, timeout=40)
        state = subprocess.run(
            ["systemctl", "show", UNIT, "--property=ActiveState,SubState,ExecMainStatus,MemoryPeak,MemoryMax"],
            capture_output=True, text=True, timeout=20,
        )
        receipt["service_state_after_stop"] = state.stdout.strip()
        receipt["stop_return_code"] = stop.returncode
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(OUTPUT / "host-receipt.json", receipt)
        # Always attempt the small failure receipt first. Full results can fail
        # independently without disguising a worker as still running.
        prefix = f"{args.prefix}/outputs/{args.architecture}"
        client.put_object(Bucket=args.bucket, Key=f"{prefix}/host-receipt.json",
                          Body=(OUTPUT / "host-receipt.json").read_bytes())
        target = Path("/var/lib/organism-native/results.tar.gz")
        bounded_results(OUTPUT, target)
        client.upload_file(str(target), args.bucket, f"{prefix}/results.tar.gz")
        client.put_object(
            Bucket=args.bucket, Key=f"{prefix}/results-identity.json",
            Body=json.dumps({"sha256": digest(target), "bytes": target.stat().st_size}).encode(),
        )
        print(json.dumps({"output_prefix": prefix, "receipt": receipt}), flush=True)
    return 0 if receipt.get("service_return_code") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
