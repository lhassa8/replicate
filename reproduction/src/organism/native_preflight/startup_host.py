"""Disposable native host preparation and independent seed-service cleanup."""

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

from .common import ARCHES, digest, extract, write_json
from .host import bounded_results, download
from .startup_common import BASE, BOOT, MAX_CATALOG, NAMESPACE, NS_PATH, OUTPUT, ROOT, UNIT, validate_catalog
from .verify_libraries import verify_carried


def command(arguments, *, timeout=30):
    return subprocess.run(arguments, capture_output=True, text=True, timeout=timeout, check=False,
                          env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C",
                               "SYSTEMD_PAGER": "", "SYSTEMD_COLORS": "0"})


def prepare(client, args):
    ROOT.mkdir()
    response = client.get_object(Bucket=args.bucket, Key=f"{args.prefix}/inputs/catalog.json")
    try:
        if not 0 < response["ContentLength"] <= MAX_CATALOG:
            raise ValueError("Startup catalog exceeds its bound")
        raw = response["Body"].read(MAX_CATALOG + 1)
    finally:
        response["Body"].close()
    if hashlib.sha256(raw).hexdigest() != args.catalog_sha256:
        raise ValueError("Startup catalog differs from the approved launch")
    catalog = validate_catalog(json.loads(raw))
    (ROOT / "catalog.json").write_bytes(raw)
    (OUTPUT / "input-catalog.json").write_bytes(raw)
    capsule = ROOT / "capsule"
    capsule.mkdir()
    for name, specification in catalog["files"].items():
        target = capsule / name
        target.parent.mkdir(parents=True, exist_ok=True)
        download(client, args.bucket, f"{args.prefix}/inputs/capsule/{name}", target, specification)
    (capsule / ".manifest.json").write_bytes(base64.b64decode(catalog["manifest_base64"], validate=True))
    BOOT.mkdir()
    native = catalog["platforms"]["linux-" + args.architecture]
    extract(capsule / native["python"], BOOT)
    extract(capsule / native["dependencies"], BOOT)
    extract(capsule / native["inference"], BOOT / "tools/llama.cpp/b9850")
    search = verify_carried(BOOT, OUTPUT, args.architecture)
    return catalog, search


def runner_command(architecture, libraries, *, operation="run"):
    environment = {
        "PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LC_ALL": "C.UTF-8",
        "LD_LIBRARY_PATH": ":".join(map(str, libraries)),
        "OPENSSL_CONF": "/dev/null", "OPENSSL_MODULES": str(BOOT / "supplemental/lib"),
        "OPENSSL_ENGINES": str(BOOT / "supplemental/lib"),
        "AWS_EC2_METADATA_DISABLED": "true", "OTEL_SDK_DISABLED": "true",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    }
    return ["/usr/bin/env", "-i", *[key + "=" + value for key, value in environment.items()],
            str(BOOT / "python/bin/python3.12"), "-I", str(Path(__file__).with_name("startup_runner.py")),
            "--architecture", architecture, "--operation", operation]


def service_command(architecture, libraries):
    properties = {
        "User": "root", "Group": "root", "NetworkNamespacePath": NS_PATH,
        "PrivateTmp": "yes", "UMask": "0077", "KillMode": "control-group",
        "MemoryAccounting": "yes", "MemoryMax": "768M", "MemorySwapMax": "0",
        "CPUQuota": "100%", "TasksMax": "64", "RuntimeMaxSec": "1500", "TimeoutStopSec": "20",
    }
    return ["systemd-run", f"--unit={UNIT}", "--wait", "--pipe", "--service-type=exec",
            *["--property=" + key + "=" + value for key, value in properties.items()],
            *runner_command(architecture, libraries)]


def create_namespace(on_created=lambda: None):
    if os.path.lexists(NS_PATH):
        raise ValueError("Do not reuse an existing native-test network namespace")
    created = command(["ip", "netns", "add", NAMESPACE])
    if created.returncode:
        raise RuntimeError("Could not create the native-test network namespace")
    on_created()
    up = command(["ip", "-n", NAMESPACE, "link", "set", "lo", "up"])
    links = command(["ip", "-j", "-n", NAMESPACE, "link", "show"])
    routes = command(["ip", "-j", "-n", NAMESPACE, "route", "show", "table", "main"])
    if up.returncode or links.returncode or routes.returncode:
        raise RuntimeError("Native-test network namespace could not be verified")
    values = json.loads(links.stdout)
    if len(values) != 1 or values[0]["ifname"] != "lo" or "UP" not in values[0]["flags"] or json.loads(routes.stdout):
        raise RuntimeError("Native-test namespace has external interfaces or routes")
    return {"interfaces": values, "main_ipv4_routes": [], "namespace_path": NS_PATH}


def group_empty(name):
    if not Path("/sys/fs/cgroup/cgroup.controllers").is_file():
        raise ValueError("Native cgroup v2 observation is unavailable")
    group = Path("/sys/fs/cgroup/system.slice") / name
    if not group.exists():
        return True
    if group.is_symlink() or not group.is_dir():
        return False
    fields = dict(line.split() for line in (group / "cgroup.events").read_text().splitlines())
    return fields.get("populated") == "0"


def stop_and_observe(name):
    stopped = command(["systemctl", "stop", name], timeout=40)
    shown = command(["systemctl", "show", name, "--property=Id,LoadState,ActiveState,SubState,MainPID,Job"])
    values = dict(line.split("=", 1) for line in shown.stdout.splitlines() if "=" in line)
    valid_response = shown.returncode == 0 or (shown.returncode == 4 and values.get("LoadState") == "not-found")
    absent = (valid_response and values.get("Id") == name and values.get("ActiveState") in ("inactive", "failed")
              and values.get("MainPID") == "0" and values.get("Job") in ("", "0")
              and group_empty(name))
    return {"unit": name, "stop_return_code": stopped.returncode, "show_return_code": shown.returncode, "properties": values,
            "entire_cgroup_empty": group_empty(name), "absence_verified": absent}


def owned_units(config, unit_root=Path("/run/systemd/system"), *, maximum_starts=1):
    if type(maximum_starts) is not int or maximum_starts not in (1, 3):
        raise ValueError("Cleanup requires a declared single- or three-start allowance")
    identity = config["device_id"]
    if not re.fullmatch("[0-9a-f]{64}", identity):
        raise ValueError("Cleanup device identity is invalid")
    prefix = "organism-" + identity[:16] + "-"
    found, starts, individuals = [], set(), set()
    for path in unit_root.glob(prefix + "*.service"):
        entry = path.lstat()
        if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.geteuid()
                or entry.st_nlink != 1 or entry.st_mode & 0o022 or entry.st_size > 65536):
            raise ValueError("Cleanup unit is not controlled by this owner")
        matched = re.fullmatch(
            re.escape(prefix) + "([0-9a-f]{32})-([0-9a-f]{32})(-check)?\\.service", path.name)
        if matched is None:
            raise ValueError("Unexpected service under this seed identity")
        text = path.read_text()
        if not any(line.startswith("Description=organism-device_id=" + identity + " ") for line in text.splitlines()):
            raise ValueError("Cleanup unit description belongs to another device")
        found.append(path.name)
        individuals.add(matched[1])
        starts.add((matched[1], matched[2]))
    if len(found) > 2 * maximum_starts or len(starts) > maximum_starts or len(individuals) > 1:
        raise ValueError("Native seed exceeded its declared application/checker allowance")
    return sorted(found, key=lambda name: (not name.endswith("-check.service"), name))


def read_owner_config():
    fd = os.open(BASE / "receiver.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        entry = os.fstat(source.fileno())
        if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != 0 or entry.st_nlink != 1
                or entry.st_size > 16384 or entry.st_mode & 0o077):
            raise ValueError("Independent cleanup configuration is not owner-private")
        raw = source.read(16385)
    if len(raw) > 16384:
        raise ValueError("Independent cleanup configuration exceeds its bound")
    return json.loads(raw)


def cleanup(architecture, libraries, *, namespace_created):
    result = {"passed": False}
    result["driver"] = stop_and_observe(UNIT + ".service")
    config_path = BASE / "receiver.json"
    units = []
    if config_path.exists():
        config = read_owner_config()
        # This is a separate process after the driver was asked to stop. The
        # fallback below still stops the exact owned units if the journal or
        # portable interpreter cannot be read.
        try:
            retirement = command(runner_command(architecture, libraries, operation="retire"), timeout=90)
            result["owner_retirement"] = {"return_code": retirement.returncode,
                                          "stdout": retirement.stdout[-65536:], "stderr": retirement.stderr[-4000:]}
        except Exception as exc:
            result["retirement_failure"] = type(exc).__name__ + ": " + str(exc)
        units = owned_units(config)
    result["application_and_checker"] = []
    for name in units:
        try:
            result["application_and_checker"].append(stop_and_observe(name))
        except Exception as exc:
            result["application_and_checker"].append({"unit": name, "absence_verified": False,
                                                       "failure": type(exc).__name__ + ": " + str(exc)})
    empty = result["driver"]["absence_verified"] and all(v["absence_verified"] for v in result["application_and_checker"])
    result["namespace_removed"] = False
    if namespace_created and empty:
        pids = command(["ip", "netns", "pids", NAMESPACE])
        result["remaining_namespace_pids"] = pids.stdout.strip().splitlines()
        if pids.returncode == 0 and not pids.stdout.strip():
            removed = command(["ip", "netns", "delete", NAMESPACE])
            result["namespace_removed"] = removed.returncode == 0 and not os.path.lexists(NS_PATH)
    result["passed"] = empty and (not namespace_created or result["namespace_removed"])
    return result


def copy_evidence():
    """Copy only designated evidence trees; device keys and admission secrets stay off output."""
    target = OUTPUT / "seed-evidence"
    target.mkdir()
    roots = [BASE / "work"]
    if (BASE / "state").exists():
        roots += [path for path in (BASE / "state").iterdir()
                  if re.fullmatch("[0-9a-f]{32}-check", path.name)]
    files = [BASE / name for name in ("receiver.json", "enrollment.json", "discovery.json",
                                      "state/receiver-status.json", "state/supervision.sqlite3", "state/supervision.enrolled",
                                      "state/peer-audit.sqlite3", "state/peer-audit.enrolled")]
    for root in roots:
        if root.exists():
            if root.is_symlink():
                raise ValueError("Seed evidence directory is a symlink")
            for path in root.rglob("*"):
                if path.is_symlink():
                    raise ValueError("Seed evidence contains a symlink")
                if path.is_file():
                    files.append(path)
    files = [path for path in files if path.exists()]
    if len(files) > 900 or sum(path.stat().st_size for path in files) > 384 * 1024**2:
        raise ValueError("Seed evidence exceeds its output allowance")
    for path in files:
        entry = path.lstat()
        if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
            raise ValueError("Seed evidence is not a regular private file")
        destination = target / path.relative_to(BASE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return {"files": len(files), "device_private_keys_copied": False, "admission_state_database_copied": False}


def capture_journals():
    config = BASE / "receiver.json"
    units = [UNIT + ".service"]
    if config.exists():
        units += owned_units(read_owner_config())
    target = OUTPUT / "journals"
    target.mkdir()
    result = []
    for name in units:
        log = command(["journalctl", "-u", name, "--no-pager", "-n", "1000", "--output=short-monotonic"])
        raw = log.stdout.encode()
        if len(raw) > 4 * 1024**2:
            raise ValueError("Native service journal exceeds its capture allowance")
        (target / (name + ".log")).write_bytes(raw)
        result.append({"unit": name, "return_code": log.returncode, "bytes": len(raw), "stderr": log.stderr[-4000:]})
    return result


def memory_sample():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, text = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            values[key + "_kib"] = int(text.split()[0])
    units = [UNIT + ".service"]
    config = BASE / "receiver.json"
    if config.exists():
        units += owned_units(read_owner_config())
    groups = {}
    for name in units:
        group = Path("/sys/fs/cgroup/system.slice") / name
        if not group.exists():
            continue
        try:
            groups[name] = {field: (group / field).read_text().strip()
                            for field in ("memory.current", "memory.peak", "memory.events", "memory.swap.current")}
        except FileNotFoundError:
            groups[name] = {"observation": "one or more cgroup counters unavailable during sample"}
    return {"monotonic_seconds": time.monotonic(), **values, "cgroups": groups}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--architecture", choices=ARCHES, required=True)
    parser.add_argument("--catalog-sha256", required=True)
    args = parser.parse_args()
    import boto3
    client = boto3.client("s3", region_name="us-east-1")
    OUTPUT.mkdir(parents=True)
    receipt = {"schema": "organism-native-startup-host-v1", "started_at": datetime.now(timezone.utc).isoformat(),
               "architecture": args.architecture, "passed": False, "qualification_passed": False,
               "trained_for_organism": False, "autonomous_reproduction_verified": False}
    search, owned_namespace = [], False
    process = None
    samples = []
    try:
        catalog, search = prepare(client, args)
        receipt["release_id"] = catalog["release_id"]
        sample = memory_sample()
        if not 7 * 1024**2 <= sample["MemTotal_kib"] <= 8 * 1024**2 or sample["SwapTotal_kib"] != 0:
            raise ValueError("Native host does not match the declared 8-GiB, no-swap profile")
        def namespace_owned():
            nonlocal owned_namespace
            owned_namespace = True
        receipt["namespace"] = create_namespace(namespace_owned)
        with (OUTPUT / "startup-service.log").open("x") as log:
            process = subprocess.Popen(service_command(args.architecture, search), stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 1560
            while process.poll() is None:
                samples.append(memory_sample())
                if time.monotonic() >= deadline:
                    raise TimeoutError("Native startup observer deadline exceeded")
                time.sleep(1)
            receipt["service_return_code"] = process.returncode
        seed = json.loads((OUTPUT / "seed-receipt.json").read_bytes())
        receipt["seed_passed"] = seed["passed"]
    except Exception as exc:
        receipt["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        try:
            receipt["cleanup"] = cleanup(args.architecture, search, namespace_created=owned_namespace)
        except Exception as exc:
            receipt["cleanup"] = {"passed": False, "failure": type(exc).__name__ + ": " + str(exc)}
        if process is not None and process.poll() is None:
            # The client process is not the service. Its exit never substitutes
            # for the independent unit/cgroup observations above.
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        try:
            receipt["evidence_copy"] = copy_evidence()
            receipt["journals"] = capture_journals()
        except Exception as exc:
            receipt["evidence_copy_failure"] = type(exc).__name__ + ": " + str(exc)
        write_json(OUTPUT / "memory-samples.json", samples)
        receipt["passed"] = bool(receipt.get("service_return_code") == 0 and receipt.get("seed_passed")
                                  and receipt["cleanup"]["passed"] and not receipt.get("failure")
                                  and not receipt.get("evidence_copy_failure"))
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(OUTPUT / "host-receipt.json", receipt)
        prefix = f"{args.prefix}/outputs/{args.architecture}"
        client.put_object(Bucket=args.bucket, Key=prefix + "/host-receipt.json",
                          Body=(OUTPUT / "host-receipt.json").read_bytes())
        target = OUTPUT.parent / "results.tar.gz"
        bounded_results(OUTPUT, target)
        client.upload_file(str(target), args.bucket, prefix + "/results.tar.gz")
        client.put_object(Bucket=args.bucket, Key=prefix + "/results-identity.json",
                          Body=json.dumps({"sha256": digest(target), "bytes": target.stat().st_size}).encode())
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
