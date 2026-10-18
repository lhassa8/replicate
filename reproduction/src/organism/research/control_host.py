"""Owner-only download, bounded control execution, public evidence upload."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import traceback

import boto3


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--assets-sha256", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    root = Path("/opt/replicate-paper")
    root.mkdir(mode=0o700, exist_ok=False)
    root.chmod(0o700)
    assets_dir, inputs, owner, output = (root / x for x in ("assets", "inputs", "owner", "results"))
    for path in (assets_dir, inputs, owner):
        path.mkdir(mode=0o700)
    client = boto3.client("s3", region_name=args.region)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "failure": None,
              "deadline": args.deadline, "inputs_verified": False, "worker_returncode": None}
    process = None
    try:
        assets_path = assets_dir / "assets.json"
        client.download_file(args.bucket, args.prefix + "/inputs/assets.json", str(assets_path))
        if sha(assets_path) != args.assets_sha256:
            raise ValueError("Frozen assets manifest differs")
        assets = json.loads(assets_path.read_text())
        for relative, expected in assets["files"].items():
            path = assets_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(args.bucket, args.prefix + "/inputs/" + relative, str(path))
            if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
                raise ValueError("Downloaded input identity differs: " + relative)
        for arm, specification in assets["arms"].items():
            capsule = inputs / "capsules" / arm
            capsule.mkdir(parents=True)
            manifest_path = assets_dir / specification["manifest_asset"]
            manifest = json.loads(manifest_path.read_text())
            shutil.copyfile(manifest_path, capsule / ".manifest.json")
            for relative in manifest["body"]["files"]:
                source = (assets_dir / specification["model_asset"] if relative == manifest["body"]["model"]["path"]
                          else assets_dir / assets["shared"][relative]["asset"])
                target = capsule / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        for name in ("release-trust.json", "smoke-cases.json"):
            shutil.copyfile(assets_dir / name, inputs / name)
        with tarfile.open(assets_dir / "owner-source.tar.gz") as tar:
            tar.extractall(owner, filter="data")
        for role in ("python", "dependencies"):
            source = assets_dir / "shared/linux-x86_64" / (role + ".tar.gz")
            destination = root / role
            destination.mkdir()
            with tarfile.open(source) as tar:
                tar.extractall(destination, filter="data")
        # The carried Python archive has its own top-level python directory.
        python = root / "python/python/bin/python3.12"
        dependencies = root / "dependencies/site-packages"
        libraries = [root / "python/python/lib", root / "dependencies/supplemental/lib"]
        environment = {
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/nonexistent", "LC_ALL": "C.UTF-8",
            "PYTHONPATH": str(owner) + ":" + str(dependencies),
            "LD_LIBRARY_PATH": ":".join(map(str, libraries)),
            "OTEL_SDK_DISABLED": "true", "AWS_EC2_METADATA_DISABLED": "true",
        }
        # The application users must traverse only the owner's public trial root,
        # never the seed input store or private download directory above.
        Path("/var/lib/replicate-paper").mkdir(mode=0o755, exist_ok=True)
        report["inputs_verified"] = True
        report["input_manifest_sha256"] = args.assets_sha256
        report["host"] = {"uname": os.uname()._asdict() if hasattr(os.uname(), "_asdict") else list(os.uname()),
                          "meminfo": Path("/proc/meminfo").read_text(),
                          "cpuinfo": Path("/proc/cpuinfo").read_text()}
        remaining = datetime.fromisoformat(args.deadline).timestamp() - time.time() - 600
        if remaining < 1800:
            raise RuntimeError("Too little of the fixed lease remains")
        with (root / "execution.log").open("wb") as log:
            process = subprocess.Popen([str(python), "-m", "organism.research.control_worker",
                                        "--inputs", str(inputs), "--output", str(output)],
                                       env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                if datetime.fromisoformat(args.deadline).timestamp() - time.time() < 600:
                    process.terminate()
                    raise TimeoutError("Fixed lease cleanup margin reached")
                progress = {"at": datetime.now(timezone.utc).isoformat(), "worker_pid": process.pid,
                            "summary": json.loads((output / "summary.json").read_text())
                            if (output / "summary.json").exists() else [],
                            "tail": (root / "execution.log").read_text(errors="replace")[-12000:]}
                client.put_object(Bucket=args.bucket, Key=args.prefix + "/outputs/x86_64/progress.json",
                                  Body=json.dumps(progress).encode(), ContentType="application/json")
                time.sleep(10)
            report["worker_returncode"] = process.returncode
            if process.returncode:
                raise RuntimeError("Control worker returned " + str(process.returncode))
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        subprocess.run(["systemctl", "stop", "replicate-paper-*.service"], check=False)
        output.mkdir(exist_ok=True)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        (output / "host-report.json").write_text(json.dumps(report, indent=2) + "\n")
        if (root / "execution.log").exists():
            shutil.copyfile(root / "execution.log", output / "execution.log")
        archive = root / "results.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(output, arcname="results", recursive=True)
        if archive.stat().st_size > 256 * 1024**2:
            raise RuntimeError("Public result archive exceeds the upload allowance")
        receipt = {"bytes": archive.stat().st_size, "sha256": sha(archive),
                   "failure": report["failure"], "completed_at": report["completed_at"]}
        client.upload_file(str(archive), args.bucket, args.prefix + "/outputs/x86_64/results.tar.gz")
        client.put_object(Bucket=args.bucket, Key=args.prefix + "/outputs/x86_64/receipt.json",
                          Body=json.dumps(receipt).encode(), ContentType="application/json")
        subprocess.run(["shutdown", "-h", "now"], check=False)


if __name__ == "__main__":
    main()
