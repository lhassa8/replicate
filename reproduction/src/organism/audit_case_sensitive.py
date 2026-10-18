"""Run archive or capsule checks on one bounded case-sensitive macOS disk image.

Creates only a new image under /private/tmp, mounts that image at its own new
mount directory, and detaches the device returned by that attach operation.
No existing volume is reformatted or modified. Failed detach preserves the
image and its directory for owner inspection instead of deleting mounted data.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import shutil
import subprocess
import tempfile

from .audit_preparation_archives import run
from .capsule import PLATFORMS
from .safe_archive import case_sensitive_directory


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--capsule", type=Path, help="Audit this complete development artifact on a 5-GiB image")
    parser.add_argument("--platform", choices=sorted(PLATFORMS),
                        help="With --capsule, prepare one platform on this fresh image; verify all capsule payloads")
    args = parser.parse_args(argv)
    if args.platform is not None and args.capsule is None:
        parser.error("--platform requires --capsule")
    return args


def main():
    args = arguments()
    if platform.system() != "Darwin":
        raise RuntimeError("This bounded filesystem audit is only for the development Mac")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Audit output already exists")
    size_gib = 5 if args.capsule is not None else 1
    if shutil.disk_usage("/private/tmp").free < (4 if args.capsule else 1) * 1024**3:
        raise RuntimeError("Insufficient host free space for the temporary filesystem audit")
    workspace = Path(tempfile.mkdtemp(prefix="organism-case-audit-", dir="/private/tmp"))
    image, mount = workspace / "archive-audit.sparseimage", workspace / "mount"
    mount.mkdir()
    receipt = {"started_at": datetime.now(timezone.utc).isoformat(), "temporary_directory": str(workspace),
               "image_capacity_bytes": size_gib * 1024**3, "existing_volumes_modified": False,
               "native_linux_execution": False, "aws_resources_created": False, "passed": False,
               "selected_platform": args.platform}
    device = None
    detached = False
    try:
        created = subprocess.run(
            ["hdiutil", "create", "-size", str(size_gib) + "g", "-type", "SPARSE", "-fs", "Case-sensitive APFS",
             "-volname", "OrganismArchiveAudit", "-nospotlight", str(image)],
            text=True, capture_output=True, timeout=120,
        )
        if created.returncode:
            raise RuntimeError("Temporary image creation failed: " + created.stderr[-2000:])
        print(f"Created the dedicated {size_gib}-GiB audit image.", flush=True)
        attached = subprocess.run(
            ["hdiutil", "attach", "-nobrowse", "-noautoopen", "-owners", "on",
             "-mountpoint", str(mount), "-plist", str(image)],
            capture_output=True, timeout=60,
        )
        if attached.returncode:
            raise RuntimeError("Temporary image attach failed: " + attached.stderr.decode()[-2000:])
        entities = plistlib.loads(attached.stdout)["system-entities"]
        devices = [item["dev-entry"] for item in entities if re.fullmatch(r"/dev/disk[0-9]+", item.get("dev-entry", ""))]
        if not devices:
            raise RuntimeError("Attach did not return an identifiable owned image device")
        device = devices[0]
        receipt["attached_device"] = device
        if not any(item.get("mount-point") == str(mount) for item in entities):
            raise RuntimeError("Image was not attached at the requested private mount point")
        receipt["case_sensitive_verified"] = case_sensitive_directory(mount)
        if not receipt["case_sensitive_verified"]:
            raise RuntimeError("Temporary filesystem does not preserve letter case")
        print("Verified the temporary case-sensitive filesystem.", flush=True)
        if args.capsule is not None:
            from .audit_complete_capsule import run as run_complete
            report = run_complete(Path(__file__).resolve().parent, args.capsule.resolve(), output,
                                  temporary_root=mount, selected_platform=args.platform)
            receipt["both_platforms_prepared"] = report["both_platforms_prepared"]
        else:
            report = run(Path(__file__).resolve().parent, output, temporary_root=mount)
        receipt["passed"] = report["passed"]
    except Exception as exc:
        receipt["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        if device:
            try:
                result = subprocess.run(["hdiutil", "detach", device], capture_output=True, text=True, timeout=60)
                detached = result.returncode == 0
                receipt["detach_return_code"] = result.returncode
                if not detached:
                    receipt["detach_error"] = result.stderr[-2000:]
            except Exception as exc:
                receipt["detach_error"] = type(exc).__name__ + ": " + str(exc)
        # If attach failed without returning a device, preserve all files:
        # an observation failure is not proof that an attach did not occur.
        if detached:
            shutil.rmtree(workspace)
            receipt["temporary_image_removed"] = True
        else:
            receipt["temporary_image_removed"] = False
        receipt["image_detached"] = detached
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        output.mkdir(parents=True, exist_ok=True)
        (output / "mac-filesystem-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": receipt["passed"], "image_detached": detached,
                      "temporary_image_removed": receipt["temporary_image_removed"], "failure": receipt.get("failure")}))
    return 0 if receipt["passed"] and detached else 1


if __name__ == "__main__":
    raise SystemExit(main())
