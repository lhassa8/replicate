"""Prepare a bounded three-device request; admission and AWS calls stay external."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import stat
import tempfile

from ..training_inputs import require
from .build_lineage import checked
from .common import check_file
from .first_lineage_profile import CASES_SHA256
from .infrastructure import validate_launch
from .lineage_enrollment import MAX_BUNDLE, extract_bundle, verify
from .lineage_infrastructure import LABELS, worker_template
from .profile_operations import REGIONS


def checked_enrollment(directory, catalog):
    """Bind actual private bundles to the public enrollment without returning keys."""
    directory = Path(directory).absolute()
    workspace = Path(__file__).resolve().parents[2]
    require(not directory.resolve().is_relative_to(workspace),
            "Private enrollment must stay outside the repository")
    root = directory.lstat()
    require(stat.S_ISDIR(root.st_mode) and not root.st_mode & 0o077,
            "Enrollment directory must be private")
    receipt_path = directory / "receipt.json"
    entry = receipt_path.lstat()
    require(stat.S_ISREG(entry.st_mode) and entry.st_nlink == 1
            and not entry.st_mode & 0o077 and entry.st_size <= 65536,
            "Invalid enrollment receipt")
    receipt = json.loads(receipt_path.read_bytes())
    require(receipt["schema"] == "organism-lineage-owner-enrollment-v1"
            and receipt["release_id"] == catalog["release_id"]
            and receipt["execution_slots"] == list(LABELS)
            and set(receipt["bundles"]) == set(LABELS)
            and receipt["bootstrap_has_execution_slot"] is False
            and receipt["owner_private_key_serialized"] is False
            and not receipt["receiver_started"] and not receipt["descendant_started"],
            "Enrollment changes its release or execution scope")
    inputs = {}
    for slot in LABELS:
        bundle = receipt["bundles"][slot]
        path = directory / (slot + ".tar.gz")
        entry = path.lstat()
        require(bundle["path"] == str(path) and stat.S_ISREG(entry.st_mode)
                and entry.st_nlink == 1 and not entry.st_mode & 0o077
                and 0 < entry.st_size <= MAX_BUNDLE, "Invalid private bundle path or permissions")
        check_file(path, bundle)
        with tempfile.TemporaryDirectory(prefix="lineage-launch-check-") as temporary:
            extracted = Path(temporary) / slot
            extract_bundle(path, extracted)
            node, policy, book = verify(extracted, slot, catalog)
            require(node["device_ids"] == receipt["device_ids"]
                    and policy.policy_id == receipt["policy_id"]
                    and book.book_id == receipt["book_id"]
                    and node["owner_public"] == receipt["owner_public"]
                    and receipt["owner_transport_principal"] == node["device_ids"]["bootstrap"],
                    "Enrollment receipt differs from the verified bundled node")
        inputs["inputs/nodes/" + slot + ".tar.gz"] = {
            key: bundle[key] for key in ("bytes", "sha256")}
    return inputs


def launch_packet(directory, single_kit, enrollment_directory, *, bucket, prefix,
                  region, zone, image, account_id, deadline, approval_reference, now=None,
                  seed_expiry_at=None):
    require(region in REGIONS and re.fullmatch(re.escape(region) + "[a-z]", zone),
            "Lineage requires an explicitly selected U.S. commercial zone")
    require(re.fullmatch(r"[0-9]{12}", account_id)
            and re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket)
            and re.fullmatch(r"ami-[0-9a-f]{17}", image), "Invalid lineage account, bucket or image")
    require(isinstance(approval_reference, str) and approval_reference.strip(),
            "Lineage spending authority is required")
    require(validate_launch(prefix, deadline, now=now) >= 11400,
            "Allow preparation plus the original three-hour minimum for lineage admission")
    build, catalog, inventory = checked(directory, single_kit)
    inputs = {name: {key: entry[key] for key in ("bytes", "sha256")}
              for name, entry in inventory.items()}
    inputs.update(checked_enrollment(enrollment_directory, catalog))
    parameters = {
        "ArtifactBucket": bucket, "RunPrefix": prefix, "AvailabilityZone": zone,
        "KitSha256": build["kit_sha256"], "CatalogSha256": build["catalog_sha256"],
        "CasesSha256": CASES_SHA256, "DeadlineDate": deadline.split("T")[0],
        "DeadlineTime": deadline.split("T")[1], "ImageId": image,
        **{label + "BundleSha256": inputs["inputs/nodes/" + slot + ".tar.gz"]["sha256"]
           for slot, label in LABELS.items()},
    }
    if seed_expiry_at is not None:
        current = now or datetime.now(timezone.utc)
        end = datetime.strptime(deadline, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        stop = datetime.strptime(seed_expiry_at, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        require((end - stop).total_seconds() == 9000
                and (stop - current).total_seconds() >= 4800,
                "Earlier seed expiry must preserve its fixed timing and preparation margin")
    template = worker_template(x86_instance_type=build.get("x86_instance_type"),
                               seed_expiry_at=seed_expiry_at)
    require(set(parameters) == set(template["Parameters"]), "Lineage template parameters changed")
    return {
        "schema": "organism-native-lineage-launch-v1",
        "region": region, "account_id": account_id, "deadline": deadline,
        "approval_reference": approval_reference, "parameters": parameters,
        **({"seed_expiry_at": seed_expiry_at} if seed_expiry_at is not None else {}),
        "input_identities": inputs, "release_id": build["release_id"],
        **({"x86_instance_type": build["x86_instance_type"]}
           if "x86_instance_type" in build else {}),
        "prelaunch_requirements": [
            "Independently reconstruct this release's successful single-device native trial and its complete cloud cleanup.",
            "Verify fresh capacity, spending and aggregate CPU-hour bounds before allocating three devices.",
            "Verify the dedicated private bucket and every exact input, including these fresh private enrollment bundles.",
            "Keep the original fixed deadline across retries and verify all deployed controls before publishing permits.",
            "Only the seed receives the full package from owner storage; descendants obtain it through their peers.",
        ],
        "request": {
            "service_name": "cloudformation", "operation_name": "CreateStack", "region_name": region,
            "params": {
                "StackName": prefix, "TemplateBody": json.dumps(template),
                "Parameters": [{"ParameterKey": name, "ParameterValue": value}
                               for name, value in parameters.items()],
                "Capabilities": ["CAPABILITY_IAM"], "OnFailure": "ROLLBACK", "TimeoutInMinutes": 15,
                "Tags": [{"Key": "Purpose", "Value": "organism-native-preflight"},
                         {"Key": "Experiment", "Value": prefix}],
            },
        },
    }
