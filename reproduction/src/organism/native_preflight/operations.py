"""Generate bounded AWS MCP request packets after explicit spending approval.

No API calls are made here. Live resource IDs come from recorded AWS responses,
never from name searches across unrelated projects.
"""

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import re

from .common import check_file, write_json
from .infrastructure import validate_launch


def launch_packet(kit, *, bucket, prefix, availability_zone, approval_reference, now=None):
    """Return exact CreateStack parameters; the operator rechecks time at use."""
    if not approval_reference.strip():
        raise ValueError("A direct user spending-approval reference is required")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
        raise ValueError("Malformed dedicated artifact bucket")
    if availability_zone not in ("us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d", "us-east-1f"):
        raise ValueError("Availability zone was not verified for both instance types")
    kit = Path(kit)
    inventory = json.loads((kit / "upload-inventory.json").read_text())
    for entry in inventory["files"]:
        check_file(entry["path"], entry)
    hashes = {Path(entry["path"]).name: entry["sha256"] for entry in inventory["files"]}
    now = now or datetime.now(timezone.utc)
    deadline = (now + timedelta(hours=4)).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")
    validate_launch(prefix, deadline, now=now)
    day, clock = deadline.split("T")
    parameters = {
        "ArtifactBucket": bucket, "RunPrefix": prefix, "AvailabilityZone": availability_zone,
        "CatalogSha256": hashes["catalog.json"], "KitSha256": hashes["kit.tar.gz"],
        "DeadlineDate": day, "DeadlineTime": clock,
    }
    # The reviewed template must still match its build receipt.
    build_receipt = json.loads((kit / "build-receipt.json").read_text())
    template = kit / "worker-template.json"
    check_file(template, {"bytes": template.stat().st_size, "sha256": build_receipt["worker_template_sha256"]})
    return {
        "region": "us-east-1", "approval_reference": approval_reference,
        "generated_at": now.isoformat(), "expires_utc": deadline,
        "prelaunch_requirements": [
            "Record direct user approval and its native-check spending ceiling in the execution ledger.",
            "No other pair from this experiment is pending, running, stopping, or shutting down.",
            "This attempt is within the recorded user authorization, including any explicit amendment; aggregate CPU hours remain within 16.",
            "Verify uploaded S3 objects against upload inventory; no mutation after worker launch.",
            "Revalidate expiry immediately before CreateStack; preserve every returned resource identifier.",
        ],
        "request": {"service_name": "cloudformation", "operation_name": "CreateStack", "region_name": "us-east-1",
                    "params": {"StackName": prefix, "TemplateBody": template.read_text(),
                               "Parameters": [{"ParameterKey": key, "ParameterValue": value} for key, value in parameters.items()],
                               "Capabilities": ["CAPABILITY_IAM"], "OnFailure": "ROLLBACK",
                               "TimeoutInMinutes": 15,
                               "Tags": [{"Key": "Purpose", "Value": "organism-native-preflight"},
                                        {"Key": "Experiment", "Value": prefix}]}},
    }


def cleanup_packet(ledger, *, preserved_receipt):
    if not preserved_receipt.strip():
        raise ValueError("Preserve failure/success evidence before deleting it")
    prefix = ledger["prefix"]
    if not re.fullmatch(r"native-[0-9]{8}-[a-z0-9]{4,12}", prefix):
        raise ValueError("Not a native experiment ledger")
    instances = ledger["instance_ids"]
    if not 1 <= len(instances) <= 2 or len(set(instances)) != len(instances):
        raise ValueError("Cleanup must identify exactly the recorded worker pair")
    if any(not re.fullmatch(r"i-[0-9a-f]{8,17}", value) for value in instances):
        raise ValueError("Malformed recorded instance ID")
    keys = ledger["object_keys"]
    if len(keys) > 1000 or any(not key.startswith(prefix + "/") for key in keys):
        raise ValueError("Cleanup cannot include other experiment objects")
    stack = ledger["worker_stack_id"]
    if ":stack/" + prefix + "/" not in stack or not stack.startswith("arn:aws:cloudformation:us-east-1:"):
        raise ValueError("Worker stack does not match the ledger")
    return {
        "scope": "Exact recorded resources only; execute phases sequentially with state verification",
        "preserved_receipt": preserved_receipt,
        "phases": [
            {"service_name": "ec2", "operation_name": "TerminateInstances", "region_name": "us-east-1",
             "params": {"InstanceIds": instances}},
            {"required_verification": "DescribeInstances confirms both terminated; record EBS and ENI deletion. Retry only this pair if required."},
            {"service_name": "cloudformation", "operation_name": "DeleteStack", "region_name": "us-east-1",
             "params": {"StackName": stack}},
            {"required_verification": "Wait for DELETE_COMPLETE; confirm the recorded expiry schedule, roles, profile, network and placement group are removed."},
            {"service_name": "s3", "operation_name": "DeleteObjects", "region_name": "us-east-1",
             "params": {"Bucket": ledger["bucket"], "Delete": {"Objects": [{"Key": key} for key in keys]}}},
            {"required_verification": "Inspect all DeleteObjects errors; list this prefix again. Delete the dedicated storage stack only after all its experiment prefixes are preserved and empty."},
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--availability-zone", default="us-east-1a")
    parser.add_argument("--approval-reference", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = launch_packet(args.kit, bucket=args.bucket, prefix=args.prefix,
                           availability_zone=args.availability_zone, approval_reference=args.approval_reference)
    if args.output.exists():
        raise ValueError("A launch packet must not overwrite an earlier attempt")
    write_json(args.output, packet)
