"""Single CPU export host and exact-instance expiry; generates no AWS calls."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .native_preflight.infrastructure import storage_template as native_storage, worker_template as native_worker
from .training_inputs import load, require, verified_path
from .training_worker.operations import verify_expiry_role
from .export_worker import STANDARD_PROFILE, limits_for_profile, lease_envelope, profile_for_job
from .gguf_export import canonical


AMI = "ami-0fa5967347d08d2df"
INSTANCE = "r7i.2xlarge"
MAX_LEASE_SECONDS = 21600
MIN_LEASE_SECONDS = 15900


def validate_launch(prefix, deadline, *, now=None, worker_profile=STANDARD_PROFILE):
    require(re.fullmatch(r"export-[0-9]{8}-[a-z0-9]{4,12}", prefix), "Distinct CPU export prefix required")
    moment = datetime.strptime(deadline, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    seconds = (moment - (now or datetime.now(timezone.utc))).total_seconds()
    envelope = lease_envelope(worker_profile)
    minimum = envelope["workload_seconds"] + envelope["cleanup_seconds"] + envelope["preparation_seconds"]
    require(minimum <= seconds <= envelope["maximum_lease_seconds"],
            "CPU export lease must preserve preparation, the selected workload and cleanup within its profile")
    return seconds


def user_data():
    return """#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/organism-export-bootstrap.log /dev/console) 2>&1
trap 'systemctl poweroff --no-block' EXIT
export PYTHONDONTWRITEBYTECODE=1
cat > /etc/systemd/system/organism-export-expiry.service <<'EOF'
[Unit]
Description=Terminate this temporary organism CPU export host
[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl poweroff --no-block
EOF
cat > /etc/systemd/system/organism-export-expiry.timer <<'EOF'
[Unit]
Description=Absolute expiry of this temporary organism CPU export host
[Timer]
OnCalendar=${DeadlineDate} ${DeadlineTime} UTC
Persistent=true
AccuracySec=1s
Unit=organism-export-expiry.service
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now organism-export-expiry.timer
export DEBIAN_FRONTEND=noninteractive
timeout 600 apt-get update -q
timeout 600 apt-get install -y --no-install-recommends python3-boto3 python3-venv
useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin organism-export
mkdir -p /srv/organism-export/workspace
python3 - <<'PY'
import boto3, hashlib, pathlib, tarfile
s3 = boto3.client("s3", region_name="us-east-1")
response = s3.get_object(Bucket="${ArtifactBucket}", Key="${RunPrefix}/inputs/kit.tar.gz")
path = pathlib.Path("/srv/organism-export/kit.tar.gz")
try:
    if not 0 < response["ContentLength"] <= 64 * 1024**2:
        raise ValueError("CPU export source kit exceeds its bound")
    digest, total = hashlib.sha256(), 0
    with path.open("xb") as target:
        while block := response["Body"].read(1024**2):
            total += len(block)
            if total > response["ContentLength"]:
                raise ValueError("CPU export source kit exceeds its declared length")
            digest.update(block)
            target.write(block)
    if total != response["ContentLength"] or digest.hexdigest() != "${KitSha256}":
        raise ValueError("CPU export source kit identity differs")
finally:
    response["Body"].close()
with tarfile.open(path, "r:gz") as archive:
    members, names, total = [], set(), 0
    for item in archive:
        total += item.size
        if len(members) >= 256 or total > 256 * 1024**2 or not item.isfile() or item.name in names:
            raise ValueError("CPU export source extraction exceeds its contract")
        names.add(item.name)
        members.append(item)
    archive.extractall("/srv/organism-export/workspace", members=members, filter="data")
PY
cd /srv/organism-export/workspace
python3 -m organism.export_host --bucket '${ArtifactBucket}' --prefix '${RunPrefix}' --deadline '${DeadlineDate}T${DeadlineTime}' --kit-sha256 '${KitSha256}'
"""


def storage_template():
    template = deepcopy(native_storage())
    template["Description"] = "Temporary private inputs and outputs for one authorized CPU model export"
    template["Resources"]["Artifacts"]["Properties"]["Tags"] = [{"Key": "Purpose", "Value": "organism-model-export"}]
    return template


def worker_template():
    template = deepcopy(native_worker())
    resources = template["Resources"]
    for name in ("ArmRole", "ArmProfile", "ArmWorker", "Spread"):
        resources.pop(name)
    instance = resources["X86Worker"]["Properties"]
    instance.update(ImageId=AMI, InstanceType=INSTANCE)
    instance.pop("PlacementGroupName")
    instance["BlockDeviceMappings"] = [{"DeviceName": "/dev/sda1", "Ebs": {
        "VolumeSize": 192, "VolumeType": "gp3", "Encrypted": True, "DeleteOnTermination": True}}]
    instance["UserData"] = {"Fn::Base64": {"Fn::Sub": user_data()}}
    instance["Tags"][0]["Value"] = "organism-model-export"
    policy = resources["X86Role"]["Properties"]["Policies"][0]
    policy["PolicyName"] = "ExportArtifactsOnly"
    policy["PolicyDocument"]["Statement"][1]["Resource"]["Fn::Sub"] = (
        "arn:${AWS::Partition}:s3:::${ArtifactBucket}/${RunPrefix}/outputs/*")
    expiry_policy = resources["ExpiryRole"]["Properties"]["Policies"][0]
    expiry_policy["PolicyName"] = "TerminateOnlyThisExportHost"
    expiry_policy["PolicyDocument"]["Statement"][0]["Resource"] = [
        {"Fn::Sub": "arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:instance/${X86Worker}"}]
    resources["Expiry"]["Properties"]["Description"] = "Independent termination of this bounded CPU export host"
    resources["Expiry"]["Properties"]["Target"]["Input"] = {"Fn::Sub": '{"InstanceIds":["${X86Worker}"]}'}
    template["Description"] = "One bounded CPU merge/export job; no training or device qualification"
    template["Parameters"].pop("CatalogSha256")
    template["Parameters"]["RunPrefix"]["AllowedPattern"] = "export-[0-9]{8}-[a-z0-9]{4,12}"
    template["Outputs"] = {"ExportInstanceId": {"Value": {"Ref": "X86Worker"}},
                           "ExpirySchedule": {"Value": {"Ref": "Expiry"}}}
    return json.loads(json.dumps(template).replace("X86", "Export"))


def permit_packet(*, prefix, instance_id, deadline, kit_sha256, approval_reference,
                  schedule, role, inline_policies, attached_policies, now=None,
                  worker_profile=STANDARD_PROFILE):
    # Permit inspection can happen after launch; only the fixed upper expiry
    # bound is checked here. The host separately reserves the complete workload.
    require(re.fullmatch(r"export-[0-9]{8}-[a-z0-9]{4,12}", prefix)
            and re.fullmatch(r"i-[0-9a-f]{17}", instance_id)
            and re.fullmatch(r"[0-9a-f]{64}", kit_sha256), "Invalid CPU export permit identity")
    moment = datetime.strptime(deadline, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    seconds = (moment - (now or datetime.now(timezone.utc))).total_seconds()
    envelope = lease_envelope(worker_profile)
    require(envelope["workload_seconds"] + envelope["cleanup_seconds"] < seconds
            <= envelope["maximum_lease_seconds"], "CPU export permit leaves insufficient workload/cleanup time")
    require(schedule["State"] == "ENABLED" and schedule["ScheduleExpressionTimezone"] == "UTC"
            and schedule["ScheduleExpression"] == "at(" + deadline + ")"
            and schedule["FlexibleTimeWindow"] == {"Mode": "OFF"}, "CPU export expiry is disabled or changed")
    target = schedule["Target"]
    require(target["Arn"] == "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances"
            and json.loads(target["Input"]) == {"InstanceIds": [instance_id]}
            and target["RetryPolicy"] == {"MaximumEventAgeInSeconds": 300, "MaximumRetryAttempts": 2},
            "CPU export expiry does not terminate exactly this instance")
    verify_expiry_role(schedule, instance_id, role, inline_policies, attached_policies, region="us-east-1")
    require(isinstance(approval_reference, str) and approval_reference.strip(), "CPU export scope approval required")
    permit = {"schema": "organism-cpu-export-permit-v1", "instance_id": instance_id,
            "prefix": prefix, "deadline": deadline, "kit_sha256": kit_sha256,
            "schedule_arn": schedule["Arn"], "independent_expiry_configuration_verified": True,
            "schedule_invocation_verified": False, "approval_reference": approval_reference}
    if worker_profile != STANDARD_PROFILE:
        permit["worker_profile"] = worker_profile
    return permit


def launch_packet(kit, *, bucket, prefix, zone, deadline, approval_reference, provisioning_request=1, now=None):
    require(isinstance(approval_reference, str) and approval_reference.strip(), "Direct CPU export scope approval required")
    require(re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket)
            and zone in ("us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d", "us-east-1f"),
            "CPU export bucket or region offering differs")
    require(type(provisioning_request) is int and 1 <= provisioning_request <= 4,
            "CPU export allows at most four provisioning requests and one allocated lease")
    kit = Path(kit)
    build = load(kit / "build-receipt.json")
    require(build.get("schema") == "organism-cpu-export-build-v1" and build.get("completed_training_bound") is True,
            "A package bound to real completed training is required")
    for name, entry in build["artifacts"].items():
        verified_path(kit, {"path": name, **entry})
    profile = profile_for_job(build)
    manifest = load(kit / "export-input.json")
    require(profile_for_job(manifest) == profile
            and canonical(manifest["limits"]) == canonical(limits_for_profile(profile)),
            "CPU export build and input profiles differ")
    seconds = validate_launch(prefix, deadline, now=now, worker_profile=profile)
    require(load(kit / "worker-template.json") == worker_template(), "CPU export template differs from its reviewed source")
    day, moment = deadline.split("T")
    parameters = {"ArtifactBucket": bucket, "RunPrefix": prefix, "AvailabilityZone": zone,
                  "KitSha256": build["artifacts"]["kit.tar.gz"]["sha256"],
                  "DeadlineDate": day, "DeadlineTime": moment}
    return {
        "schema": "organism-cpu-export-launch-packet-v1", "approval_reference": approval_reference,
        "region": "us-east-1", "deadline": deadline, "remaining_lease_seconds": seconds,
        "provisioning_request": provisioning_request, "maximum_allocated_leases": 1,
        "required_live_checks": [
            "Bind this job to the user's standing authorization for in-scope work below $400.",
            ("Reconcile prior cost and instance-hours; reserve at most $20 for this first-lineage export while keeping combined project spending and reservations below $400 and CPU work below the aggregate ceiling recorded in SPENDING_AUTHORIZATION.md."
             if profile != STANDARD_PROFILE else
             "Reconcile prior cost and instance-hours; reserve at most $10 for this job while keeping combined project spending and reservations below $400 and CPU work below the aggregate ceiling recorded in SPENDING_AUTHORIZATION.md."),
            "No CPU export lease has already been allocated; preserve the same original deadline across capacity requests.",
            "Revalidate instance type, image owner/state, regional offering, current price and relevant quota before launch.",
            "All three private S3 inputs match the final source-kit and external-input byte counts and SHA-256 identities.",
            "Preserve actual resource IDs; verify the exact-instance schedule, role trust and all policies before publishing the permit.",
        ],
        "request": {"service_name": "cloudformation", "operation_name": "CreateStack", "region_name": "us-east-1",
                    "params": {"StackName": prefix, "TemplateBody": (kit / "worker-template.json").read_text(),
                               "Parameters": [{"ParameterKey": key, "ParameterValue": value} for key, value in parameters.items()],
                               "Capabilities": ["CAPABILITY_IAM"], "OnFailure": "ROLLBACK", "TimeoutInMinutes": 15,
                               "Tags": [{"Key": "Purpose", "Value": "organism-model-export"},
                                        {"Key": "Experiment", "Value": prefix}]}},
    }


def cleanup_packet(ledger, *, preservation_receipt):
    require(ledger.get("schema") == "organism-cpu-export-ledger-v1"
            and ledger.get("region") == "us-east-1" and ledger.get("instance_type") == INSTANCE,
            "An exact CPU export resource ledger is required")
    require(isinstance(preservation_receipt, str) and preservation_receipt.strip(), "Preserve export evidence first")
    prefix, instance_id = ledger["prefix"], ledger["instance_id"]
    require(re.fullmatch(r"export-[0-9]{8}-[a-z0-9]{4,12}", prefix)
            and re.fullmatch(r"i-[0-9a-f]{17}", instance_id), "CPU export cleanup identity differs")
    require(re.fullmatch(r"arn:aws:cloudformation:us-east-1:[0-9]{12}:stack/" + re.escape(prefix)
                         + r"/[A-Za-z0-9-]+", ledger["worker_stack_id"]), "CPU export cleanup stack differs")
    keys = ledger["object_keys"]
    require(len(keys) <= 1000 and all(key.startswith(prefix + "/") for key in keys),
            "CPU export cleanup exceeds its recorded prefix")
    return {"preservation_receipt": preservation_receipt, "phases": [
        {"service_name": "ec2", "operation_name": "TerminateInstances", "region_name": "us-east-1",
         "params": {"InstanceIds": [instance_id]}},
        {"required_verification": "Confirm the recorded instance is terminated and its root disk and network interface are removed."},
        {"service_name": "cloudformation", "operation_name": "DeleteStack", "region_name": "us-east-1",
         "params": {"StackName": ledger["worker_stack_id"]}},
        {"required_verification": "Verify removal of the recorded worker stack, exact-instance schedule, IAM roles, profile and network."},
        {"service_name": "s3", "operation_name": "DeleteObjects", "region_name": "us-east-1",
         "params": {"Bucket": ledger["bucket"], "Delete": {"Objects": [{"Key": key} for key in keys]}}},
        {"required_verification": "Inspect deletion errors and confirm the dedicated bucket is empty before deleting its storage stack."},
    ]}
