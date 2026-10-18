"""Prepare and verify the three-disk native pair without making AWS calls."""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile

from ..export_task_inputs import checked_pair, identity
from ..training_inputs import require
from .build_profile import SOURCE_NAMES, FIRST_LINEAGE_SOURCES
from .common import ARCHES, check_file
from .infrastructure import INSTANCE, storage_template, validate_launch
from .profile_infrastructure import selected_architectures, selected_x86_instance, worker_template
from .trained_inputs import validate_catalog
from . import first_lineage_profile


LABELS = {"x86_64": "X86", "arm64": "Arm"}
DISKS = {"root": ("/dev/sda1", 8), "data": ("/dev/sdf", 22), "control": ("/dev/sdg", 2)}
REGIONS = ("us-east-1", "us-east-2", "us-west-1", "us-west-2")


def read(path):
    return json.loads(Path(path).read_bytes())


def bound_architectures(record, kind):
    if record["schema"] == "organism-native-profile-" + kind + "-v1":
        require("architectures" not in record and "first_lineage" not in record
                and "x86_instance_type" not in record,
                "Legacy native profile cannot override its pair or observer")
        return ARCHES
    version = record["schema"].removeprefix("organism-native-profile-" + kind + "-")
    require(version in ("v2", "v3")
            and record["schema"] == "organism-native-profile-" + kind + "-" + version,
            "Unknown native profile schema")
    architectures = selected_architectures(record.get("architectures"))
    require(architectures == ("x86_64",), "Native single profile must bind one x86-64 device")
    if version == "v3":
        require(record.get("first_lineage") == first_lineage_profile.descriptor(),
                "Native first-lineage observer changed its cases or allowances")
        selected_x86_instance(record.get("x86_instance_type"), first_lineage=True)
    else:
        require("first_lineage" not in record and "x86_instance_type" not in record,
                "Native v2 cannot override its observer or hardware")
    return architectures


def checked_sources(directory, build):
    """Read every carried source without extracting it or trusting receipt hashes."""
    sources = FIRST_LINEAGE_SOURCES if "first_lineage" in build else SOURCE_NAMES
    expected_names = {"organism/" + name for name in sources}
    require(set(build["sources"]) == set(sources), "Native source inventory differs")
    require((directory / "kit.tar.gz").stat().st_size <= 2 * 1024**2,
            "Native source archive exceeds the bootstrap bound")
    seen, total = set(), 0
    workspace = Path(__file__).resolve().parents[2]
    with tarfile.open(directory / "kit.tar.gz", "r:gz") as archive:
        for member in archive:
            require(member.name in expected_names and member.name not in seen
                    and member.isfile() and 0 <= member.size <= 1024**2,
                    "Native source archive has an unexpected, duplicate or invalid member")
            total += member.size
            require(total <= 8 * 1024**2, "Expanded native source archive exceeds its bound")
            with archive.extractfile(member) as stream:
                raw = stream.read(member.size + 1)
            observed = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            require(len(raw) == member.size
                    and observed == build["sources"][member.name.removeprefix("organism/")]
                    and observed == identity(workspace / member.name),
                    "Carried native source differs from its receipt or current implementation")
            seen.add(member.name)
    require(seen == expected_names, "Native source archive is incomplete")


def checked_kit(directory):
    directory = Path(directory).resolve()
    build = read(directory / "build-receipt.json")
    architectures = bound_architectures(build, "kit")
    first_lineage = "first_lineage" in build
    require(build["independent_instance_permits_required"] is True
            and build["qualification_passed"] is False, "Expected the trained native profile kit")
    require(read(directory / "worker-template.json") == worker_template(
        architectures=architectures, first_lineage=first_lineage,
        x86_instance_type=build.get("x86_instance_type"))
            and identity(directory / "worker-template.json")["sha256"] == build["worker_template_sha256"],
            "Native profile template changed")
    require(read(directory / "storage-template.json") == storage_template()
            and identity(directory / "storage-template.json")["sha256"] == build["storage_template_sha256"],
            "Native artifact storage template changed")
    catalog_path = directory / "catalog.json"
    require(identity(catalog_path)["sha256"] == build["catalog_sha256"]
            and identity(directory / "kit.tar.gz")["sha256"] == build["kit_sha256"]
            and catalog_path.read_bytes() == (directory / "candidate/trained-catalog.json").read_bytes(),
            "Native kit or trained catalog changed")
    checked_sources(directory, build)
    catalog = validate_catalog(read(catalog_path))
    candidate = read(directory / "candidate/receipt.json")
    require(candidate["catalog"] == identity(catalog_path)
            and candidate["release_id"] == build["release_id"] == catalog["release_id"]
            and candidate["signed_provenance_verified"] is True, "Candidate binding changed")
    # Reconstruct the actual export immediately before preparing a cloud job.
    pair = checked_pair(candidate["export_pair"]["export_directory"],
                        catalog["trained_provenance"]["training_admission"])
    require(pair == candidate["export_pair"], "Prepared candidate belongs to another export")
    proof = catalog["trained_provenance"]
    require(proof["export_audit"] == pair["audit"] and proof["export_workload"] == pair["workload"]
            and proof["total_stored_parameters"] == pair["total_stored_parameters"]
            and proof["model"] == {k: pair["models"]["trained-merged"][k] for k in ("bytes", "sha256")},
            "Signed native provenance differs from the reconstructed export")
    for name in ("quantization_precision", "response_audit", "export_job", "export_report"):
        require(proof.get(name) == pair.get(name), "Signed native precision provenance differs")
    capsule = Path(candidate["source_capsule"]) / "capsule"
    expected = {
        "inputs/capsule/" + name: {"path": str(capsule / name), **entry}
        for name, entry in catalog["files"].items()}
    expected.update({"inputs/" + name: {"path": str(directory / name), **identity(directory / name)}
                     for name in ("kit.tar.gz", "catalog.json")})
    if first_lineage:
        name = first_lineage_profile.CASES_NAME
        first_lineage_profile.validate_cases(directory / name)
        expected["inputs/" + name] = {"path": str(directory / name), **identity(directory / name)}
    inventory = read(directory / "upload-inventory.json")
    entries = inventory["files"]
    require(len(entries) == len(expected) and len({e["key_suffix"] for e in entries}) == len(entries),
            "Native input inventory is incomplete or duplicated")
    for entry in entries:
        require(entry["key_suffix"] in expected
                and {k: v for k, v in entry.items() if k != "key_suffix"} == expected[entry["key_suffix"]],
                "Native input path or identity differs from the signed candidate")
        check_file(entry["path"], entry)
    require(inventory["total_upload_bytes"] == sum(e["bytes"] for e in entries),
            "Native upload total differs")
    return build, expected


def launch_packet(directory, *, bucket, prefix, region, zone, images, account_id,
                  deadline, approval_reference, now=None):
    require(region in REGIONS and re.fullmatch(re.escape(region) + "[a-z]", zone),
            "Native pair must use an explicitly selected U.S. commercial zone")
    require(re.fullmatch(r"[0-9]{12}", account_id)
            and re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket), "Invalid account or artifact bucket")
    require(isinstance(approval_reference, str) and approval_reference.strip(), "Spending authority is required")
    require(validate_launch(prefix, deadline, now=now) >= 3600,
            "Allow at least one hour for native preparation, execution and cleanup")
    build, inputs = checked_kit(directory)
    architectures = bound_architectures(build, "kit")
    first_lineage = "first_lineage" in build
    if first_lineage:
        require(validate_launch(prefix, deadline, now=now) >=
                first_lineage_profile.MINIMUM_REMAINING_SECONDS + 600,
                "Insufficient fixed lease for first-lineage preparation and observation")
    require(set(images) == set(architectures)
            and all(re.fullmatch(r"ami-[0-9a-f]{17}", v) for v in images.values()),
            "Explicit native image identities must match the prepared devices")
    parameters = {
        "ArtifactBucket": bucket, "RunPrefix": prefix, "AvailabilityZone": zone,
        "KitSha256": build["kit_sha256"], "CatalogSha256": build["catalog_sha256"],
        "DeadlineDate": deadline.split("T")[0], "DeadlineTime": deadline.split("T")[1],
        **{LABELS[a] + "ImageId": images[a] for a in architectures},
    }
    if first_lineage:
        parameters["CasesSha256"] = first_lineage_profile.CASES_SHA256
    packet = {
        "schema": "organism-native-profile-launch-v1", "region": region, "account_id": account_id,
        "deadline": deadline, "approval_reference": approval_reference, "parameters": parameters,
        "input_identities": {n: {k: v[k] for k in ("bytes", "sha256")} for n, v in inputs.items()},
        "prelaunch_requirements": [
            "Verify current images, offerings, quota usage and price for each selected instance type.",
            "Record these devices within standing spending authority, current reservations and aggregate CPU-hour bounds.",
            "Verify the dedicated artifact bucket's actual owner, region, public-access block, encryption, ownership and TLS-only policy.",
            "Verify every uploaded input and retain this original expiry across any provisioning retry.",
            "Read all deployed controls and complete role policies before constructing any instance permit.",
        ],
        "request": {
            "service_name": "cloudformation", "operation_name": "CreateStack", "region_name": region,
            "params": {
                "StackName": prefix, "TemplateBody": json.dumps(worker_template(
                    architectures=architectures, first_lineage=first_lineage,
                    x86_instance_type=build.get("x86_instance_type"))),
                "Parameters": [{"ParameterKey": n, "ParameterValue": v} for n, v in parameters.items()],
                "Capabilities": ["CAPABILITY_IAM"], "OnFailure": "ROLLBACK", "TimeoutInMinutes": 15,
                "Tags": [{"Key": "Purpose", "Value": "organism-native-preflight"},
                         {"Key": "Experiment", "Value": prefix}],
            },
        },
    }
    if architectures != ARCHES:
        packet.update(schema="organism-native-profile-launch-v2", architectures=list(architectures))
    if first_lineage:
        packet.update(schema="organism-native-profile-launch-v3",
                      first_lineage=first_lineage_profile.descriptor())
    if "x86_instance_type" in build:
        packet["x86_instance_type"] = build["x86_instance_type"]
    return packet


def single(rows, message):
    require(len(rows) == 1, message)
    return rows[0]


def checked_role(observation, *, name, account, trust, policy_name, policy):
    role = observation["role"]["Role"]
    require(role["RoleName"] == name and role["Arn"] == "arn:aws:iam::" + account + ":role/" + name
            and not role.get("PermissionsBoundary") and role["AssumeRolePolicyDocument"] == trust,
            "Native role identity or trust differs")
    require(observation["attached"]["AttachedPolicies"] == []
            and observation["inline_names"]["PolicyNames"] == [policy_name]
            and set(observation["inline"]) == {policy_name}, "Unexpected native role policies")
    observed = observation["inline"][policy_name]
    require(observed["RoleName"] == name and observed["PolicyName"] == policy_name
            and observed["PolicyDocument"] == policy, "Native role permissions differ")
    return role["Arn"]


def permits(packet, observed, *, now=None):
    """Verify actual AWS API responses and return the selected unpublished permits."""
    now = now or datetime.now(timezone.utc)
    architectures = bound_architectures(packet, "launch")
    first_lineage = "first_lineage" in packet
    count = len(architectures)
    expected_template = worker_template(
        architectures=architectures, first_lineage=first_lineage,
        x86_instance_type=packet.get("x86_instance_type"))
    expected_types = {architecture: INSTANCE[architecture] for architecture in architectures}
    expected_types["x86_64"] = selected_x86_instance(
        packet.get("x86_instance_type"), first_lineage=first_lineage)
    region, account, parameters = packet["region"], packet["account_id"], packet["parameters"]
    require(region in REGIONS
            and re.fullmatch(r"[0-9]{12}", account), "Invalid native launch binding")
    require(set(parameters) == set(expected_template["Parameters"])
            and re.fullmatch(re.escape(region) + "[a-z]", parameters["AvailabilityZone"])
            and all(re.fullmatch(r"[0-9a-f]{64}", parameters[n]) for n in ("KitSha256", "CatalogSha256")),
            "Native launch parameter structure differs")
    for name, parameter in (("kit.tar.gz", "KitSha256"), ("catalog.json", "CatalogSha256")):
        require(packet["input_identities"]["inputs/" + name]["sha256"] == parameters[parameter],
                "Native input checksum differs from its launch parameter")
    if first_lineage:
        require(parameters["CasesSha256"] == first_lineage_profile.CASES_SHA256
                and packet["input_identities"].get("inputs/" + first_lineage_profile.CASES_NAME)
                == first_lineage_profile.descriptor()["cases"],
                "Native observer cases differ from their frozen input")
    prefix, bucket = parameters["RunPrefix"], parameters["ArtifactBucket"]
    require(packet["deadline"] == parameters["DeadlineDate"] + "T" + parameters["DeadlineTime"],
            "Native deadline binding changed")
    minimum = first_lineage_profile.MINIMUM_REMAINING_SECONDS + 300 if first_lineage else 2100
    require(validate_launch(prefix, packet["deadline"], now=now) >= minimum,
            "Insufficient time remains before the fixed native expiry")
    require(packet["approval_reference"].strip(), "Native spending authority is missing")
    age = (now - datetime.fromisoformat(observed["observed_at"])).total_seconds()
    require(0 <= age <= 300, "Native control observations must be current")
    template = observed["template"]["TemplateBody"]
    template = json.loads(template) if isinstance(template, str) else template
    require(template == expected_template, "Deployed native template differs")
    stack = single(observed["stack"]["Stacks"], "Ambiguous native stack")
    require(stack["StackStatus"] == "CREATE_COMPLETE" and stack["StackName"] == prefix
            and re.fullmatch(r"arn:aws:cloudformation:" + re.escape(region) + ":" + account
                             + ":stack/" + re.escape(prefix) + r"/[A-Za-z0-9-]+", stack["StackId"]),
            "Native stack identity or status differs")
    require(len(stack["Parameters"]) == len(parameters)
            and {p["ParameterKey"]: p["ParameterValue"] for p in stack["Parameters"]} == parameters,
            "Deployed native parameters differ")
    rows = observed["resources"]["StackResources"]
    resources = {r["LogicalResourceId"]: r for r in rows}
    require(len(rows) == len(resources) and set(resources) == set(template["Resources"]),
            "Native resource inventory is incomplete or duplicated")
    for name, resource in resources.items():
        require(resource["StackId"] == stack["StackId"] and resource["ResourceStatus"] == "CREATE_COMPLETE"
                and resource["ResourceType"] == template["Resources"][name]["Type"],
                "A native resource belongs to another stack or is incomplete")
    ids = {name: r["PhysicalResourceId"] for name, r in resources.items()}
    instances = [i for r in observed["instances"]["Reservations"] for i in r["Instances"]]
    require(len(instances) == count and len({i["InstanceId"] for i in instances}) == count,
            "Native instance count differs from the prepared devices")
    require(all(r["OwnerId"] == account for r in observed["instances"]["Reservations"]),
            "Native instances belong to another account")
    types = {v["InstanceType"]: v for v in observed["instance_types"]["InstanceTypes"]}
    images = {v["ImageId"]: v for v in observed["images"]["Images"]}
    require(len(observed["instance_types"]["InstanceTypes"]) == len(types) == count
            and set(types) == {expected_types[a] for a in architectures}
            and len(observed["images"]["Images"]) == len(images) == count
            and set(images) == {parameters[LABELS[a] + "ImageId"] for a in architectures},
            "Native type or image inventory differs")
    volumes = observed["volumes"]["Volumes"]
    interfaces = observed["interfaces"]["NetworkInterfaces"]
    require(len(volumes) == 3 * count and len({v["VolumeId"] for v in volumes}) == 3 * count
            and len(interfaces) == count, "Native devices require exactly three disks and one interface each")
    group = single(observed["security_group"]["SecurityGroups"], "Ambiguous native security group")
    require(group["GroupId"] == ids["SecurityGroup"] and group["VpcId"] == ids["Vpc"]
            and not group["IpPermissions"], "Native inbound policy differs")
    require(sorted(group["IpPermissionsEgress"], key=lambda p: p["FromPort"]) == [
        {"IpProtocol": "tcp", "FromPort": p, "ToPort": p, "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
         "Ipv6Ranges": [], "PrefixListIds": [], "UserIdGroupPairs": []} for p in (80, 443)],
        "Native outbound policy differs")
    spread = single(observed["placement_group"]["PlacementGroups"], "Ambiguous native placement group")
    require(spread["GroupName"] == ids["Spread"] and spread["Strategy"] == "spread"
            and spread["State"] == "available", "Independent native placement differs")
    require(set(observed["inputs"]) == set(packet["input_identities"]), "Native S3 input inventory differs")
    for name, expected in packet["input_identities"].items():
        head = observed["inputs"][name]
        require(head["ContentLength"] == expected["bytes"] and head["ChecksumType"] == "FULL_OBJECT"
                and base64.b64decode(head["ChecksumSHA256"], validate=True).hex() == expected["sha256"]
                and head["ServerSideEncryption"] == "AES256", "Native S3 input identity differs")

    result, used_volumes, used_interfaces = {}, set(), set()
    worker_trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    for architecture in architectures:
        label = LABELS[architecture]
        instance = single([i for i in instances if i["InstanceId"] == ids[label + "Worker"]],
                          "Native instance is missing from its stack")
        iid, image_id = instance["InstanceId"], parameters[label + "ImageId"]
        require(re.fullmatch(r"i-[0-9a-f]{17}", iid) and instance["State"]["Name"] == "running"
                and instance["Architecture"] == architecture and instance["ImageId"] == image_id
                and instance["InstanceType"] == expected_types[architecture], "Native instance profile differs")
        description, image = types[expected_types[architecture]], images[image_id]
        require(description["MemoryInfo"]["SizeInMiB"] == 8192
                and description["VCpuInfo"]["DefaultVCpus"] == 4
                and architecture in description["ProcessorInfo"]["SupportedArchitectures"],
                "Native instance capacity differs")
        if expected_types[architecture] == "c7a.xlarge":
            require(description["VCpuInfo"].get("DefaultCores") == 4
                    and description["VCpuInfo"].get("DefaultThreadsPerCore") == 1
                    and instance.get("CpuOptions") == {"CoreCount": 4, "ThreadsPerCore": 1},
                    "Explicit C7a physical-core topology differs")
        require(image["Architecture"] == architecture and image["State"] == "available"
                and image["OwnerId"] == "099720109477", "Native image architecture, owner or state differs")
        require(instance["Placement"]["AvailabilityZone"] == parameters["AvailabilityZone"]
                and instance["Placement"]["GroupName"] == ids["Spread"]
                and instance["VpcId"] == ids["Vpc"] and instance["SubnetId"] == ids["Subnet"],
                "Native instance placement or network differs")
        require(instance["MetadataOptions"]["HttpTokens"] == "required"
                and instance["MetadataOptions"]["HttpPutResponseHopLimit"] == 1
                and instance["MetadataOptions"]["State"] == "applied"
                and observed["shutdown"][iid]["InstanceInitiatedShutdownBehavior"]["Value"] == "terminate",
                "Native metadata or shutdown policy differs")
        require(instance["RootDeviceName"] == DISKS["root"][0] and instance["RootDeviceType"] == "ebs",
                "Native root device differs")
        mappings = instance["BlockDeviceMappings"]
        require(len(mappings) == 3 and {m["DeviceName"] for m in mappings} == {v[0] for v in DISKS.values()},
                "Native instance block mapping differs")
        disk_ids = {}
        for role, (device, size) in DISKS.items():
            mapping = single([m for m in mappings if m["DeviceName"] == device], "Ambiguous native disk role")["Ebs"]
            volume = single([v for v in volumes if v["VolumeId"] == mapping["VolumeId"]], "Missing native disk")
            attachment = single(volume["Attachments"], "Native disk has another attachment")
            require(re.fullmatch(r"vol-[0-9a-f]{17}", volume["VolumeId"])
                    and volume["VolumeId"] not in used_volumes
                    and volume["Size"] == size and volume["VolumeType"] == "gp3" and volume["Encrypted"] is True
                    and volume["AvailabilityZone"] == parameters["AvailabilityZone"]
                    and volume["State"] == "in-use" and mapping["Status"] == "attached"
                    and mapping["DeleteOnTermination"] is True
                    and attachment["InstanceId"] == iid and attachment["Device"] == device
                    and attachment["State"] == "attached" and attachment["DeleteOnTermination"] is True,
                    "Native disk identity, size, encryption or cleanup policy differs")
            disk_ids[role] = volume["VolumeId"]
            used_volumes.add(volume["VolumeId"])
        interface = single([e for e in interfaces if e["Attachment"]["InstanceId"] == iid],
                           "Native interface is missing or duplicated")
        require(interface["NetworkInterfaceId"] not in used_interfaces
                and re.fullmatch(r"eni-[0-9a-f]{17}", interface["NetworkInterfaceId"])
                and interface["OwnerId"] == account and interface["Status"] == "in-use"
                and interface["Attachment"]["DeleteOnTermination"] is True
                and interface["Attachment"]["DeviceIndex"] == 0
                and interface["SubnetId"] == ids["Subnet"] and interface["VpcId"] == ids["Vpc"]
                and [g["GroupId"] for g in interface["Groups"]] == [ids["SecurityGroup"]],
                "Native interface identity or policy differs")
        used_interfaces.add(interface["NetworkInterfaceId"])
        object_arn = "arn:aws:s3:::" + bucket + "/" + prefix
        role_arn = checked_role(observed["roles"][label + "Role"], name=ids[label + "Role"], account=account,
                               trust=worker_trust, policy_name="NativeArtifactsOnly", policy={
            "Version": "2012-10-17", "Statement": [
                {"Effect": "Allow", "Action": "s3:GetObject", "Resource": object_arn + "/inputs/*"},
                {"Effect": "Allow", "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
                 "Resource": object_arn + "/outputs/" + architecture + "/*"}]})
        profile = observed["profiles"][label]["InstanceProfile"]
        require(profile["InstanceProfileName"] == ids[label + "Profile"]
                and profile["Arn"] == "arn:aws:iam::" + account + ":instance-profile/" + ids[label + "Profile"]
                and profile["Arn"] == instance["IamInstanceProfile"]["Arn"]
                and [r["Arn"] for r in profile["Roles"]] == [role_arn], "Native instance profile differs")
        result[architecture] = {
            "schema": "organism-native-profile-permit-v1", "instance_id": iid, "architecture": architecture,
            "prefix": prefix, "region": region, "deadline": packet["deadline"],
            "kit_sha256": parameters["KitSha256"], "catalog_sha256": parameters["CatalogSha256"],
            "volumes": disk_ids, "independent_expiry_configuration_verified": True,
            "schedule_arn": observed["schedule"]["Arn"], "approval_reference": packet["approval_reference"],
        }
        if first_lineage:
            result[architecture].update(schema="organism-native-profile-permit-v2",
                                        cases_sha256=first_lineage_profile.CASES_SHA256)
    schedule = observed["schedule"]
    require(schedule["Name"] == ids["Expiry"]
            and schedule["Arn"] == "arn:aws:scheduler:" + region + ":" + account + ":schedule/default/" + ids["Expiry"]
            and schedule["State"] == "ENABLED" and schedule["ScheduleExpressionTimezone"] == "UTC"
            and schedule["ScheduleExpression"] == "at(" + packet["deadline"] + ")"
            and schedule["FlexibleTimeWindow"] == {"Mode": "OFF"}, "Native expiry configuration differs")
    instance_ids = [result[a]["instance_id"] for a in architectures]
    target = schedule["Target"]
    require(target["Arn"] == "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances"
            and json.loads(target["Input"]) == {"InstanceIds": instance_ids}
            and target["RetryPolicy"] == {"MaximumEventAgeInSeconds": 300, "MaximumRetryAttempts": 2},
            "Native expiry must terminate exactly the prepared devices")
    role_arn = checked_role(observed["roles"]["ExpiryRole"], name=ids["ExpiryRole"], account=account,
                            trust={"Version": "2012-10-17", "Statement": [{
                                "Effect": "Allow", "Principal": {"Service": "scheduler.amazonaws.com"},
                                "Action": "sts:AssumeRole", "Condition": {"StringEquals": {"aws:SourceAccount": account}}}]},
                            policy_name="TerminateOnlyThisPair" if count == 2 else "TerminateOnlyThisDevice", policy={
                                "Version": "2012-10-17", "Statement": [{
                                    "Effect": "Allow", "Action": "ec2:TerminateInstances",
                                    "Resource": ["arn:aws:ec2:" + region + ":" + account + ":instance/" + iid
                                                 for iid in instance_ids]}]})
    require(target["RoleArn"] == role_arn, "Native expiry uses an uninspected role")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create a reviewable request from a passing trained kit")
    prepare.add_argument("--kit", type=Path, required=True)
    for name in ("bucket", "prefix", "region", "zone", "x86-image",
                 "account-id", "deadline", "approval-reference"):
        prepare.add_argument("--" + name, required=True)
    prepare.add_argument("--arm-image", help="Required only for an original paired-architecture kit")
    prepare.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="Check fresh AWS observations and write unpublished permits")
    for name in ("packet", "observations", "output"):
        verify.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "Output must be new")
    if args.command == "prepare":
        value = launch_packet(
            args.kit, bucket=args.bucket, prefix=args.prefix, region=args.region, zone=args.zone,
            images={"x86_64": args.x86_image, **({"arm64": args.arm_image} if args.arm_image else {})},
            account_id=args.account_id,
            deadline=args.deadline, approval_reference=args.approval_reference)
    else:
        value = {
            "schema": "organism-native-profile-permit-readback-v1",
            "permits": permits(read(args.packet), read(args.observations)),
            "packet": identity(args.packet), "observations": identity(args.observations),
            "configuration_verified": True, "permits_published": False,
            "native_execution_verified": False, "qualification_passed": False,
        }
    # Validation completes before output creation; no request or permit is sent.
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
