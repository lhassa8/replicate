"""Verify actual three-device AWS controls before producing unpublished permits."""

import base64
from datetime import datetime, timezone
import json
import re

from ..training_inputs import require
from . import first_lineage_profile, lineage_network
from .infrastructure import validate_launch
from .lineage_infrastructure import LABELS, worker_template
from .profile_operations import DISKS, REGIONS, checked_role, single


def substitute(value, variables):
    """Resolve only reviewed literal references and Fn::Sub strings."""
    if isinstance(value, dict):
        if set(value) == {"Ref"}:
            return variables[value["Ref"]]
        if set(value) == {"Fn::Sub"}:
            return re.sub(r"\$\{([^}]+)\}", lambda match: variables[match.group(1)], value["Fn::Sub"])
        return {key: substitute(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute(item, variables) for item in value]
    return value


def input_readback(packet, observed):
    parameters = packet["parameters"]
    require(set(observed["inputs"]) == set(packet["input_identities"]), "Lineage S3 input inventory differs")
    for name, expected in packet["input_identities"].items():
        head = observed["inputs"][name]
        require(head["ContentLength"] == expected["bytes"] and head["ChecksumType"] == "FULL_OBJECT"
                and base64.b64decode(head["ChecksumSHA256"], validate=True).hex() == expected["sha256"]
                and head["ServerSideEncryption"] == "AES256", "A lineage S3 input changed")
    for name, field in (("kit.tar.gz", "KitSha256"), ("catalog.json", "CatalogSha256"),
                        *((f"nodes/{slot}.tar.gz", label + "BundleSha256") for slot, label in LABELS.items())):
        require(packet["input_identities"]["inputs/" + name]["sha256"] == parameters[field],
                "Lineage input differs from its launch parameter")
    require(parameters["CasesSha256"] == first_lineage_profile.CASES_SHA256
            and packet["input_identities"]["inputs/" + first_lineage_profile.CASES_NAME]
            == first_lineage_profile.descriptor()["cases"], "Lineage task cases changed")


def network_readback(observed, ids):
    group = single(observed["security_group"]["SecurityGroups"], "Ambiguous lineage security group")
    require(group["GroupId"] == ids["SecurityGroup"] and group["VpcId"] == ids["Vpc"],
            "Lineage security group belongs to another network")

    def rules(rows):
        result = []
        for row in rows:
            require(not row["Ipv6Ranges"] and not row["PrefixListIds"] and not row["UserIdGroupPairs"],
                    "Lineage network adds another address family or security-group peer")
            require(set(row) == {"IpProtocol", "FromPort", "ToPort", "IpRanges",
                                 "Ipv6Ranges", "PrefixListIds", "UserIdGroupPairs"},
                    "Lineage network permission has unexpected fields")
            for address in row["IpRanges"]:
                require(set(address) == {"CidrIp"}, "Unexpected lineage IP permission")
                result.append((row["IpProtocol"], row["FromPort"], row["ToPort"], address["CidrIp"]))
        require(len(result) == len(set(result)), "Duplicate lineage network permission")
        return sorted(result)

    peers = [("udp", lineage_network.PORT, lineage_network.PORT, ip + "/32")
             for ip in lineage_network.UNDERLAY.values()]
    require(rules(group["IpPermissions"]) == sorted(peers)
            and rules(group["IpPermissionsEgress"]) == sorted(
                peers + [("tcp", port, port, "0.0.0.0/0") for port in (80, 443)]),
            "Lineage network exposes a different peer or preparation route")


def disks(instance, observed, zone, used):
    iid = instance["InstanceId"]
    require(instance["RootDeviceName"] == DISKS["root"][0] and instance["RootDeviceType"] == "ebs",
            "Lineage root device differs")
    mappings, result = instance["BlockDeviceMappings"], {}
    require(len(mappings) == 3 and {item["DeviceName"] for item in mappings} == {v[0] for v in DISKS.values()},
            "Lineage device requires exactly its three original disks")
    for role, (device, size) in DISKS.items():
        mapping = single([item for item in mappings if item["DeviceName"] == device], "Ambiguous disk role")["Ebs"]
        volume = single([item for item in observed["volumes"]["Volumes"]
                         if item["VolumeId"] == mapping["VolumeId"]], "Missing lineage disk")
        attached = single(volume["Attachments"], "Lineage disk has another attachment")
        require(re.fullmatch(r"vol-[0-9a-f]{17}", volume["VolumeId"]) and volume["VolumeId"] not in used
                and volume["Size"] == size and volume["VolumeType"] == "gp3" and volume["Encrypted"] is True
                and volume["AvailabilityZone"] == zone and volume["State"] == "in-use"
                and mapping["Status"] == "attached" and mapping["DeleteOnTermination"] is True
                and attached["InstanceId"] == iid and attached["Device"] == device
                and attached["State"] == "attached" and attached["DeleteOnTermination"] is True,
                "Lineage disk is shared or changes its capacity, encryption or deletion policy")
        result[role] = volume["VolumeId"]
        used.add(volume["VolumeId"])
    return result


def interface(instance, slot, observed, ids, account, used):
    iid = instance["InstanceId"]
    row = single([item for item in observed["interfaces"]["NetworkInterfaces"]
                  if item["Attachment"]["InstanceId"] == iid], "Missing or repeated lineage interface")
    address = lineage_network.UNDERLAY[slot]
    require(re.fullmatch(r"eni-[0-9a-f]{17}", row["NetworkInterfaceId"])
            and row["NetworkInterfaceId"] not in used and row["OwnerId"] == account
            and row["Status"] == "in-use" and row["Attachment"]["DeleteOnTermination"] is True
            and row["Attachment"]["DeviceIndex"] == 0 and row["SubnetId"] == ids["Subnet"]
            and row["VpcId"] == ids["Vpc"] and instance["PrivateIpAddress"] == row["PrivateIpAddress"] == address
            and [item["GroupId"] for item in row["Groups"]] == [ids["SecurityGroup"]]
            and not row.get("Ipv6Addresses")
            and len(row["PrivateIpAddresses"]) == 1
            and row["PrivateIpAddresses"][0]["PrivateIpAddress"] == address
            and row["PrivateIpAddresses"][0]["Primary"] is True
            and len(instance["NetworkInterfaces"]) == 1
            and instance["NetworkInterfaces"][0]["NetworkInterfaceId"] == row["NetworkInterfaceId"],
            "Lineage interface changed its slot, network or deletion policy")
    used.add(row["NetworkInterfaceId"])
    return row["NetworkInterfaceId"]


def permits(packet, observed, *, now=None):
    """No AWS calls. Caller must separately admit model, single-device run and spend."""
    now = now or datetime.now(timezone.utc)
    template = worker_template(x86_instance_type=packet.get("x86_instance_type"),
                               seed_expiry_at=packet.get("seed_expiry_at"))
    instance_type = template["Resources"]["SeedWorker"]["Properties"]["InstanceType"]
    region, account, parameters = packet["region"], packet["account_id"], packet["parameters"]
    require(packet["schema"] == "organism-native-lineage-launch-v1"
            and region in REGIONS and re.fullmatch(r"[0-9]{12}", account)
            and set(parameters) == set(template["Parameters"])
            and re.fullmatch(re.escape(region) + "[a-z]", parameters["AvailabilityZone"])
            and re.fullmatch(r"ami-[0-9a-f]{17}", parameters["ImageId"])
            and all(re.fullmatch("[0-9a-f]{64}", parameters[name]) for name in (
                "KitSha256", "CatalogSha256", "CasesSha256",
                *(label + "BundleSha256" for label in LABELS.values()))),
            "Lineage launch parameters differ")
    prefix, bucket = parameters["RunPrefix"], parameters["ArtifactBucket"]
    require(re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket)
            and packet["deadline"] == parameters["DeadlineDate"] + "T" + parameters["DeadlineTime"]
            and validate_launch(prefix, packet["deadline"], now=now) >= 10800
            and packet["approval_reference"].strip(), "Lineage fixed lease or authority differs")
    age = (now - datetime.fromisoformat(observed["observed_at"])).total_seconds()
    require(0 <= age <= 300, "Lineage controls are stale or from the future")
    actual_template = observed["template"]["TemplateBody"]
    actual_template = json.loads(actual_template) if isinstance(actual_template, str) else actual_template
    require(actual_template == template, "Actual lineage template changed")
    stack = single(observed["stack"]["Stacks"], "Ambiguous lineage stack")
    require(stack["StackStatus"] == "CREATE_COMPLETE" and stack["StackName"] == prefix
            and re.fullmatch(r"arn:aws:cloudformation:" + re.escape(region) + ":" + account
                             + ":stack/" + re.escape(prefix) + r"/[A-Za-z0-9-]+", stack["StackId"])
            and len(stack["Parameters"]) == len(parameters)
            and {row["ParameterKey"]: row["ParameterValue"] for row in stack["Parameters"]} == parameters,
            "Actual lineage stack or parameter binding differs")
    request = packet["request"]
    require(request["service_name"] == "cloudformation" and request["operation_name"] == "CreateStack"
            and request["region_name"] == region
            and json.loads(request["params"]["TemplateBody"]) == template
            and request["params"]["StackName"] == prefix
            and request["params"]["Parameters"] == [
                {"ParameterKey": name, "ParameterValue": value} for name, value in parameters.items()]
            and request["params"]["Capabilities"] == ["CAPABILITY_IAM"]
            and request["params"]["OnFailure"] == "ROLLBACK" and request["params"]["TimeoutInMinutes"] == 15
            and observed["launch"]["request"] == request["params"]
            and observed["launch"]["response"]["StackId"] == stack["StackId"],
            "Actual CreateStack call differs from the bounded request")
    rows = observed["resources"]["StackResources"]
    resources = {row["LogicalResourceId"]: row for row in rows}
    require(len(rows) == len(resources) and set(resources) == set(template["Resources"]),
            "Lineage resource inventory is incomplete or duplicated")
    for name, row in resources.items():
        require(row["StackId"] == stack["StackId"] and row["ResourceStatus"] == "CREATE_COMPLETE"
                and row["ResourceType"] == template["Resources"][name]["Type"],
                "Lineage resource belongs to another stack or is incomplete")
    ids = {name: row["PhysicalResourceId"] for name, row in resources.items()}
    reservations = observed["instances"]["Reservations"]
    instances = [row for reservation in reservations for row in reservation["Instances"]]
    instance_ids = {slot: ids[label + "Worker"] for slot, label in LABELS.items()}
    require(len(instances) == len(set(instance_ids.values())) == 3
            and {row["InstanceId"] for row in instances} == set(instance_ids.values())
            and all(row["OwnerId"] == account for row in reservations), "Lineage instance inventory differs")
    description = single(observed["instance_types"]["InstanceTypes"], "Ambiguous lineage instance type")
    image = single(observed["images"]["Images"], "Ambiguous lineage image")
    require(description["InstanceType"] == instance_type and description["MemoryInfo"]["SizeInMiB"] == 8192
            and description["VCpuInfo"]["DefaultVCpus"] == 4
            and "x86_64" in description["ProcessorInfo"]["SupportedArchitectures"]
            and image["ImageId"] == parameters["ImageId"] and image["Architecture"] == "x86_64"
            and image["State"] == "available" and image["OwnerId"] == "099720109477",
            "Lineage hardware capacity or image provenance differs")
    if instance_type == "c7a.xlarge":
        require(description["VCpuInfo"].get("DefaultCores") == 4
                and description["VCpuInfo"].get("DefaultThreadsPerCore") == 1,
                "Explicit C7a lineage physical-core topology differs")
    volumes, interfaces = observed["volumes"]["Volumes"], observed["interfaces"]["NetworkInterfaces"]
    require(len(volumes) == len({row["VolumeId"] for row in volumes}) == 9
            and len(interfaces) == len({row["NetworkInterfaceId"] for row in interfaces}) == 3,
            "Lineage requires nine distinct disks and three interfaces")
    require(set(observed["roles"]) == {*(label + "Role" for label in LABELS.values()), "ExpiryRole"}
            and set(observed["profiles"]) == set(LABELS.values())
            and set(observed["shutdown"]) == set(observed["user_data"]) == set(instance_ids.values()),
            "Lineage control inventory is incomplete")
    network_readback(observed, ids)
    input_readback(packet, observed)
    variables = {**parameters, **ids, "AWS::Partition": "aws", "AWS::Region": region, "AWS::AccountId": account}
    used_volumes, used_interfaces, result = set(), set(), {}
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    for slot, label in LABELS.items():
        instance = single([row for row in instances if row["InstanceId"] == instance_ids[slot]],
                          "Lineage slot is missing")
        iid = instance["InstanceId"]
        require(re.fullmatch(r"i-[0-9a-f]{17}", iid) and instance["State"]["Name"] == "running"
                and instance["Architecture"] == "x86_64" and instance["ImageId"] == parameters["ImageId"]
                and instance["InstanceType"] == instance_type
                and instance["Placement"]["AvailabilityZone"] == parameters["AvailabilityZone"]
                and not instance["Placement"].get("GroupName")
                and instance["VpcId"] == ids["Vpc"] and instance["SubnetId"] == ids["Subnet"]
                and instance["MetadataOptions"]["HttpTokens"] == "required"
                and instance["MetadataOptions"]["HttpPutResponseHopLimit"] == 1
                and instance["MetadataOptions"]["State"] == "applied"
                and observed["shutdown"][iid]["InstanceInitiatedShutdownBehavior"]["Value"] == "terminate",
                "Lineage instance profile, placement or metadata policy differs")
        if instance_type == "c7a.xlarge":
            require(instance.get("CpuOptions") == {"CoreCount": 4, "ThreadsPerCore": 1},
                    "Actual C7a lineage CPU configuration differs")
        script = template["Resources"][label + "Worker"]["Properties"]["UserData"]["Fn::Base64"]
        require(base64.b64decode(observed["user_data"][iid]["UserData"]["Value"], validate=True).decode()
                == substitute(script, variables), "Actual lineage bootstrap changed")
        disk_ids = disks(instance, observed, parameters["AvailabilityZone"], used_volumes)
        interface(instance, slot, observed, ids, account, used_interfaces)
        role_spec = template["Resources"][label + "Role"]["Properties"]["Policies"][0]
        role_arn = checked_role(
            observed["roles"][label + "Role"], name=ids[label + "Role"], account=account,
            trust=trust, policy_name=role_spec["PolicyName"],
            policy=substitute(role_spec["PolicyDocument"], variables))
        profile = observed["profiles"][label]["InstanceProfile"]
        require(profile["InstanceProfileName"] == ids[label + "Profile"]
                and profile["Arn"] == "arn:aws:iam::" + account + ":instance-profile/" + ids[label + "Profile"]
                and profile["Arn"] == instance["IamInstanceProfile"]["Arn"]
                and [row["Arn"] for row in profile["Roles"]] == [role_arn],
                "Lineage slot has another instance profile")
        result[slot] = {
            "schema": "organism-native-lineage-permit-v1", "instance_id": iid,
            "slot": slot, "architecture": "x86_64", "prefix": prefix, "region": region,
            "deadline": packet["deadline"], "kit_sha256": parameters["KitSha256"],
            "catalog_sha256": parameters["CatalogSha256"], "cases_sha256": parameters["CasesSha256"],
            "bundle_sha256": parameters[label + "BundleSha256"], "volumes": disk_ids, "instances": instance_ids,
            "independent_expiry_configuration_verified": True, "schedule_arn": observed["schedule"]["Arn"],
            "approval_reference": packet["approval_reference"],
        }
    schedule, target = observed["schedule"], observed["schedule"]["Target"]
    require(schedule["Name"] == ids["Expiry"]
            and schedule["Arn"] == "arn:aws:scheduler:" + region + ":" + account + ":schedule/default/" + ids["Expiry"]
            and schedule["State"] == "ENABLED" and schedule["ScheduleExpressionTimezone"] == "UTC"
            and schedule["ScheduleExpression"] == "at(" + packet["deadline"] + ")"
            and schedule["FlexibleTimeWindow"] == {"Mode": "OFF"}
            and target["Arn"] == "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances"
            and json.loads(target["Input"]) == {"InstanceIds": list(instance_ids.values())}
            and target["RetryPolicy"] == {"MaximumEventAgeInSeconds": 300, "MaximumRetryAttempts": 2},
            "Lineage independent expiry is disabled, changed or targets different devices")
    expiry_spec = template["Resources"]["ExpiryRole"]["Properties"]
    policy = expiry_spec["Policies"][0]
    expiry_arn = checked_role(
        observed["roles"]["ExpiryRole"], name=ids["ExpiryRole"], account=account,
        trust=substitute(expiry_spec["AssumeRolePolicyDocument"], variables),
        policy_name=policy["PolicyName"], policy=substitute(policy["PolicyDocument"], variables))
    require(target["RoleArn"] == expiry_arn, "Lineage expiry uses an uninspected role")
    if packet.get("seed_expiry_at") is not None:
        stop = datetime.strptime(packet["seed_expiry_at"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        end = datetime.strptime(packet["deadline"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        require((end - stop).total_seconds() == 9000 and (stop - now).total_seconds() >= 4200,
                "Earlier seed expiry changed its fixed timing or leaves insufficient seed time")
        earlier = observed["seed_schedule"]
        require(earlier["Name"] == ids["SeedExpiry"]
                and earlier["Arn"] == "arn:aws:scheduler:" + region + ":" + account
                    + ":schedule/default/" + ids["SeedExpiry"]
                and earlier["GroupName"] == "default"
                and earlier["State"] == "ENABLED"
                and earlier["ScheduleExpressionTimezone"] == "UTC"
                and earlier["ScheduleExpression"] == "at(" + packet["seed_expiry_at"] + ")"
                and earlier["FlexibleTimeWindow"] == {"Mode": "OFF"}
                and earlier["Target"] == {
                    "Arn": "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances",
                    "RoleArn": expiry_arn,
                    "Input": json.dumps({"InstanceIds": [instance_ids["seed"]]}, separators=(",", ":")),
                    "RetryPolicy": {"MaximumEventAgeInSeconds": 300, "MaximumRetryAttempts": 2}},
                "Earlier seed expiry is disabled, changed, targets another device or uses another role")
    else:
        require("seed_schedule" not in observed, "Unexpected earlier seed schedule readback")
    return result
