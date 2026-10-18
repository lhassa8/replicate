"""Prepare three fresh, separately fenced native hosts; no AWS calls."""

from copy import deepcopy
from datetime import datetime
import json

from . import lineage_network as network
from .infrastructure import attributes, ref, sub
from .profile_infrastructure import worker_template as single_template


LABELS = {"seed": "Seed", "child": "Child", "grandchild": "Grandchild"}


def worker_template(*, x86_instance_type=None, seed_expiry_at=None):
    template = single_template(architectures=("x86_64",), first_lineage=True,
                               x86_instance_type=x86_instance_type)
    template["Description"] = "Three fresh native lineage devices with separate owner admission"
    resources = template["Resources"]
    role = resources.pop("X86Role")
    profile = resources.pop("X86Profile")
    worker = resources.pop("X86Worker")
    resources.pop("Spread")
    template["Outputs"].pop("PlacementGroup")
    template["Outputs"].pop("X86InstanceId")
    template["Parameters"]["ImageId"] = template["Parameters"].pop("X86ImageId")
    group = resources["SecurityGroup"]["Properties"]
    group["GroupDescription"] = "Owner preparation and private three-device VXLAN"
    group["SecurityGroupIngress"] = [
        {"IpProtocol": "udp", "FromPort": network.PORT, "ToPort": network.PORT,
         "CidrIp": address + "/32"} for address in network.UNDERLAY.values()]
    group["SecurityGroupEgress"] += deepcopy(group["SecurityGroupIngress"])
    base = "arn:${AWS::Partition}:s3:::${ArtifactBucket}/${RunPrefix}"
    for slot, label in LABELS.items():
        template["Parameters"][label + "BundleSha256"] = {
            "Type": "String", "AllowedPattern": "[0-9a-f]{64}"}
        own_role = deepcopy(role)
        own_role["Properties"]["Policies"] = [{
            "PolicyName": "LineageOwnerInputsOnly",
            "PolicyDocument": {"Version": "2012-10-17", "Statement": [
                {"Effect": "Allow", "Action": "s3:GetObject",
                 "Resource": [sub(base + suffix) for suffix in (
                     "/inputs/kit.tar.gz", "/inputs/catalog.json",
                     "/inputs/temperature-cases.frozen.json",
                     "/inputs/receiver/*", "/inputs/nodes/" + slot + ".tar.gz",
                     "/inputs/permits/" + slot + ".json",
                     "/inputs/controls/" + slot + "/*",
                     *(("/inputs/capsule/*",) if slot == "seed" else ()))]},
                {"Effect": "Allow", "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
                 "Resource": sub(base + "/outputs/" + slot + "/*")},
            ]},
        }]
        resources[label + "Role"] = own_role
        own_profile = deepcopy(profile)
        own_profile["Properties"]["Roles"] = [ref(label + "Role")]
        resources[label + "Profile"] = own_profile
        own_worker = deepcopy(worker)
        properties = own_worker["Properties"]
        properties["ImageId"] = ref("ImageId")
        properties["IamInstanceProfile"] = ref(label + "Profile")
        properties.pop("PlacementGroupName")
        properties["NetworkInterfaces"][0]["PrivateIpAddress"] = network.UNDERLAY[slot]
        script = properties["UserData"]["Fn::Base64"]["Fn::Sub"]
        old = "python3 -m organism.native_preflight.profile_host "
        if script.count(old) != 1:
            raise ValueError("Reviewed single-device bootstrap entry point changed")
        script = script.replace(old, "python3 -m organism.native_preflight.lineage_host ")
        script = script.replace("--architecture x86_64", "--architecture x86_64 --slot " + slot
                                + " --bundle-sha256 '${" + label + "BundleSha256}'")
        properties["UserData"]["Fn::Base64"]["Fn::Sub"] = script
        properties["Tags"].append({"Key": "LineageSlot", "Value": slot})
        resources[label + "Worker"] = own_worker
        template["Outputs"][label + "InstanceId"] = {"Value": ref(label + "Worker")}
    policy = resources["ExpiryRole"]["Properties"]["Policies"][0]
    policy["PolicyName"] = "TerminateOnlyTheseThreeDevices"
    policy["PolicyDocument"]["Statement"][0]["Resource"] = [
        sub("arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:instance/${" + label + "Worker}")
        for label in LABELS.values()]
    expiry = resources["Expiry"]["Properties"]
    expiry["Description"] = "Independent fixed expiry for the three lineage devices"
    expiry["Target"]["Input"] = sub(json.dumps({
        "InstanceIds": ["${" + label + "Worker}" for label in LABELS.values()]}, separators=(",", ":")))
    expiry["Target"]["RoleArn"] = attributes("ExpiryRole", "Arn")
    if seed_expiry_at is not None:
        # The optional earlier seed stop belongs to the original allocation.
        parsed = datetime.strptime(seed_expiry_at, "%Y-%m-%dT%H:%M:%S")
        if parsed.strftime("%Y-%m-%dT%H:%M:%S") != seed_expiry_at:
            raise ValueError("Seed expiry must be an exact UTC second")
        earlier = deepcopy(expiry)
        earlier["Description"] = "Earlier fixed seed termination for descendant endurance"
        earlier["ScheduleExpression"] = "at(" + seed_expiry_at + ")"
        earlier["Target"]["Input"] = sub('{"InstanceIds":["${SeedWorker}"]}')
        resources["SeedExpiry"] = {"Type": "AWS::Scheduler::Schedule", "Properties": earlier}
        template["Outputs"]["SeedExpirySchedule"] = {"Value": ref("SeedExpiry")}
        # Keep the original all-device schedule; stop this seed early locally too.
        seed_script = resources["SeedWorker"]["Properties"]["UserData"]["Fn::Base64"]["Fn::Sub"]
        timer = "OnCalendar=${DeadlineDate} ${DeadlineTime} UTC"
        if seed_script.count(timer) != 1:
            raise ValueError("Reviewed seed guest expiry timer changed")
        seed_script = seed_script.replace(timer, "OnCalendar=" + seed_expiry_at.replace("T", " ") + " UTC")
        resources["SeedWorker"]["Properties"]["UserData"]["Fn::Base64"]["Fn::Sub"] = seed_script
    return template
