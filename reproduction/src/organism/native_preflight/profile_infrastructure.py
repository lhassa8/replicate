"""Separate three-disk native seed template; no AWS calls or provisioning."""

from .infrastructure import INSTANCE, ref, user_data, worker_template as legacy_template
from .common import ARCHES


def selected_architectures(architectures):
    if not isinstance(architectures, (list, tuple)) or tuple(architectures) not in (ARCHES, ("x86_64",)):
        raise ValueError("Native profile supports the original pair or one x86-64 device")
    return tuple(architectures)


def selected_x86_instance(value, *, first_lineage):
    if value is None:
        return INSTANCE["x86_64"]
    if not first_lineage or value not in ("c7i.xlarge", "c7a.xlarge"):
        raise ValueError("Only the first-lineage 8 GiB four-vCPU x86 choices are supported")
    return value


def worker_template(*, architectures=ARCHES, first_lineage=False, x86_instance_type=None):
    architectures = selected_architectures(architectures)
    if type(first_lineage) is not bool or (first_lineage and architectures != ("x86_64",)):
        raise ValueError("First-lineage native observation requires one x86-64 device")
    instance_type = selected_x86_instance(x86_instance_type, first_lineage=first_lineage)
    template = legacy_template(startup=True)
    template["Resources"]["X86Worker"]["Properties"]["InstanceType"] = instance_type
    if first_lineage:
        from .first_lineage_profile import CASES_SHA256
        template["Parameters"]["CasesSha256"] = {
            "Type": "String", "AllowedValues": [CASES_SHA256]}
    template["Description"] = "Bounded trained native seed pair with shared owner device controls"
    for architecture, label in (("x86_64", "X86"), ("arm64", "Arm")):
        properties = template["Resources"][label + "Worker"]["Properties"]
        parameter = label + "ImageId"
        template["Parameters"][parameter] = {"Type": "AWS::EC2::Image::Id"}
        properties["ImageId"] = ref(parameter)
        properties["BlockDeviceMappings"] = [
            {"DeviceName": name, "Ebs": {"VolumeSize": size, "VolumeType": "gp3",
                                        "Encrypted": True, "DeleteOnTermination": True}}
            for name, size in (("/dev/sda1", 8), ("/dev/sdf", 22), ("/dev/sdg", 2))]
        script = user_data(architecture, startup=True)
        script = script.replace("python3-boto3 gpgv iproute2",
                                "python3-boto3 python3-cryptography gpgv iproute2 ethtool e2fsprogs util-linux")
        script = script.replace("region_name=\"us-east-1\"", "region_name=\"${AWS::Region}\"")
        old = (
            "python3 -m organism.native_preflight.startup_host --bucket '${ArtifactBucket}' "
            "--prefix '${RunPrefix}' --catalog-sha256 '${CatalogSha256}' --architecture " + architecture)
        new = (
            "python3 -m organism.native_preflight.profile_host --bucket '${ArtifactBucket}' "
            "--prefix '${RunPrefix}' --region '${AWS::Region}' "
            "--kit-sha256 '${KitSha256}' --catalog-sha256 '${CatalogSha256}' "
            "--deadline '${DeadlineDate}T${DeadlineTime}' --architecture " + architecture)
        if first_lineage:
            new += " --cases-sha256 '${CasesSha256}'"
        if old not in script:
            raise ValueError("Native bootstrap command changed; review the new integration explicitly")
        properties["UserData"]["Fn::Base64"]["Fn::Sub"] = script.replace(old, new)
    if architectures == ("x86_64",):
        template["Description"] = "Bounded trained native seed on one x86-64 owner-controlled device"
        for name in ("ArmWorker", "ArmProfile", "ArmRole"):
            del template["Resources"][name]
        del template["Parameters"]["ArmImageId"]
        del template["Outputs"]["ArmInstanceId"]
        policies = template["Resources"]["ExpiryRole"]["Properties"]["Policies"]
        policies[0]["PolicyName"] = "TerminateOnlyThisDevice"
        policies[0]["PolicyDocument"]["Statement"][0]["Resource"] = [
            {"Fn::Sub": "arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:instance/${X86Worker}"}
        ]
        expiry = template["Resources"]["Expiry"]["Properties"]
        expiry["Description"] = "Independent AWS termination of the bounded native x86-64 device"
        expiry["Target"]["Input"] = {"Fn::Sub": '{"InstanceIds":["${X86Worker}"]}'}
    return template
