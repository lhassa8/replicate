"""Generate reviewable CloudFormation; this module never calls AWS."""

from datetime import datetime, timezone
import json
import re


AMI = {"x86_64": "ami-0fa5967347d08d2df", "arm64": "ami-0e1ab5c876cc030e8"}
INSTANCE = {"x86_64": "c7i.xlarge", "arm64": "c7g.xlarge"}


def ref(name):
    return {"Ref": name}


def sub(value):
    return {"Fn::Sub": value}


def attributes(name, field):
    return {"Fn::GetAtt": [name, field]}


def validate_launch(prefix, expiry, *, now=None):
    if not re.fullmatch(r"native-[0-9]{8}-[a-z0-9]{4,12}", prefix):
        raise ValueError("Launch needs a distinct native experiment prefix")
    deadline = datetime.strptime(expiry, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    seconds = (deadline - (now or datetime.now(timezone.utc))).total_seconds()
    if not 1800 <= seconds <= 4 * 3600:
        raise ValueError("Expiry must be 30 minutes to four hours ahead")
    return seconds


def storage_template():
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "Temporary private artifacts for the authorized organism native preflight",
        "Resources": {
            "Artifacts": {"Type": "AWS::S3::Bucket", "Properties": {
                "BucketEncryption": {"ServerSideEncryptionConfiguration": [
                    {"ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]},
                "PublicAccessBlockConfiguration": {
                    "BlockPublicAcls": True, "BlockPublicPolicy": True,
                    "IgnorePublicAcls": True, "RestrictPublicBuckets": True},
                "OwnershipControls": {"Rules": [{"ObjectOwnership": "BucketOwnerEnforced"}]},
                "LifecycleConfiguration": {"Rules": [
                    {"Id": "ExpireTemporaryArtifacts", "Status": "Enabled",
                     "ExpirationInDays": 7, "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}}]},
                "Tags": [{"Key": "Purpose", "Value": "organism-native-preflight"}],
            }},
            "TlsOnly": {"Type": "AWS::S3::BucketPolicy", "Properties": {
                "Bucket": ref("Artifacts"), "PolicyDocument": {"Version": "2012-10-17", "Statement": [{
                    "Effect": "Deny", "Principal": "*", "Action": "s3:*",
                    "Resource": [sub("arn:${AWS::Partition}:s3:::${Artifacts}"),
                                 sub("arn:${AWS::Partition}:s3:::${Artifacts}/*")],
                    "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                }]},
            }},
        },
        "Outputs": {"ArtifactBucket": {"Value": ref("Artifacts")}},
    }


def user_data(architecture, *, startup=False):
    # Only catalog/kit hashes and resource names enter this script. No secrets.
    # CloudFormation parameters constrain their characters before substitution.
    script = """#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/organism-bootstrap.log /dev/console) 2>&1
trap 'systemctl poweroff --no-block' EXIT

# Install the guest-side absolute expiry before any online preparation.
cat > /etc/systemd/system/organism-expiry.service <<'EOF'
[Unit]
Description=Terminate the temporary organism preflight worker
[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl poweroff --no-block
EOF
cat > /etc/systemd/system/organism-expiry.timer <<'EOF'
[Unit]
Description=Absolute expiry of the temporary organism preflight worker
[Timer]
OnCalendar=${DeadlineDate} ${DeadlineTime} UTC
Persistent=true
AccuracySec=1s
Unit=organism-expiry.service
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now organism-expiry.timer
export DEBIAN_FRONTEND=noninteractive
timeout 600 apt-get update -q
timeout 600 apt-get install -y --no-install-recommends python3-boto3 libgomp1 libstdc++6 libgcc-s1 libssl3t64 libcrypt1 binutils
useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin organism
mkdir -p /opt/organism-kit
python3 - <<'PY'
import boto3, hashlib, pathlib, tarfile
s3 = boto3.client("s3", region_name="us-east-1")
response = s3.get_object(Bucket="${ArtifactBucket}", Key="${RunPrefix}/inputs/kit.tar.gz")
try:
    if response["ContentLength"] > 2 * 1024**2:
        raise ValueError("Source kit exceeds the bootstrap bound")
    body = response["Body"].read(2 * 1024**2 + 1)
finally:
    response["Body"].close()
if hashlib.sha256(body).hexdigest() != "${KitSha256}":
    raise ValueError("Source kit hash differs from approved launch")
path = pathlib.Path("/opt/organism-kit.tar.gz")
path.write_bytes(body)
with tarfile.open(path, "r:gz") as archive:
    members = archive.getmembers()
    if len(members) > 100 or sum(m.size for m in members) > 8 * 1024**2:
        raise ValueError("Source kit extraction exceeds bound")
    archive.extractall("/opt/organism-kit", members=members, filter="data")
PY
cd /opt/organism-kit
python3 -m organism.native_preflight.host --bucket '${ArtifactBucket}' --prefix '${RunPrefix}' --catalog-sha256 '${CatalogSha256}' --architecture ARCHITECTURE
""".replace("ARCHITECTURE", architecture)
    if startup:
        script = script.replace("python3-boto3 libgomp1", "python3-boto3 gpgv iproute2 libgomp1")
        script = script.replace("python3 -m organism.native_preflight.host ",
                                "python3 -m organism.native_preflight.startup_host ")
    return script


def worker_template(*, startup=False):
    resources = {
        "Vpc": {"Type": "AWS::EC2::VPC", "Properties": {
            "CidrBlock": "10.198.0.0/24", "EnableDnsSupport": True, "EnableDnsHostnames": True}},
        "InternetGateway": {"Type": "AWS::EC2::InternetGateway"},
        "GatewayAttachment": {"Type": "AWS::EC2::VPCGatewayAttachment", "Properties": {
            "VpcId": ref("Vpc"), "InternetGatewayId": ref("InternetGateway")}},
        "Subnet": {"Type": "AWS::EC2::Subnet", "Properties": {
            "VpcId": ref("Vpc"), "CidrBlock": "10.198.0.0/26", "AvailabilityZone": ref("AvailabilityZone")}},
        "Routes": {"Type": "AWS::EC2::RouteTable", "Properties": {"VpcId": ref("Vpc")}},
        "InternetRoute": {"Type": "AWS::EC2::Route", "DependsOn": "GatewayAttachment", "Properties": {
            "RouteTableId": ref("Routes"), "DestinationCidrBlock": "0.0.0.0/0",
            "GatewayId": ref("InternetGateway")}},
        "RouteAssociation": {"Type": "AWS::EC2::SubnetRouteTableAssociation", "Properties": {
            "SubnetId": ref("Subnet"), "RouteTableId": ref("Routes")}},
        "SecurityGroup": {"Type": "AWS::EC2::SecurityGroup", "Properties": {
            "GroupDescription": "No inbound access; online artifact preparation only",
            "VpcId": ref("Vpc"), "SecurityGroupIngress": [],
            "SecurityGroupEgress": [{"IpProtocol": "tcp", "FromPort": port, "ToPort": port,
                                    "CidrIp": "0.0.0.0/0"} for port in (80, 443)]}},
        "Spread": {"Type": "AWS::EC2::PlacementGroup", "Properties": {"Strategy": "spread"}},
    }
    for architecture, label in (("x86_64", "X86"), ("arm64", "Arm")):
        role = label + "Role"
        resources[role] = {"Type": "AWS::IAM::Role", "Properties": {
            "AssumeRolePolicyDocument": {"Version": "2012-10-17", "Statement": [{
                "Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]},
            "Policies": [{"PolicyName": "NativeArtifactsOnly", "PolicyDocument": {
                "Version": "2012-10-17", "Statement": [
                    {"Effect": "Allow", "Action": "s3:GetObject",
                     "Resource": sub("arn:${AWS::Partition}:s3:::${ArtifactBucket}/${RunPrefix}/inputs/*")},
                    {"Effect": "Allow", "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
                     "Resource": sub("arn:${AWS::Partition}:s3:::${ArtifactBucket}/${RunPrefix}/outputs/" + architecture + "/*")},
                ]}}],
        }}
        resources[label + "Profile"] = {"Type": "AWS::IAM::InstanceProfile", "Properties": {"Roles": [ref(role)]}}
        resources[label + "Worker"] = {"Type": "AWS::EC2::Instance",
            "DependsOn": ["InternetRoute", "RouteAssociation"], "Properties": {
                "ImageId": AMI[architecture], "InstanceType": INSTANCE[architecture],
                "IamInstanceProfile": ref(label + "Profile"), "PlacementGroupName": ref("Spread"),
                "InstanceInitiatedShutdownBehavior": "terminate",
                "PropagateTagsToVolumeOnCreation": True,
                "MetadataOptions": {"HttpTokens": "required", "HttpPutResponseHopLimit": 1},
                "NetworkInterfaces": [{"DeviceIndex": "0", "AssociatePublicIpAddress": True, "DeleteOnTermination": True,
                                       "SubnetId": ref("Subnet"), "GroupSet": [ref("SecurityGroup")]}],
                "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {
                    "VolumeSize": 48, "VolumeType": "gp3", "Encrypted": True, "DeleteOnTermination": True}}],
                "UserData": {"Fn::Base64": sub(user_data(architecture, startup=startup))},
                "Tags": [{"Key": "Purpose", "Value": "organism-native-preflight"},
                         {"Key": "Experiment", "Value": ref("RunPrefix")},
                         {"Key": "Architecture", "Value": architecture}],
            }}
    resources["ExpiryRole"] = {"Type": "AWS::IAM::Role", "Properties": {
        "AssumeRolePolicyDocument": {"Version": "2012-10-17", "Statement": [{
            "Effect": "Allow", "Principal": {"Service": "scheduler.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {"StringEquals": {"aws:SourceAccount": ref("AWS::AccountId")}},
        }]},
        "Policies": [{"PolicyName": "TerminateOnlyThisPair", "PolicyDocument": {
            "Version": "2012-10-17", "Statement": [{
                "Effect": "Allow", "Action": "ec2:TerminateInstances",
                "Resource": [sub("arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:instance/${X86Worker}"),
                             sub("arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:instance/${ArmWorker}")],
            }]}}],
    }}
    resources["Expiry"] = {"Type": "AWS::Scheduler::Schedule", "Properties": {
        "Description": "Independent AWS termination of the bounded native preflight pair",
        "State": "ENABLED", "ScheduleExpression": sub("at(${DeadlineDate}T${DeadlineTime})"),
        "ScheduleExpressionTimezone": "UTC", "FlexibleTimeWindow": {"Mode": "OFF"},
        "Target": {
            "Arn": sub("arn:${AWS::Partition}:scheduler:::aws-sdk:ec2:terminateInstances"),
            "RoleArn": attributes("ExpiryRole", "Arn"),
            "Input": sub('{"InstanceIds":["${X86Worker}","${ArmWorker}"]}'),
            "RetryPolicy": {"MaximumEventAgeInSeconds": 300, "MaximumRetryAttempts": 2},
        },
    }}
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "Bounded two-architecture CPU compatibility experiment; no training or replication",
        "Parameters": {
            "ArtifactBucket": {"Type": "String", "AllowedPattern": "[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]"},
            "RunPrefix": {"Type": "String", "AllowedPattern": "native-[0-9]{8}-[a-z0-9]{4,12}"},
            "CatalogSha256": {"Type": "String", "AllowedPattern": "[0-9a-f]{64}"},
            "KitSha256": {"Type": "String", "AllowedPattern": "[0-9a-f]{64}"},
            "DeadlineDate": {"Type": "String", "AllowedPattern": "[0-9]{4}-[0-9]{2}-[0-9]{2}"},
            "DeadlineTime": {"Type": "String", "AllowedPattern": "[0-9]{2}:[0-9]{2}:[0-9]{2}"},
            "AvailabilityZone": {"Type": "AWS::EC2::AvailabilityZone::Name"},
        },
        "Resources": resources,
        "Outputs": {**{label + "InstanceId": {"Value": ref(label + "Worker")} for label in ("X86", "Arm")},
                    "ExpirySchedule": {"Value": ref("Expiry")},
                    "PlacementGroup": {"Value": ref("Spread")}},
    }


if __name__ == "__main__":
    print(json.dumps(worker_template(), indent=2))
