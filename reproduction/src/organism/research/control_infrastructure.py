"""Render bounded comparison infrastructure using existing owner templates."""

import argparse
import json
from pathlib import Path

from organism.native_preflight.infrastructure import storage_template
from organism.native_preflight.profile_infrastructure import worker_template


BOOTSTRAP = r"""#!/bin/bash
set -euo pipefail
umask 077
trap 'shutdown -h now' ERR
cat >/etc/systemd/system/replicate-paper-expiry.service <<'UNIT'
[Unit]
Description=Fixed local paper-comparison expiry
[Service]
Type=oneshot
ExecStart=/sbin/shutdown -h now
UNIT
cat >/etc/systemd/system/replicate-paper-expiry.timer <<'UNIT'
[Unit]
Description=Fixed local paper-comparison deadline
[Timer]
OnCalendar=${DeadlineDate} ${DeadlineTime} UTC
Persistent=true
[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now replicate-paper-expiry.timer
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-boto3
mkdir -p /root/replicate-paper-bootstrap
cd /root/replicate-paper-bootstrap
python3 - <<'PY'
import boto3, hashlib, json, pathlib, time, urllib.request
from botocore.exceptions import ClientError
s3 = boto3.client("s3", region_name="${AWS::Region}")
token_request = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
    headers={"X-aws-ec2-metadata-token-ttl-seconds":"60"})
token = urllib.request.urlopen(token_request, timeout=3).read().decode()
instance_request = urllib.request.Request("http://169.254.169.254/latest/meta-data/instance-id",
    headers={"X-aws-ec2-metadata-token":token})
instance_id = urllib.request.urlopen(instance_request, timeout=3).read().decode()
deadline = time.time() + 900
while time.time() < deadline:
    try:
        permit = json.loads(s3.get_object(Bucket="${ArtifactBucket}", Key="${RunPrefix}/inputs/permit.json")["Body"].read())
        assert permit["assets_sha256"] == "${CatalogSha256}"
        assert permit["deadline"] == "${DeadlineDate}T${DeadlineTime}+00:00"
        assert permit["instance_id"] == instance_id
        break
    except ClientError as exc:
        # A role without ListBucket receives AccessDenied for an absent object.
        # This closed wait never grants execution; only the bound permit does.
        if exc.response.get("Error", {}).get("Code") not in ("NoSuchKey", "AccessDenied", "404", "403"):
            raise
        time.sleep(5)
else:
    raise RuntimeError("Owner execution permit did not arrive")
raw = s3.get_object(Bucket="${ArtifactBucket}", Key="${RunPrefix}/inputs/host.py")["Body"].read()
assert hashlib.sha256(raw).hexdigest() == "${KitSha256}"
pathlib.Path("host.py").write_bytes(raw)
PY
python3 host.py --bucket '${ArtifactBucket}' --prefix '${RunPrefix}' \
  --region '${AWS::Region}' --assets-sha256 '${CatalogSha256}' \
  --deadline '${DeadlineDate}T${DeadlineTime}+00:00'
"""


def render(output):
    output.mkdir(parents=True, exist_ok=True)
    template = worker_template(architectures=("x86_64",), first_lineage=True,
                               x86_instance_type="c7a.xlarge")
    template["Description"] = "Bounded cohosted native comparison; three policies, zero hosted inference"
    del template["Parameters"]["CasesSha256"]
    worker = template["Resources"]["X86Worker"]["Properties"]
    worker["InstanceType"] = "c7a.2xlarge"
    worker["BlockDeviceMappings"] = [{"DeviceName": "/dev/sda1", "Ebs": {
        "VolumeSize": 96, "VolumeType": "gp3", "Encrypted": True, "DeleteOnTermination": True}}]
    worker["UserData"]["Fn::Base64"]["Fn::Sub"] = BOOTSTRAP
    for name, value in (("worker-template.json", template), ("storage-template.json", storage_template())):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.output)
