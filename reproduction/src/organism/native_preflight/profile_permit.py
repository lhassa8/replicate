"""Instance-bound owner admission before initializing native test disks."""

from datetime import datetime, timezone
import json
import re
import time
import urllib.request


def require(value, message):
    if not value:
        raise ValueError(message)


def instance_identity():
    root = "http://169.254.169.254/latest/"
    request = urllib.request.Request(root + "api/token", method="PUT",
                                    headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
    # Do not use proxy environment variables for link-local instance metadata.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=5) as reply:
        token = reply.read(4097).decode()
    require(0 < len(token) <= 4096, "Instance metadata token exceeds its bound")
    request = urllib.request.Request(root + "meta-data/instance-id",
                                    headers={"X-aws-ec2-metadata-token": token})
    with opener.open(request, timeout=5) as reply:
        value = reply.read(129).decode().strip()
    require(re.fullmatch(r"i-[0-9a-f]{17}", value), "Instance metadata identity is invalid")
    return value


def remaining(deadline):
    parsed = datetime.strptime(deadline, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return (parsed - datetime.now(timezone.utc)).total_seconds()


def validate(permit, args, instance):
    from .first_lineage_profile import validate_selection
    cases_sha256 = getattr(args, "cases_sha256", None)
    first_lineage = validate_selection(cases_sha256, args.architecture)
    expected = {"schema", "instance_id", "architecture", "prefix", "region", "deadline",
                "kit_sha256", "catalog_sha256", "volumes", "independent_expiry_configuration_verified",
                "schedule_arn", "approval_reference"}
    if first_lineage:
        expected.add("cases_sha256")
    require(isinstance(permit, dict) and set(permit) == expected
            and permit["schema"] == "organism-native-profile-permit-" + ("v2" if first_lineage else "v1"),
            "Unexpected native profile permit")
    require(not first_lineage or permit["cases_sha256"] == cases_sha256,
            "Native permit does not bind the exact frozen observer cases")
    require(permit["instance_id"] == instance and all(
        permit[name] == getattr(args, name)
        for name in ("architecture", "prefix", "region", "deadline", "kit_sha256", "catalog_sha256")),
        "Native permit does not bind this exact host, inputs and expiry")
    require(permit["independent_expiry_configuration_verified"] is True
            and isinstance(permit["schedule_arn"], str)
            and re.fullmatch(r"arn:aws:scheduler:us-(east|west)-[12]:[0-9]{12}:schedule/default/[A-Za-z0-9_-]+",
                             permit["schedule_arn"])
            and permit["schedule_arn"].split(":")[3] == args.region
            and isinstance(permit["approval_reference"], str) and permit["approval_reference"].strip(),
            "Native permit lacks the externally verified expiry and spending authority")
    volumes = permit["volumes"]
    require(isinstance(volumes, dict) and set(volumes) == {"root", "data", "control"}
            and all(isinstance(value, str) and re.fullmatch("vol-[0-9a-f]{17}", value) for value in volumes.values())
            and len(set(volumes.values())) == 3, "Native permit disk identities are invalid")
    require(120 < remaining(args.deadline) <= 4 * 3600,
            "Native permit arrived outside the bounded execution interval")
    return permit


def wait(client, args, instance, evidence):
    from botocore.exceptions import ClientError
    started = time.monotonic()
    evidence.update(started_at=datetime.now(timezone.utc).isoformat(), attempts=0,
                    unavailable_responses=0, last_unavailable_code=None, permit_verified=False)
    while time.monotonic() - started < 600 and remaining(args.deadline) > 120:
        evidence["attempts"] += 1
        try:
            response = client.get_object(Bucket=args.bucket,
                                         Key=args.prefix + "/inputs/permits/" + args.architecture + ".json")
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            # Without ListBucket, a not-yet-published permit can return 403.
            # A denied read never admits the host; persistent denial expires.
            if code not in ("NoSuchKey", "404", "AccessDenied", "403"):
                raise
            evidence["unavailable_responses"] += 1
            evidence["last_unavailable_code"] = code
            time.sleep(2)
            continue
        try:
            require(0 < response["ContentLength"] <= 16384, "Native permit exceeds its size bound")
            raw = response["Body"].read(16385)
        finally:
            response["Body"].close()
        require(len(raw) == response["ContentLength"] and time.monotonic() - started < 600,
                "Native permit is truncated or arrived after its admission window")
        result = validate(json.loads(raw), args, instance)
        evidence["completed_at"] = datetime.now(timezone.utc).isoformat()
        evidence["permit_verified"] = True
        return result
    raise TimeoutError("Native launch permit was not available before its fixed admission deadline")
