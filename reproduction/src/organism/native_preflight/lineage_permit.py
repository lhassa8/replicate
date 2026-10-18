"""Bound fresh owner admission to a slot, its private bundle and all three hosts."""

from datetime import datetime, timezone
import json
import re
import time

from ..training_inputs import require
from .first_lineage_profile import CASES_SHA256
from . import lineage_network as network
from .profile_permit import remaining


def validate(permit, args, instance):
    network.slot_value(args.slot)
    names = {
        "schema", "instance_id", "slot", "architecture", "prefix", "region", "deadline",
        "kit_sha256", "catalog_sha256", "cases_sha256", "bundle_sha256", "volumes",
        "instances", "independent_expiry_configuration_verified", "schedule_arn", "approval_reference",
    }
    require(isinstance(permit, dict) and set(permit) == names
            and permit["schema"] == "organism-native-lineage-permit-v1",
            "Unexpected lineage owner permit")
    require(permit["instance_id"] == instance and all(permit[name] == getattr(args, name) for name in (
        "slot", "architecture", "prefix", "region", "deadline", "kit_sha256",
        "catalog_sha256", "cases_sha256", "bundle_sha256")),
        "Lineage permit differs from this host, slot, inputs or fixed deadline")
    require(args.architecture == "x86_64" and args.cases_sha256 == CASES_SHA256
            and re.fullmatch(r"native-[0-9]{8}-[a-z0-9]{4,12}", args.prefix)
            and re.fullmatch(r"us-(east|west)-[12]", args.region),
            "Lineage permit changes the reduced execution scope")
    for name in ("kit_sha256", "catalog_sha256", "bundle_sha256"):
        require(isinstance(permit[name], str) and re.fullmatch("[0-9a-f]{64}", permit[name]),
                "Lineage input checksum is invalid")
    instances = permit["instances"]
    require(isinstance(instances, dict) and set(instances) == set(network.SLOTS)
            and len(set(instances.values())) == 3 and instances[args.slot] == instance
            and all(isinstance(value, str) and re.fullmatch("i-[0-9a-f]{17}", value)
                    for value in instances.values()), "Lineage instance slots are ambiguous")
    volumes = permit["volumes"]
    require(isinstance(volumes, dict) and set(volumes) == {"root", "data", "control"}
            and len(set(volumes.values())) == 3 and all(
                isinstance(value, str) and re.fullmatch("vol-[0-9a-f]{17}", value)
                for value in volumes.values()), "Lineage permit lacks its three exact disks")
    require(permit["independent_expiry_configuration_verified"] is True
            and isinstance(permit["schedule_arn"], str)
            and re.fullmatch(r"arn:aws:scheduler:" + re.escape(args.region)
                             + r":[0-9]{12}:schedule/default/[A-Za-z0-9_-]+", permit["schedule_arn"])
            and isinstance(permit["approval_reference"], str) and permit["approval_reference"].strip(),
            "Lineage permit lacks verified independent expiry or spending authority")
    require(120 < remaining(args.deadline) <= 4 * 3600,
            "Lineage permit arrived outside its fixed lease")
    return permit


def wait(client, args, instance, evidence):
    from botocore.exceptions import ClientError
    started = time.monotonic()
    evidence.update(started_at=datetime.now(timezone.utc).isoformat(), attempts=0,
                    unavailable_responses=0, permit_verified=False)
    while time.monotonic() - started < 600 and remaining(args.deadline) > 120:
        evidence["attempts"] += 1
        try:
            response = client.get_object(
                Bucket=args.bucket, Key=args.prefix + "/inputs/permits/" + args.slot + ".json")
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code not in ("NoSuchKey", "404", "AccessDenied", "403"):
                raise
            evidence["unavailable_responses"] += 1
            evidence["last_unavailable_code"] = code
            time.sleep(2)
            continue
        try:
            require(0 < response["ContentLength"] <= 16384, "Lineage permit exceeds its size bound")
            raw = response["Body"].read(16385)
        finally:
            response["Body"].close()
        require(len(raw) == response["ContentLength"] and time.monotonic() - started < 600,
                "Lineage permit was truncated or late")
        result = validate(json.loads(raw), args, instance)
        evidence.update(completed_at=datetime.now(timezone.utc).isoformat(), permit_verified=True)
        return result
    raise TimeoutError("Lineage owner permit was not available within its admission window")
