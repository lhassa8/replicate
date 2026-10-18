"""Standard-library layout and catalog checks for the full-capsule seed test."""

import base64
import hashlib
import json
from pathlib import Path
import re


SCHEMA = "organism-native-startup-v1"
ROOT = Path("/opt/organism-startup")
BOOT = Path("/opt/organism-bootstrap")
BASE = Path("/var/lib/organism-startup")
OUTPUT = Path("/var/lib/organism-native/output")
UNIT = "organism-native-startup"
NAMESPACE = "organism-native-startup"
NS_PATH = "/run/netns/" + NAMESPACE
MAX_CATALOG = 1024**2


def relative_name(name):
    if (not isinstance(name, str) or len(name) > 384
            or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", part)
                   or part in (".", "..") for part in name.split("/"))):
        raise ValueError("Unsafe capsule catalog path")
    return name


def validate_catalog(value):
    return validate_payload_catalog(value, expected_schema=SCHEMA, expected_trained=False)


def validate_payload_catalog(value, *, expected_schema, expected_trained):
    """Shared inventory checks; each entry point fixes its own model kind and schema."""
    if (value.get("schema") != expected_schema or value.get("trained_for_organism") is not expected_trained
            or value.get("qualification_passed") is not False):
        raise ValueError("Unexpected full-capsule startup catalog kind")
    raw = base64.b64decode(value["manifest_base64"], validate=True)
    if not 0 < len(raw) <= MAX_CATALOG:
        raise ValueError("Startup manifest exceeds its bound")
    envelope = json.loads(raw)
    body = envelope["body"]
    if (envelope["release_id"] != value["release_id"]
            or body["model"]["trained_for_organism"] is not expected_trained
            or body["platforms"] != value["platforms"]):
        raise ValueError("Catalog and capsule release identities differ")
    if set(body["platforms"]) != {"linux-x86_64", "linux-arm64"} or len(body["files"]) != 11:
        raise ValueError("Startup test requires both complete capsule platforms")
    references = {body["model"]["path"], *(body[role] for role in ("agent", "knowledge", "self_test", "notices"))}
    for entry in body["platforms"].values():
        if set(entry) != {"python", "inference", "dependencies"}:
            raise ValueError("Startup capsule platform is incomplete")
        references.update(entry.values())
    if references != set(body["files"]) or value["files"] != body["files"]:
        raise ValueError("Startup payload inventory differs from the manifest")
    total = 0
    for name, entry in value["files"].items():
        relative_name(name)
        if (set(entry) != {"bytes", "sha256"} or type(entry["bytes"]) is not int
                or not 0 < entry["bytes"] <= 3 * 1024**3
                or not re.fullmatch("[0-9a-f]{64}", entry["sha256"])):
            raise ValueError("Invalid startup payload declaration")
        total += entry["bytes"]
    if total > 3 * 1024**3:
        raise ValueError("Startup capsule exceeds the native preflight allowance")
    # Host preparation uses the approved catalog hash. The portable receiver
    # additionally verifies the actual Ed25519 signature before any admission.
    if len(value["release_keys"]) != 1 or envelope["signer_id"] not in value["release_keys"]:
        raise ValueError("Expected the single approved development release key")
    for name, text in value["release_keys"].items():
        public = base64.b64decode(text, validate=True)
        if len(public) != 32 or hashlib.sha256(public).hexdigest() != name:
            raise ValueError("Startup release key identity differs")
    return value
