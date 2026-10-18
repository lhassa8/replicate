"""Offline owner-signed lab membership, separate from reproductive payloads."""

import base64
import hashlib
import json
import re

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from .capsule import canonical


SCHEMA = "organism-enrollment-v1"
MAX_POLICY_BYTES = 32 * 1024
HEX = re.compile(r"[0-9a-f]{64}\Z")


class EnrollmentError(ValueError):
    pass


def key_id(public):
    return hashlib.sha256(public).hexdigest()


def public_bytes(key):
    return key.public_bytes(Encoding.Raw, PublicFormat.Raw)


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise EnrollmentError("Duplicate enrollment field")
        result[name] = value
    return result


def exact(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise EnrollmentError("Unexpected enrollment fields")


def decode_keys(values):
    if not isinstance(values, dict) or not values:
        raise EnrollmentError("Missing enrolled keys")
    decoded = {}
    for identity, encoded in values.items():
        if not isinstance(identity, str) or not HEX.fullmatch(identity):
            raise EnrollmentError("Invalid enrolled identity")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) != 32 or key_id(raw) != identity:
            raise EnrollmentError("Enrolled key does not match its identity")
        Ed25519PublicKey.from_public_bytes(raw)
        decoded[identity] = raw
    return decoded


def validate(body):
    exact(body, {"schema", "experiment_id", "epoch", "devices", "release_keys",
                 "require_trained", "max_payload_bytes", "birth_seconds",
                 "max_births_per_device", "max_process_starts"})
    if body["schema"] != SCHEMA:
        raise EnrollmentError("Unsupported enrollment policy")
    if not isinstance(body["experiment_id"], str) or not 1 <= len(body["experiment_id"]) <= 128:
        raise EnrollmentError("Invalid experiment identity")
    ranges = {
        "epoch": (1, 2**31 - 1), "max_payload_bytes": (1, 8 * 1024**3),
        "birth_seconds": (1, 1200), "max_births_per_device": (1, 64),
        "max_process_starts": (1, 8),
    }
    for name, (minimum, maximum) in ranges.items():
        if type(body[name]) is not int or not minimum <= body[name] <= maximum:
            raise EnrollmentError("Enrollment limit outside bounds: " + name)
    if type(body["require_trained"]) is not bool:
        raise EnrollmentError("Training policy must be explicit")
    if not isinstance(body["devices"], dict) or not 1 <= len(body["devices"]) <= 6:
        raise EnrollmentError("Lab enrollment is bounded to six device identities")
    devices = {}
    for identity, entry in body["devices"].items():
        exact(entry, {"public_key", "platform"})
        if entry["platform"] not in ("linux-x86_64", "linux-arm64"):
            raise EnrollmentError("Unsupported enrolled platform")
        devices[identity] = entry["public_key"]
    decode_keys(devices)
    if len(body["release_keys"]) > 4:
        raise EnrollmentError("Too many release trust keys")
    decode_keys(body["release_keys"])
    return body


def sign_enrollment(body, owner_key):
    validate(body)
    return canonical({
        "body": body, "signature": base64.b64encode(owner_key.sign(canonical(body))).decode(),
        "owner_id": key_id(public_bytes(owner_key.public_key())),
    })


class Enrollment:
    """Verified immutable policy. The owner key is supplied out of band."""

    def __init__(self, signed_policy, trusted_owner_public):
        if not isinstance(signed_policy, bytes) or not 0 < len(signed_policy) <= MAX_POLICY_BYTES:
            raise EnrollmentError("Enrollment exceeds bounds")
        try:
            envelope = json.loads(signed_policy, object_pairs_hook=unique_object)
            exact(envelope, {"body", "signature", "owner_id"})
            if envelope["owner_id"] != key_id(trusted_owner_public):
                raise EnrollmentError("Enrollment owner is not trusted")
            Ed25519PublicKey.from_public_bytes(trusted_owner_public).verify(
                base64.b64decode(envelope["signature"], validate=True), canonical(envelope["body"]),
            )
            validate(envelope["body"])
        except EnrollmentError:
            raise
        except (InvalidSignature, TypeError, KeyError, ValueError, RecursionError, UnicodeError) as exc:
            raise EnrollmentError("Invalid signed enrollment") from exc
        # Store canonical bytes rather than an exposed mutable policy dict.
        self._body = canonical(envelope["body"])
        self.signed = canonical(envelope)
        self.policy_id = hashlib.sha256(self.signed).hexdigest()
        self.owner_id = envelope["owner_id"]

    @property
    def body(self):
        return json.loads(self._body)

    @property
    def release_keys(self):
        return decode_keys(self.body["release_keys"])

    def platform(self, identity):
        try:
            return self.body["devices"][identity]["platform"]
        except KeyError as exc:
            raise EnrollmentError("Device is not enrolled") from exc

    def peer_key(self, identity):
        try:
            value = self.body["devices"][identity]["public_key"]
        except KeyError as exc:
            raise EnrollmentError("Peer is not enrolled") from exc
        return base64.b64decode(value, validate=True)
