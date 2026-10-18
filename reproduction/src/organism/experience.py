"""Bounded, endorsed reader procedures; no observations or free-form instructions.

Signatures authenticate the producer and verifier's assertions. They are not
proof of physical measurements or uncompromised peers. Callers supply trusted
enrollment, release and checker identities independently of received records.
"""

import base64
from copy import deepcopy
import hashlib
import json
import re

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .capsule import canonical
from .enrollment import key_id, public_bytes


SCHEMA = "organism-reader-experience-v1"
MAX_RECORD_BYTES = 8192
MAX_LIFETIME_SECONDS = 7 * 24 * 3600
HASH = re.compile(r"[0-9a-f]{64}\Z")
TOKEN = re.compile(r"[0-9a-f]{32}\Z")
SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
PREFIX = b"organism-reader-experience-v1\0"


class ExperienceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ExperienceError(message)


def exact(value, fields):
    require(isinstance(value, dict) and set(value) == set(fields), "Unexpected experience fields")


def hash_value(value):
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def token(value):
    return isinstance(value, str) and TOKEN.fullmatch(value) is not None


def integer(value):
    return type(value) is int and 0 <= value <= 4000000000


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate experience property")
        result[key] = value
    return result


def selector(value):
    if not isinstance(value, str) or len(value) > 160:
        return False
    if value == "" or value.startswith("/"):
        return re.search(r"~(?![01])", value) is None
    match = re.fullmatch(r"(holding|input):(0|[1-9][0-9]{0,4})", value)
    return match is not None and int(match[2]) <= 65535


def validate_reader(reader):
    exact(reader, ("source_id", "value_selectors", "timestamp_selector", "validity_selector"))
    values = reader["value_selectors"]
    require(isinstance(reader["source_id"], str) and SOURCE_ID.fullmatch(reader["source_id"]),
            "Invalid experience source identity")
    require(isinstance(values, list) and 1 <= len(values) <= 4 and all(selector(value) for value in values)
            and len(set(values)) == len(values), "Invalid experience measurement selectors")
    require(selector(reader["timestamp_selector"])
            and (reader["validity_selector"] is None or selector(reader["validity_selector"])),
            "Invalid experience sample selectors")


def validate_body(body, policy):
    exact(body, ("schema", "policy_id", "release_id", "producer_device_id", "producer_individual_id",
                 "created_unix_seconds", "expires_unix_seconds", "capability_sha256",
                 "reader", "source", "verification"))
    require(body["schema"] == SCHEMA and body["policy_id"] == policy.policy_id,
            "Experience belongs to another schema or enrollment")
    for name in ("release_id", "producer_device_id", "capability_sha256"):
        require(hash_value(body[name]), "Invalid experience identity")
    require(token(body["producer_individual_id"]), "Invalid source individual identity")
    created, expires = body["created_unix_seconds"], body["expires_unix_seconds"]
    require(integer(created) and integer(expires) and 0 < expires - created <= MAX_LIFETIME_SECONDS,
            "Experience lifetime exceeds bounds")
    validate_reader(body["reader"])
    source = body["source"]
    exact(source, ("job_sha256", "observation_sha256", "validation_sha256", "kind"))
    require(all(hash_value(source[key]) for key in ("job_sha256", "observation_sha256", "validation_sha256"))
            and source["kind"] in ("live_model", "development_fixture"), "Invalid source evidence references")
    verification = body["verification"]
    exact(verification, ("device_id", "checker_sha256", "outcome"))
    require(hash_value(verification["device_id"]) and hash_value(verification["checker_sha256"])
            and verification["outcome"] == "passed", "Experience has no successful verification endorsement")
    require(verification["device_id"] != body["producer_device_id"], "Producer cannot endorse its own experience")
    try:
        policy.peer_key(body["producer_device_id"])
        policy.peer_key(verification["device_id"])
    except ValueError as exc:
        raise ExperienceError("Experience signer is not enrolled") from exc
    require(len(canonical(body)) <= MAX_RECORD_BYTES - 512, "Experience body exceeds its bound")


def capability_hash(published):
    exact(published, ("device_id", "family", "task", "sources"))
    # Equipment instance identity changes between source and destination.
    # All published task, source, field and encoding information stays bound.
    return hashlib.sha256(canonical({key: value for key, value in published.items() if key != "device_id"})).hexdigest()


def seal_record(body, producer_key, verifier_key, policy):
    """Sign a checked body. Signing alone does not perform semantic validation."""
    validate_body(body, policy)
    require(key_id(public_bytes(producer_key.public_key())) == body["producer_device_id"]
            and key_id(public_bytes(verifier_key.public_key())) == body["verification"]["device_id"],
            "Experience signing key differs from its declared role")
    payload = canonical(body)
    result = canonical({
        "body": body, "note_id": hashlib.sha256(payload).hexdigest(),
        "producer_signature": base64.b64encode(producer_key.sign(PREFIX + b"producer\0" + payload)).decode(),
        "verifier_signature": base64.b64encode(verifier_key.sign(PREFIX + b"verifier\0" + payload)).decode(),
    })
    require(len(result) <= MAX_RECORD_BYTES, "Signed experience exceeds its bound")
    return result


class VerifiedExperience:
    def __init__(self, raw, policy, *, release_id, allowed_checkers, now, require_live_source=True):
        require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_RECORD_BYTES, "Experience bytes exceed bounds")
        require(hash_value(release_id) and isinstance(allowed_checkers, (set, frozenset))
                and 1 <= len(allowed_checkers) <= 4 and all(hash_value(value) for value in allowed_checkers),
                "Trusted release and checker identities must be supplied independently")
        require(type(require_live_source) is bool and integer(now), "Invalid experience verification policy or clock")
        try:
            envelope = json.loads(raw, object_pairs_hook=unique_object)
            exact(envelope, ("body", "note_id", "producer_signature", "verifier_signature"))
            body = envelope["body"]
            validate_body(body, policy)
            payload = canonical(body)
            require(envelope["note_id"] == hashlib.sha256(payload).hexdigest(), "Experience identity differs from content")
            require(body["release_id"] == release_id, "Experience belongs to another release")
            require(body["verification"]["checker_sha256"] in allowed_checkers, "Untrusted experience checker")
            require(not require_live_source or body["source"]["kind"] == "live_model",
                    "Development fixture experience cannot stand in for a live source task")
            require(body["created_unix_seconds"] <= now < body["expires_unix_seconds"],
                    "Experience is expired or not yet valid")
            for role, identity in (("producer", body["producer_device_id"]),
                                   ("verifier", body["verification"]["device_id"])):
                signature = base64.b64decode(envelope[role + "_signature"], validate=True)
                Ed25519PublicKey.from_public_bytes(policy.peer_key(identity)).verify(
                    signature, PREFIX + role.encode() + b"\0" + payload)
        except ExperienceError:
            raise
        except (InvalidSignature, TypeError, KeyError, ValueError, RecursionError, UnicodeError) as exc:
            raise ExperienceError("Invalid signed experience") from exc
        self.raw, self._body = canonical(envelope), payload
        self.note_id = envelope["note_id"]

    @property
    def body(self):
        return json.loads(self._body)

    def applicable(self, published, *, now):
        body = self.body
        require(integer(now), "Invalid applicability clock")
        return (body["created_unix_seconds"] <= now < body["expires_unix_seconds"]
                and capability_hash(published) == body["capability_sha256"])

    def prepare_reader(self, session):
        """Reuse only a reader choice; obtain fresh evidence and no diagnosis."""
        now = session.directory.clock()
        require(type(now) in (int, float) and 0 <= now <= 4000000000, "Invalid current reader clock")
        capabilities = session.inspect_equipment(session.device_id)
        require(self.applicable(capabilities["descriptor"], now=int(now)),
                "Experience does not apply to current capabilities or time")
        arguments = {"device_id": session.device_id, **deepcopy(self.body["reader"])}
        configured = session.configure_reader(**arguments)
        return {"note_id": self.note_id, "capability_evidence_id": capabilities["evidence_id"],
                "configuration_evidence_id": configured["evidence_id"],
                "current_measurement_read": False, "diagnosis_supplied": False}
