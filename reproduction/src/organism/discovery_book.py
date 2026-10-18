"""Offline owner-signed endpoints for the finite enrolled lab."""

import base64
import hashlib
import json
import re

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .capsule import canonical
from .enrollment import exact, key_id, unique_object
from .peer_identity import LabNetwork


SCHEMA = "organism-discovery-book-v1"
EXPERIENCE_SCHEMA = "organism-discovery-book-v2"
MAX_BOOK_BYTES = 16384


class DiscoveryBookError(ValueError):
    pass


def validate(body, policy):
    version_two = isinstance(body, dict) and body.get("schema") == EXPERIENCE_SCHEMA
    fields = ("schema", "policy_id", "epoch", "lab_cidrs", "endpoints")
    exact(body, fields + (("experience_checkers",) if version_two else ()))
    if body["schema"] not in (SCHEMA, EXPERIENCE_SCHEMA) or body["policy_id"] != policy.policy_id:
        raise DiscoveryBookError("Discovery book belongs to another enrollment")
    if version_two:
        checkers = body["experience_checkers"]
        if (not isinstance(checkers, list) or not 1 <= len(checkers) <= 4
                or any(not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value) for value in checkers)
                or checkers != sorted(set(checkers))):
            raise DiscoveryBookError("Explicit distinct experience checker identities are required")
    if type(body["epoch"]) is not int or not 1 <= body["epoch"] <= 2**31 - 1:
        raise DiscoveryBookError("Invalid discovery epoch")
    if not isinstance(body["lab_cidrs"], list) or not all(isinstance(value, str) for value in body["lab_cidrs"]):
        raise DiscoveryBookError("Explicit discovery subnets are required")
    network = LabNetwork(body["lab_cidrs"])
    if not isinstance(body["endpoints"], dict) or set(body["endpoints"]) != set(policy.body["devices"]):
        raise DiscoveryBookError("Discovery endpoints must match the enrolled device roster exactly")
    locations = set()
    for identity, endpoint in body["endpoints"].items():
        exact(endpoint, ("address", "port") + (("experience_port",) if version_two else ()))
        policy.peer_key(identity)
        address = network.address(endpoint["address"])
        if type(endpoint["port"]) is not int or not 1 <= endpoint["port"] <= 65535:
            raise DiscoveryBookError("Invalid discovery port")
        ports = [endpoint["port"]]
        if version_two:
            port = endpoint["experience_port"]
            if type(port) is not int or not 1024 <= port <= 65535:
                raise DiscoveryBookError("Experience requires an explicitly declared unprivileged port")
            ports.append(port)
        for port in ports:
            location = (address, port)
            if location in locations:
                raise DiscoveryBookError("Enrolled services cannot declare the same endpoint")
            locations.add(location)


def sign_discovery_book(body, owner_key, policy):
    validate(body, policy)
    public = owner_key.public_key().public_bytes_raw()
    if key_id(public) != policy.owner_id:
        raise DiscoveryBookError("Discovery signer differs from the enrollment owner")
    return canonical({"body": body, "owner_id": policy.owner_id,
                      "signature": base64.b64encode(owner_key.sign(canonical(body))).decode()})


class DiscoveryBook:
    def __init__(self, signed, policy, owner_public):
        if not isinstance(signed, bytes) or not 0 < len(signed) <= MAX_BOOK_BYTES:
            raise DiscoveryBookError("Discovery book exceeds its bound")
        try:
            envelope = json.loads(signed, object_pairs_hook=unique_object)
            exact(envelope, ("body", "owner_id", "signature"))
            if envelope["owner_id"] != policy.owner_id or key_id(owner_public) != policy.owner_id:
                raise DiscoveryBookError("Discovery owner differs from trusted enrollment")
            Ed25519PublicKey.from_public_bytes(owner_public).verify(
                base64.b64decode(envelope["signature"], validate=True), canonical(envelope["body"]))
            validate(envelope["body"], policy)
        except (InvalidSignature, KeyError, TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise DiscoveryBookError("Invalid signed discovery book") from exc
        self.signed = canonical(envelope)
        self.book_id = hashlib.sha256(self.signed).hexdigest()
        self.policy = policy
        self._body = canonical(envelope["body"])

    @property
    def body(self):
        return json.loads(self._body)

    def endpoint(self, identity):
        value = self.body["endpoints"][identity]
        return identity, value["address"], value["port"]

    @property
    def experience_checkers(self):
        return frozenset(self.body.get("experience_checkers", ()))

    def experience_endpoint(self, identity):
        if not self.experience_checkers:
            raise DiscoveryBookError("This directory does not enable experience exchange")
        value = self.body["endpoints"][identity]
        return identity, value["address"], value["experience_port"]
