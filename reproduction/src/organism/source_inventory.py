"""Frozen procedure inputs and a separate peer's pre-inference commitment.

These signatures bind supplied software records, not physical-host attestation.
No diagnostic oracle or model-facing tool is provided by this module.
"""

import base64
import json
from types import MappingProxyType

from .capsule import canonical
from .enrollment import key_id, public_bytes
from .experience import MAX_RECORD_BYTES, VerifiedExperience, exact, hash_value, integer, require, token
from .experience_proposal import decode, moment, sha, verify_signature
from .experience_store import MAX_RECORDS


SCHEMA = "organism-source-inventory-v1"
PREFIX = b"organism-source-inventory-v1\0"
RECEIPT_SCHEMA = "organism-source-inventory-commit-v1"
RECEIPT_PREFIX = b"organism-source-inventory-commit-v1\0"
MAX_INVENTORY_BYTES = 2 * 1024 * 1024
MAX_COMMIT_BYTES = 4096


def check_snapshot(raw, policy, *, release_id, allowed_checkers, require_live_source):
    envelope = decode(raw, MAX_INVENTORY_BYTES)
    exact(envelope, ("body", "signature"))
    body = envelope["body"]
    exact(body, ("schema", "policy_id", "release_id", "producer_device_id", "producer_individual_id",
                 "request_id", "store_id", "captured_unix_seconds", "head", "records",
                 "allowed_checkers", "require_live_source"))
    require(body["schema"] == SCHEMA and body["policy_id"] == policy.policy_id
            and body["release_id"] == release_id and hash_value(release_id)
            and hash_value(body["producer_device_id"])
            and all(token(body[field]) for field in ("producer_individual_id", "request_id", "store_id"))
            and integer(body["captured_unix_seconds"]), "Invalid procedure inventory context")
    require(type(require_live_source) is bool
            and body["require_live_source"] is require_live_source
            and body["allowed_checkers"] == sorted(allowed_checkers),
            "Inventory uses another source-admission policy")
    require(isinstance(body["records"], dict) and len(body["records"]) <= MAX_RECORDS
            and isinstance(body["head"], list) and len(body["head"]) == 3
            and body["head"][0] == len(body["records"])
            and all(type(value) is int and 0 <= value <= bound
                    for value, bound in zip(body["head"], (MAX_RECORDS, 256, 256))),
            "Inventory differs from its bounded store head")
    verify_signature(policy.peer_key(body["producer_device_id"]), envelope["signature"], PREFIX + canonical(body))
    require(canonical(envelope) == raw, "Inventory encoding is not canonical")
    records = {}
    for identity, value in body["records"].items():
        require(hash_value(identity) and isinstance(value, str)
                and len(value) <= (MAX_RECORD_BYTES + 2) // 3 * 4, "Invalid inventory record")
        signed = base64.b64decode(value, validate=True)
        note_body = decode(signed, MAX_RECORD_BYTES)["body"]
        created = note_body["created_unix_seconds"]
        record = VerifiedExperience(signed, policy, release_id=release_id, allowed_checkers=allowed_checkers,
                                    now=created, require_live_source=require_live_source)
        require(record.note_id == identity and created <= body["captured_unix_seconds"],
                "Inventory record identity or creation clock differs")
        records[identity] = signed
    return body, records


def freeze(store, config, request_id):
    require(token(request_id) and config.device_id == store.device_id
            and config.release_id == store.release_id and token(config.individual_id),
            "Inventory belongs to another diagnostic runtime")
    with store.db() as db:
        store.check_head(db)
        rows = db.execute("SELECT note_id,raw FROM notes ORDER BY note_id").fetchall()
        head = list(db.execute("SELECT notes,issued,confirmed FROM head WHERE singleton=1").fetchone())
        captured = store.now()
    body = {"schema": SCHEMA, "policy_id": store.policy.policy_id, "release_id": store.release_id,
            "producer_device_id": store.device_id, "producer_individual_id": config.individual_id,
            "request_id": request_id, "store_id": store.store_id, "captured_unix_seconds": captured,
            "head": head, "records": {identity: base64.b64encode(bytes(raw)).decode() for identity, raw in rows},
            "allowed_checkers": sorted(store.allowed_checkers), "require_live_source": store.require_live_source}
    raw = canonical({"body": body, "signature": base64.b64encode(
        store.device_key.sign(PREFIX + canonical(body))).decode()})
    return FrozenExperience(raw, store.policy, release_id=store.release_id,
                            allowed_checkers=store.allowed_checkers,
                            require_live_source=store.require_live_source, clock=store.clock)


class FrozenExperience:
    """A task sees this fixed set even if peer exchange adds more procedures."""

    def __init__(self, raw, policy, *, release_id, allowed_checkers, require_live_source, clock):
        body, records = check_snapshot(raw, policy, release_id=release_id, allowed_checkers=allowed_checkers,
                                       require_live_source=require_live_source)
        self.raw, self.policy, self.release_id, self.clock = raw, policy, release_id, clock
        self.allowed_checkers, self.require_live_source = frozenset(allowed_checkers), require_live_source
        self.store_id, self.request_id = body["store_id"], body["request_id"]
        self.records = MappingProxyType(records)
        self.source_context = None
        self.preparation_seconds = 0.0

    def now(self):
        value = self.clock()
        require(moment(value), "Invalid frozen-inventory clock")
        return int(value)

    def get(self, identity):
        require(identity in self.records, "Procedure was not in the frozen inventory")
        return VerifiedExperience(self.records[identity], self.policy, release_id=self.release_id,
                                  allowed_checkers=self.allowed_checkers, now=self.now(),
                                  require_live_source=self.require_live_source)

    def inventory(self):
        now = self.now()
        return sorted(identity for identity, raw in self.records.items()
                      if (body := json.loads(raw)["body"])["created_unix_seconds"]
                      <= now < body["expires_unix_seconds"])


def commit_snapshot(raw, policy, verifier_key, *, release_id, allowed_checkers, require_live_source, now):
    body, _ = check_snapshot(raw, policy, release_id=release_id, allowed_checkers=allowed_checkers,
                             require_live_source=require_live_source)
    verifier = key_id(public_bytes(verifier_key.public_key()))
    policy.peer_key(verifier)
    require(integer(now) and 0 <= now - body["captured_unix_seconds"] <= 10
            and verifier != body["producer_device_id"], "Inventory commitment is stale or not independent")
    committed = {"schema": RECEIPT_SCHEMA, "policy_id": policy.policy_id, "release_id": release_id,
                 "producer_device_id": body["producer_device_id"],
                 "producer_individual_id": body["producer_individual_id"],
                 "verifier_device_id": verifier, "request_id": body["request_id"],
                 "store_id": body["store_id"], "inventory_sha256": sha(raw),
                 "committed_unix_seconds": now}
    receipt = canonical({"body": committed, "signature": base64.b64encode(
        verifier_key.sign(RECEIPT_PREFIX + canonical(committed))).decode()})
    require(len(receipt) <= MAX_COMMIT_BYTES, "Inventory commitment exceeds its bound")
    return receipt


def check_commit(receipt, raw, policy, *, release_id, producer_id, request_id, verifier_id=None):
    envelope = decode(receipt, MAX_COMMIT_BYTES)
    exact(envelope, ("body", "signature"))
    body = envelope["body"]
    exact(body, ("schema", "policy_id", "release_id", "producer_device_id", "producer_individual_id",
                 "verifier_device_id", "request_id", "store_id", "inventory_sha256", "committed_unix_seconds"))
    snapshot = decode(raw, MAX_INVENTORY_BYTES)["body"]
    require(body["schema"] == RECEIPT_SCHEMA and body["policy_id"] == policy.policy_id
            and body["release_id"] == release_id and body["producer_device_id"] == producer_id
            and body["request_id"] == request_id and body["inventory_sha256"] == sha(raw)
            and hash_value(body["verifier_device_id"]) and body["verifier_device_id"] != producer_id
            and (verifier_id is None or body["verifier_device_id"] == verifier_id)
            and all(body[name] == snapshot[name] for name in (
                "policy_id", "release_id", "producer_device_id", "producer_individual_id", "request_id", "store_id"))
            and integer(body["committed_unix_seconds"])
            and 0 <= body["committed_unix_seconds"] - snapshot["captured_unix_seconds"] <= 10,
            "Inventory commitment differs from the supplied snapshot")
    verify_signature(policy.peer_key(body["verifier_device_id"]), envelope["signature"],
                     RECEIPT_PREFIX + canonical(body))
    require(canonical(envelope) == receipt, "Inventory commitment encoding is not canonical")
    return body


def check_source_inventory(raw, receipt, context, policy, *, release_id, allowed_checkers,
                           require_live_source, producer_id, request_id, verifier_id):
    body, records = check_snapshot(raw, policy, release_id=release_id, allowed_checkers=allowed_checkers,
                                   require_live_source=require_live_source)
    committed = check_commit(receipt, raw, policy, release_id=release_id, producer_id=producer_id,
                             request_id=request_id, verifier_id=verifier_id)
    exact(context, ("inventory_sha256", "commit_sha256", "model_started_unix_seconds", "preparation_seconds"))
    require(context["inventory_sha256"] == sha(raw) and context["commit_sha256"] == sha(receipt)
            and moment(context["model_started_unix_seconds"]) and moment(context["preparation_seconds"])
            and 0 <= context["preparation_seconds"] <= 15
            and committed["committed_unix_seconds"] <= context["model_started_unix_seconds"]
            <= body["captured_unix_seconds"] + 15,
            "Diagnostic did not use the inventory committed before inference")
    return records
