"""Producer half of a separate-key reader-endorsement exchange.

This module has no diagnostic oracle, reviewed procedure or verifier key.
It signs a bounded source proposal, then accepts a verifier's signed decision
only when it matches the producer's original source and selected reader.
Signatures authenticate assertions; they do not attest a physical host or the
model process that generated a trace. No network service is started here.
"""

import base64
import hashlib
import json
import math

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .capsule import canonical
from .enrollment import key_id, public_bytes
from .experience import (
    PREFIX, ExperienceError, VerifiedExperience, capability_hash, exact,
    hash_value, integer, require, token, unique_object, validate_body,
)


SCHEMA = "organism-reader-source-proposal-v2"
PROPOSAL_PREFIX = b"organism-reader-source-proposal-v2\0"
APPROVAL_PREFIX = b"organism-reader-source-approval-v1\0"
MAX_PROPOSAL_BYTES = 4096
MAX_OBSERVATION_BYTES = 65536
MAX_JOB_BYTES = 512 * 1024
MAX_APPROVAL_BYTES = 16384
MAX_SOURCE_WINDOW_SECONDS = 180
MAX_SOURCE_AGE_SECONDS = 600


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw, maximum):
    require(isinstance(raw, bytes) and 0 < len(raw) <= maximum, "Endorsement input exceeds its bound")
    try:
        return json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise ExperienceError("Invalid endorsement JSON") from exc


def moment(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 4000000000


def source_facts(raw, job):
    document = decode(raw, MAX_OBSERVATION_BYTES)
    exact(job, ("record", "digest"))
    encoded = canonical(job["record"])
    require(len(encoded) <= MAX_JOB_BYTES and sha(encoded) == job["digest"], "Source job identity differs")
    record = job["record"]
    require(record.get("task_kind") == "equipment_interface_v1"
            and record.get("device_id") == document.get("device_id"), "Source diagnostic identity differs")
    events = record.get("trace", {}).get("tool_events", [])
    require(isinstance(events, list) and len(events) <= 8, "Source tool history exceeds its bound")
    configured = []
    for index, event in enumerate(events):
        if event.get("tool") == "configure_reader" and event.get("result", {}).get("status") == "configured":
            configured.append(event["arguments"])
        elif (event.get("tool") == "prepare_reader_from_experience"
              and event.get("result", {}).get("status") == "prepared_from_experience"):
            # This is only an identity claim. The independent checker must
            # reconstruct the nested operation against its committed inventory.
            attempts = [attempt for attempt in record["trace"]["inheritance"]["attempts"]
                        if attempt["tool_event_index"] == index]
            require(len(attempts) == 1, "Inherited reader has no unique source attempt")
            operations = [operation for operation in attempts[0]["operations"]
                          if operation["tool"] == "configure_reader"
                          and operation["result"].get("status") == "configured"]
            require(len(operations) == 1, "Inherited source has no executed reader configuration")
            configured.append(operations[0]["arguments"])
    require(configured, "Source contains no executed reader configuration")
    reader = {key: value for key, value in configured[-1].items() if key != "device_id"}
    context = record.get("trace", {}).get("source_inventory")
    if context is not None:
        exact(context, ("inventory_sha256", "commit_sha256", "model_started_unix_seconds", "preparation_seconds"))
        require(hash_value(context["inventory_sha256"]) and hash_value(context["commit_sha256"])
                and moment(context["model_started_unix_seconds"]) and moment(context["preparation_seconds"])
                and context["preparation_seconds"] <= 15, "Invalid source-inventory context")
    published = {key: document[key] for key in ("device_id", "family", "task")}
    published["sources"] = [{key: value for key, value in source.items() if key != "payload"}
                            for source in document["sources"]]
    return {"job_sha256": job["digest"], "observation_sha256": sha(raw),
            "reader_sha256": sha(canonical(reader)), "capability_sha256": capability_hash(published),
            "inventory_context_sha256": sha(canonical(context)) if context is not None else None}


def validate_proposal_body(body, policy, release_id):
    exact(body, ("schema", "policy_id", "release_id", "request_id", "producer_device_id",
                 "producer_individual_id", "verifier_device_id", "job_sha256", "observation_sha256",
                 "reader_sha256", "capability_sha256", "source_clock", "source_kind", "clock_window",
                 "inventory_context_sha256"))
    require(body["schema"] == SCHEMA and body["policy_id"] == policy.policy_id
            and hash_value(release_id) and body["release_id"] == release_id,
            "Source proposal belongs to another release or enrollment")
    require(token(body["request_id"]) and token(body["producer_individual_id"]), "Invalid proposal identity")
    for field in ("producer_device_id", "verifier_device_id", "job_sha256", "observation_sha256",
                  "reader_sha256", "capability_sha256"):
        require(hash_value(body[field]), "Invalid proposal evidence identity")
    require(body["producer_device_id"] != body["verifier_device_id"], "Producer cannot endorse its own source")
    require(body["inventory_context_sha256"] is None or hash_value(body["inventory_context_sha256"]),
            "Invalid inventory context identity")
    try:
        policy.peer_key(body["producer_device_id"])
        policy.peer_key(body["verifier_device_id"])
    except ValueError as exc:
        raise ExperienceError("Proposal signer is not enrolled") from exc
    require(moment(body["source_clock"]), "Invalid source clock")
    require(body["source_kind"] in ("live_model", "development_fixture"), "Invalid source provenance")
    window = body["clock_window"]
    if body["source_kind"] == "live_model":
        require(isinstance(window, list) and len(window) == 2 and all(moment(value) for value in window)
                and 0 <= window[1] - window[0] <= MAX_SOURCE_WINDOW_SECONDS
                and window[0] <= body["source_clock"] <= window[1], "Invalid live source clock window")
    else:
        require(window is None, "Fixture source cannot carry a live clock window")


def verify_signature(key, signature, payload):
    try:
        raw = base64.b64decode(signature, validate=True)
        Ed25519PublicKey.from_public_bytes(key).verify(raw, payload)
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise ExperienceError("Invalid endorsement signature") from exc


def check_proposal(proposal, policy, *, release_id):
    envelope = decode(proposal, MAX_PROPOSAL_BYTES)
    exact(envelope, ("body", "producer_signature"))
    body = envelope["body"]
    validate_proposal_body(body, policy, release_id)
    verify_signature(policy.peer_key(body["producer_device_id"]), envelope["producer_signature"],
                     PROPOSAL_PREFIX + canonical(body))
    return body


def propose(raw, job, *, policy, producer_key, producer_individual_id, verifier_device_id,
            release_id, request_id, source_clock, live, clock_window=None):
    require(type(live) is bool, "Source provenance must be explicit")
    facts = source_facts(raw, job)
    body = {"schema": SCHEMA, "policy_id": policy.policy_id, "release_id": release_id,
            "request_id": request_id, "producer_device_id": key_id(public_bytes(producer_key.public_key())),
            "producer_individual_id": producer_individual_id, "verifier_device_id": verifier_device_id,
            **facts, "source_clock": source_clock,
            "source_kind": "live_model" if live else "development_fixture", "clock_window": clock_window}
    validate_proposal_body(body, policy, release_id)
    result = canonical({"body": body, "producer_signature": base64.b64encode(
        producer_key.sign(PROPOSAL_PREFIX + canonical(body))).decode()})
    require(len(result) <= MAX_PROPOSAL_BYTES, "Source proposal exceeds its bound")
    return result


def check_approval(proposal, approval, *, policy, release_id, allowed_checkers, now,
                   require_live_source=True):
    source = check_proposal(proposal, policy, release_id=release_id)
    require(type(require_live_source) is bool and integer(now), "Invalid source admission policy or clock")
    approved = decode(approval, MAX_APPROVAL_BYTES)
    exact(approved, ("proposal_id", "body", "verifier_signature", "approval_signature"))
    require(approved["proposal_id"] == sha(canonical(source)), "Approval belongs to another source proposal")
    body = approved["body"]
    validate_body(body, policy)
    require(isinstance(allowed_checkers, (set, frozenset)) and 1 <= len(allowed_checkers) <= 4
            and all(hash_value(value) for value in allowed_checkers), "Trusted checker set must be explicit")
    require(all(body[name] == source[name] for name in
                ("policy_id", "release_id", "producer_device_id", "producer_individual_id", "capability_sha256"))
            and body["verification"]["device_id"] == source["verifier_device_id"]
            and body["verification"]["checker_sha256"] in allowed_checkers
            and all(body["source"][name] == source[name] for name in ("job_sha256", "observation_sha256"))
            and body["source"]["kind"] == source["source_kind"]
            and sha(canonical(body["reader"])) == source["reader_sha256"],
            "Approval differs from the producer's source, reader or trusted verifier")
    require(not require_live_source or source["source_kind"] == "live_model",
            "Development fixture cannot stand in for a live source")
    require(body["created_unix_seconds"] <= now < body["expires_unix_seconds"],
            "Approval is expired or not yet valid")
    require(source["source_clock"] < body["created_unix_seconds"] + 1
            and body["created_unix_seconds"] - source["source_clock"] <= MAX_SOURCE_AGE_SECONDS,
            "Approval is outside the source verification window")
    verify_signature(policy.peer_key(source["verifier_device_id"]), approved["approval_signature"],
                     APPROVAL_PREFIX + canonical({key: value for key, value in approved.items()
                                                  if key != "approval_signature"}))
    payload = canonical(body)
    verify_signature(policy.peer_key(source["verifier_device_id"]), approved["verifier_signature"],
                     PREFIX + b"verifier\0" + payload)
    return body, approved["verifier_signature"]


def accept(proposal, approval, *, policy, producer_key, release_id, allowed_checkers, now,
           require_live_source=True):
    body, verifier_signature = check_approval(
        proposal, approval, policy=policy, release_id=release_id,
        allowed_checkers=allowed_checkers, now=now, require_live_source=require_live_source)
    require(key_id(public_bytes(producer_key.public_key())) == body["producer_device_id"],
            "Producer key differs from the source proposal")
    payload = canonical(body)
    # The producer signs only after the intended verifier and all source
    # bindings pass. The verifier never receives the producer's private key.
    result = canonical({"body": body, "note_id": sha(payload),
                        "producer_signature": base64.b64encode(
                            producer_key.sign(PREFIX + b"producer\0" + payload)).decode(),
                        "verifier_signature": verifier_signature})
    return VerifiedExperience(result, policy, release_id=release_id, allowed_checkers=allowed_checkers,
                              now=now, require_live_source=require_live_source).raw
