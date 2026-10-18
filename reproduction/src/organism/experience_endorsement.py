"""Source verification using an independently selected local procedure.

The research caller supplies its private procedure. The installed reviewer
selects one from release-bound references instead. Neither path accepts a
procedure supplied by the producer or receives the producer's private key.
The original research issuer stays unchanged because draft inputs pin it.
"""

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .audit_equipment_diagnostics import inspect_job
from .audit_inherited_diagnostics import inspect_inherited_job
from .capsule import canonical
from .diagnostic_oracle import reconstruct
from .enrollment import key_id, public_bytes
from .experience import PREFIX, SCHEMA, integer, require, validate_body
from .experience_proposal import (
    APPROVAL_PREFIX, MAX_APPROVAL_BYTES, MAX_SOURCE_AGE_SECONDS, check_proposal, sha, source_facts,
)
from .source_inventory import check_source_inventory


# Preserve the research identity inputs without importing its two-key issuer
# into an installed individual. Native calls supply their own bound context.
LEGACY_CHECKER_SOURCES = (
    "experience_issue.py", "diagnostic_oracle.py", "audit_equipment_diagnostics.py",
    "audit_application.py", "experience.py", "enrollment.py", "capsule.py")
CHECKER_SOURCES = tuple(sorted(set(LEGACY_CHECKER_SOURCES) | {
    "experience_endorsement.py", "experience_proposal.py", "source_inventory.py",
    "audit_inherited_diagnostics.py", "experience_store.py", "review_inbox.py"}))


def checker_identity():
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in CHECKER_SOURCES}
    return sha(canonical(sources)), sources


def review_source(proposal, raw, job, reviewed_source_plan, *, policy, verifier_device_id, release_id,
                  created, expires, require_live_source=True, source_inventory=None,
                  inventory_checkers=None, checker_context=None):
    source = check_proposal(proposal, policy, release_id=release_id)
    require(verifier_device_id == source["verifier_device_id"], "Verifier differs from the proposed destination")
    require(type(require_live_source) is bool, "Source admission policy must be explicit")
    live = source["source_kind"] == "live_model"
    require(live or not require_live_source, "Development fixture cannot stand in for a live source")
    facts = source_facts(raw, job)
    require(all(source[name] == value for name, value in facts.items()), "Source artifacts differ from proposal")
    require(integer(created) and integer(expires)
            and source["source_clock"] < created + 1
            and created - source["source_clock"] <= MAX_SOURCE_AGE_SECONDS,
            "Endorsement is outside the source verification window")
    if live:
        require(job.get("record", {}).get("trace", {}).get("model_calls"),
                "Live experience requires source-model evidence")
    result = reconstruct(raw, reviewed_source_plan, now=source["source_clock"])
    document = json.loads(raw)
    case = {"case_id": document["device_id"], "document": document,
            **{key: result[key] for key in ("expected_reader", "expected_status", "expected_value", "expected_unit")}}
    reads = [event["result"] for event in job["record"]["trace"]["tool_events"]
             if event["tool"] == "read_equipment" and event["result"].get("status") == "observed"]
    require(reads and reads[-1]["read_unix_seconds"] == source["source_clock"],
            "Source evidence clock differs from its proposal")
    trace = job["record"]["trace"]
    inherited = any(event["tool"] == "prepare_reader_from_experience" for event in trace["tool_events"])
    context = trace.get("source_inventory")
    if context is not None:
        require(source_inventory is not None and source_inventory["context"] == context
                and inventory_checkers is not None, "Independent committed inventory was not supplied")
        records = check_source_inventory(
            source_inventory["snapshot"], source_inventory["receipt"], context, policy,
            release_id=release_id, allowed_checkers=inventory_checkers, require_live_source=require_live_source,
            producer_id=source["producer_device_id"], request_id=source["request_id"],
            verifier_id=verifier_device_id)
        require(json.loads(source_inventory["snapshot"])["body"]["producer_individual_id"]
                == source["producer_individual_id"], "Inventory belongs to another producer individual")
        require(context["model_started_unix_seconds"] <= source["source_clock"]
                and trace.get("seconds", -1) >= context["preparation_seconds"],
                "Inventory preparation was not completed and charged before the reading")
        checked = inspect_inherited_job(
            case, raw, job, policy=policy, release_id=release_id, allowed_checkers=inventory_checkers,
            records=records, require_live_source=require_live_source, live=live,
            clock_window=source["clock_window"])
        require(trace["seconds"] + 0.002 >=
                context["preparation_seconds"] + checked["inheritance_lookup_seconds"],
                "Inventory commitment and reader lookup effort were not both charged")
    else:
        require(not inherited and source_inventory is None, "Inherited source requires a committed inventory")
        checked = inspect_job(case, raw, job, live=live, clock_window=source["clock_window"])
    require(checked["semantic_passed"] is True and checked["expected_status"] == result["expected_status"]
            and checked["expected_value"] == result["expected_value"], "Source task failed independent verification")
    require(result["expected_status"] != "unknown", "Inconclusive source cannot certify a reader")
    if checker_context is None:
        checker, sources = checker_identity()
    else:
        # This is local service configuration, never part of a peer request.
        require(set(checker_context) == {"checker_sha256", "checker_sources", "reference_catalog_sha256"}
                and checker_context["checker_sha256"] == sha(canonical({
                    "sources": checker_context["checker_sources"],
                    "reference_catalog_sha256": checker_context["reference_catalog_sha256"]})),
                "Installed checker context differs from its references and sources")
        checker, sources = checker_context["checker_sha256"], checker_context["checker_sources"]
    receipt = {"schema": "organism-separate-source-check-v1", "proposal_id": sha(canonical(source)),
               "source_proposal_sha256": sha(proposal), "source_job_sha256": job["digest"],
               "observation_sha256": sha(raw), "reviewed_source_plan_sha256": result["oracle_plan_sha256"],
               "checker_sha256": checker, "checker_sources": sources,
               "reconstruction": result, "trace_check": checked, "source_kind": source["source_kind"],
               "producer_signature_verified": True, "producer_private_key_available_to_verifier": False,
               "signed_pre_inference_inventory_commit_verified": context is not None,
               "source_inventory_context": context,
               "procedure_semantic_review_claimed_by_code": False,
               "source_model_process_independently_attested": False,
               "independent_physical_hosts_verified": False, "qualification_passed": False}
    if checker_context is not None:
        receipt["reference_catalog_sha256"] = checker_context["reference_catalog_sha256"]
    reader = deepcopy(result["expected_reader"])
    reader.pop("device_id")
    require(sha(canonical(reader)) == source["reader_sha256"], "Verified reader differs from proposed reader")
    body = {"schema": SCHEMA, "policy_id": policy.policy_id, "release_id": release_id,
            "producer_device_id": source["producer_device_id"],
            "producer_individual_id": source["producer_individual_id"],
            "created_unix_seconds": created, "expires_unix_seconds": expires,
            "capability_sha256": source["capability_sha256"], "reader": reader,
            "source": {"job_sha256": job["digest"], "observation_sha256": sha(raw),
                       "validation_sha256": sha(canonical(receipt)), "kind": source["source_kind"]},
            "verification": {"device_id": verifier_device_id, "checker_sha256": checker, "outcome": "passed"}}
    validate_body(body, policy)
    return body, receipt


def endorse(proposal, raw, job, reviewed_source_plan, *, policy, verifier_key, release_id,
            created, expires, require_live_source=True, source_inventory=None, inventory_checkers=None,
            checker_context=None):
    body, receipt = review_source(
        proposal, raw, job, reviewed_source_plan, policy=policy,
        verifier_device_id=key_id(public_bytes(verifier_key.public_key())), release_id=release_id,
        created=created, expires=expires, require_live_source=require_live_source,
        source_inventory=source_inventory, inventory_checkers=inventory_checkers, checker_context=checker_context)
    approved = {"proposal_id": receipt["proposal_id"], "body": body,
                "verifier_signature": base64.b64encode(
                    verifier_key.sign(PREFIX + b"verifier\0" + canonical(body))).decode()}
    approved["approval_signature"] = base64.b64encode(
        verifier_key.sign(APPROVAL_PREFIX + canonical(approved))).decode()
    approval = canonical(approved)
    require(len(approval) <= MAX_APPROVAL_BYTES, "Endorsement response exceeds its bound")
    return approval, receipt
