"""Research verifier for source-task endorsements, excluded from deployment.

The reviewer supplies the source procedure independently of the agent's trace.
This issuer checks that trace and reconstructs its diagnosis before signing.
It does not prove that a reviewed procedure matches natural-language equipment
documentation, or that two signer identities ran on separate physical hosts.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .audit_equipment_diagnostics import inspect_job
from .diagnostic_oracle import reconstruct
from .enrollment import key_id, public_bytes
from .experience import SCHEMA, capability_hash, require, seal_record


CHECKER_SOURCES = ("experience_issue.py", "diagnostic_oracle.py", "audit_equipment_diagnostics.py",
                   "audit_application.py", "experience.py", "enrollment.py", "capsule.py")


def checker_identity():
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in CHECKER_SOURCES}
    raw = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest(), sources


def issue(raw, job, reviewed_source_plan, *, policy, producer_key, verifier_key,
          producer_individual_id, release_id, created, expires, source_clock,
          live, live_clock_window=None):
    require(type(live) is bool, "Source provenance must be explicit")
    if live:
        require(job.get("record", {}).get("trace", {}).get("model_calls")
                and live_clock_window is not None,
                "Live experience requires source-model evidence and an observed clock window")
    result = reconstruct(raw, reviewed_source_plan, now=source_clock)
    require(type(created) is int and source_clock < created + 1,
            "Experience endorsement predates the source observation")
    document = json.loads(raw)
    case = {"case_id": document["device_id"], "document": document,
            **{key: result[key] for key in ("expected_reader", "expected_status", "expected_value", "expected_unit")}}
    reads = [event["result"] for event in job["record"]["trace"]["tool_events"]
             if event["tool"] == "read_equipment" and event["result"].get("status") == "observed"]
    require(len(reads) >= 1 and reads[-1]["read_unix_seconds"] == source_clock,
            "Source evidence clock differs from the verified reading")
    checked = inspect_job(case, raw, job, live=live, clock_window=live_clock_window)
    require(checked["semantic_passed"] is True and checked["expected_status"] == result["expected_status"]
            and checked["expected_value"] == result["expected_value"], "Source task failed independent verification")
    require(result["expected_status"] != "unknown", "Inconclusive source evidence cannot certify a reader procedure")
    checker_id, sources = checker_identity()
    receipt = {"schema": "organism-experience-source-check-v1",
               "source_job_sha256": job["digest"], "observation_sha256": hashlib.sha256(raw).hexdigest(),
               "reviewed_source_plan_sha256": result["oracle_plan_sha256"], "checker_sha256": checker_id,
               "checker_sources": sources, "reconstruction": result, "trace_check": checked,
               "source_kind": "live_model" if live else "development_fixture",
               "procedure_semantic_review_claimed_by_code": False,
               "independent_physical_hosts_verified": False, "qualification_passed": False}
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    published = {key: document[key] for key in ("device_id", "family", "task")}
    published["sources"] = [{key: value for key, value in source.items() if key != "payload"}
                            for source in document["sources"]]
    reader = deepcopy(result["expected_reader"])
    reader.pop("device_id")
    body = {
        "schema": SCHEMA, "policy_id": policy.policy_id, "release_id": release_id,
        "producer_device_id": key_id(public_bytes(producer_key.public_key())),
        "producer_individual_id": producer_individual_id, "created_unix_seconds": created,
        "expires_unix_seconds": expires, "capability_sha256": capability_hash(published), "reader": reader,
        "source": {"job_sha256": job["digest"], "observation_sha256": hashlib.sha256(raw).hexdigest(),
                   "validation_sha256": hashlib.sha256(canonical).hexdigest(), "kind": receipt["source_kind"]},
        "verification": {"device_id": key_id(public_bytes(verifier_key.public_key())),
                         "checker_sha256": checker_id, "outcome": "passed"},
    }
    return seal_record(body, producer_key, verifier_key, policy), receipt
