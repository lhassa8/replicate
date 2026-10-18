"""Independent source-note, fresh-observation and complete-message verification."""

import argparse
import asyncio
import base64
from importlib.metadata import version
import json
from pathlib import Path

from .audit_equipment_dataset import inspect_record as inspect_original, verify_fixture
from .audit_equipment_diagnostics import inspect_job
from .audit_inherited_diagnostics import inspect_inherited_job
from .audit_runtime_dataset import canonical, digest, require
from .diagnostic_oracle import reconstruct
from .enrollment import Enrollment
from .experience import VerifiedExperience
from .experience_issue import checker_identity


def published(document):
    result = {key: document[key] for key in ("family", "task")}
    result["sources"] = [{key: value for key, value in source.items() if key != "payload"}
                         for source in document["sources"]]
    return result


def verify_knowledge(case, packet):
    require(set(packet) == {"schema", "context", "records", "sources"}
            and packet["schema"] == "organism-inheritance-development-support-v1",
            "Unexpected development knowledge packet")
    context = packet["context"]
    require(set(context) == {"signed_enrollment", "owner_public", "reader_device_id",
                            "checker_sha256", "checker_sources", "release_id"},
            "Unexpected public verification context")
    checker, checker_sources = checker_identity()
    fixture_release = digest(b"Public experience component fixture, not a released organism")
    require(context["checker_sha256"] == checker and context["checker_sources"] == checker_sources
            and context["release_id"] == fixture_release, "Knowledge checker or fixture release changed")
    policy = Enrollment(base64.b64decode(context["signed_enrollment"], validate=True),
                        base64.b64decode(context["owner_public"], validate=True))
    policy.peer_key(context["reader_device_id"])
    expected_count = {"applicable": 1, "absent": 0, "conflicting": 2}[case["knowledge_state"]]
    require(isinstance(packet["records"], dict) and len(packet["records"]) == expected_count
            and set(packet["sources"]) == set(packet["records"]), "Knowledge inventory or source evidence differs")
    records, omissions = {}, []
    for identity, encoded in packet["records"].items():
        signed = base64.b64decode(encoded, validate=True)
        note = VerifiedExperience(signed, policy, release_id=fixture_release, allowed_checkers={checker},
                                  now=case["clock"], require_live_source=False)
        require(note.note_id == identity and note.body["created_unix_seconds"] == case["clock"]
                and note.body["expires_unix_seconds"] == case["clock"] + 3600,
                "Development note identity or lifetime differs")
        source = packet["sources"][identity]
        require(set(source) == {"fixture", "job", "reviewed_source_plan", "validation", "omit_validity_counterexample"}
                and type(source["omit_validity_counterexample"]) is bool,
                "Source fixture provenance is incomplete")
        omit = source["omit_validity_counterexample"]
        omissions.append(omit)
        fixture, plan = source["fixture"], source["reviewed_source_plan"]
        raw = canonical(fixture["document"])
        expected_id = "source-" + case["case_id"] + ("-v0" if omit else "-v1")
        require(fixture["case_id"] == fixture["document"]["device_id"] == expected_id
                and published(fixture["document"]) == published(case["fixture"]["document"]),
                "Source identity or published capability differs")
        require(note.body["capability_sha256"] == digest(canonical(published(fixture["document"])))
                and note.body["reader"] == {key: value for key, value in plan.items()
                                            if key not in ("schema", "operation")}
                and plan["operation"] == {"kind": "scalar"}, "Source procedure or capability binding differs")
        require((plan["validity_selector"] is None) == omit,
                "Conflict fixture does not match its declared validity omission")
        reconstructed = reconstruct(raw, plan, now=case["clock"])
        require(all(fixture[key] == reconstructed[key] for key in
                    ("expected_reader", "expected_status", "expected_value", "expected_unit")),
                "Source private truth disagrees with independent reconstruction")
        expected_status = "high" if case["variation"] == "nominal" else "normal"
        require(reconstructed["expected_status"] == expected_status
                and reconstructed["expected_value"] != case["fixture"]["expected_value"],
                "Source answer was copied from the current task")
        # Recreate the issuer's numeric representation from the independent
        # oracle. Python equality treats 3 and 3.0 alike; signed JSON does not.
        reconstructed_case = {**fixture, **{key: reconstructed[key] for key in
                              ("expected_reader", "expected_status", "expected_value", "expected_unit")}}
        checked = inspect_job(reconstructed_case, raw, source["job"])
        require(checked["semantic_passed"] is True, "Source fixture failed independent tool verification")
        validation = {
            "schema": "organism-experience-source-check-v1", "source_job_sha256": source["job"]["digest"],
            "observation_sha256": digest(raw), "reviewed_source_plan_sha256": reconstructed["oracle_plan_sha256"],
            "checker_sha256": checker, "checker_sources": checker_sources,
            "reconstruction": reconstructed, "trace_check": checked, "source_kind": "development_fixture",
            "procedure_semantic_review_claimed_by_code": False,
            "independent_physical_hosts_verified": False, "qualification_passed": False,
        }
        require(canonical(source["validation"]) == canonical(validation), "Retained source validation differs")
        require(note.body["source"] == {
            "job_sha256": source["job"]["digest"], "observation_sha256": digest(raw),
            "validation_sha256": digest(canonical(validation)), "kind": "development_fixture",
        }, "Signed note does not bind the independently checked source")
        require(note.body["producer_device_id"] != context["reader_device_id"]
                and note.body["verification"]["device_id"] != context["reader_device_id"],
                "Source signer identities overlap the fixture reader")
        records[identity] = signed
    require(sorted(omissions) == ([] if expected_count == 0 else [False] if expected_count == 1 else [False, True]),
            "Unexpected conflict counterexample inventory")
    return policy, records


def verify_messages(record, contract):
    case, events = record["case"], record["events"]
    final = "Diagnosis recorded."
    require(contract["final_acknowledgement"] == final, "Inherited acknowledgement changed")
    user = "Diagnose the equipment " + case["case_id"] + "."
    strands = [{"role": "user", "content": [{"text": user}]}]
    wire = [{"role": "system", "content": contract["system_prompt"]},
            {"role": "user", "content": [{"type": "text", "text": user}]}]
    native = [{"role": "system", "content": contract["system_prompt"]}, {"role": "user", "content": user}]
    for index, event in enumerate(events):
        identity = "data-" + case["case_id"] + "-" + str(index)
        text = canonical(event["result"]).decode()
        call = {"name": event["tool"], "input": event["arguments"], "toolUseId": identity}
        result = {"toolUseId": identity, "status": "success", "content": [{"text": text}]}
        strands.extend([{"role": "assistant", "content": [{"toolUse": call}]},
                        {"role": "user", "content": [{"toolResult": result}]}])
        native.extend([{"role": "assistant", "content": "", "tool_calls": [{
            "type": "function", "id": identity,
            "function": {"name": event["tool"], "arguments": event["arguments"]}}]},
            {"role": "tool", "tool_call_id": identity, "content": text}])
        wire.extend([{"role": "assistant", "content": [], "tool_calls": [{
            "type": "function", "id": identity,
            "function": {"name": event["tool"], "arguments": json.dumps(event["arguments"])}}]},
            {"role": "tool", "tool_call_id": identity, "content": [{"type": "text", "text": text}]}])
    strands.append({"role": "assistant", "content": [{"text": final}]})
    wire.append({"role": "assistant", "content": [{"type": "text", "text": final}]})
    native.append({"role": "assistant", "content": final})
    for field, expected in (("strands_messages", strands), ("wire_messages", wire), ("messages", native)):
        require(record[field] == expected, "Inherited complete messages differ: " + field)
    require(all(key not in canonical(native).decode() for key in
                ("expected_status", "expected_reader", "source_job_sha256", "private_key", "validation_sha256")),
            "Source oracle or private context entered model messages")


def inspect_record(record, contract, release):
    if record["schema"] != "organism-inherited-equipment-runtime-trajectory-v1":
        return inspect_original(record, contract, release)
    require(record["role"] == "equipment_inherited" and "supervised_assistants" not in record,
            "Inherited references must supervise every complete assistant target")
    body = {key: value for key, value in contract.items() if key != "sha256"}
    require(digest(canonical(body)) == contract["sha256"] == record["tool_contract_sha256"],
            "Inherited tool contract changed")
    require(record["provenance"] == {
        "kind": "locally_authored_inherited_reader_runtime_reference",
        "teacher_model_used": False, "human_review_claimed": False,
        "real_peer_transport_used": False, "native_process_used": False,
        "source_kind": "development_fixture", "source_procedure_semantic_review_claimed": False,
        "independent_physical_hosts_verified": False, "private_signing_keys_retained": False,
        "replay_identity_mode": "public_only_fixture_no_signer",
        "training_admitted": False, "qualification_data_present": False,
    }, "Inherited reference provenance changed")
    case = record["case"]
    require(all(record[key] == case[key] for key in ("case_id", "role", "split", "profile_id"))
            and case["seed"] == 3 and case["knowledge_state"] in ("applicable", "absent", "conflicting"),
            "Inherited case binding or development selection differs")
    verify_fixture(case)
    policy, records = verify_knowledge(case, record["knowledge"])
    names = ["prepare_reader_from_experience"]
    if case["knowledge_state"] != "applicable":
        names.append("configure_reader")
    names += ["read_equipment", "submit_diagnosis"]
    events = record["events"]
    require([event["tool"] for event in events] == names
            and record["actions"] == [{key: event[key] for key in ("tool", "arguments")} for event in events],
            "Inherited reference repeats setup or omits a required fresh action")
    require(len(record["inheritance"]["attempts"]) == 1, "Reference must contain one charged inheritance lookup")
    job_record = {"device_id": case["case_id"], "task_kind": "equipment_interface_v1", "status": "finished",
                  "submission": record["submission"],
                  "trace": {"tool_events": events, "inheritance": record["inheritance"]}}
    context = record["knowledge"]["context"]
    checked = inspect_inherited_job(case["fixture"], canonical(case["fixture"]["document"]),
        {"record": job_record, "digest": digest(canonical(job_record))}, policy=policy,
        release_id=context["release_id"], allowed_checkers={context["checker_sha256"]},
        records=records, require_live_source=False)
    require(checked["semantic_passed"] is True and checked["reader_operations"] == 4
            and bool(checked["inherited_reader_uses"]) == (case["knowledge_state"] == "applicable"),
            "Inherited reference failed independent fresh-observation checks")
    verify_messages(record, contract)
    return {
        "passed": True, "semantic_answer_verified": True, "decimal_fixture_truth_verified": True,
        "signed_source_evidence_verified": True, "independent_record_inventory_verified": True,
        "complete_assistant_targets": len(events) + 1, "final_acknowledgement_verified": True,
        "reader_operations": checked["reader_operations"], "inherited_reader_uses": checked["inherited_reader_uses"],
        "runtime_call_budget_compatible": len(events) + 1 <= 6,
        "source_kind": "development_fixture", "timing_replay_policy": "semantic_fields_exact_measured_times_retained",
        "training_admitted": False, "native_execution_verified": False, "qualification_passed": False,
    }


async def audit(source):
    from strands.models import LlamaCppModel
    from . import inheritance_dataset as dataset
    from .runtime_dataset import provider_sources, stock_release

    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_bytes())
    require(manifest["schema"] == "organism-inherited-equipment-runtime-corpus-v1"
            and manifest["dataset_id"] == dataset.DATASET_ID
            and manifest.get("identifier_policy") == dataset.IDENTIFIER_POLICY
            and manifest["scope"] == "development_references_through_current_runtime_tools"
            and manifest["strands_version"] == version("strands-agents") == "1.57.1",
            "Inherited corpus or runtime version differs")
    require(all(manifest[key] is False for key in
                ("training_admitted", "optimization_started", "qualification_data_present", "human_review_claimed")),
            "Inherited corpus overclaims admission or qualification")
    require(set(manifest["files"]) == {"train.jsonl", "validation.jsonl", "contracts.json", "release.json"},
            "Unexpected inherited corpus file inventory")
    for name, expected in manifest["files"].items():
        raw = (source / name).read_bytes()
        require(expected == {"bytes": len(raw), "sha256": digest(raw)}, "Inherited corpus file hash changed")
    paths = {name: Path(dataset.__file__).parent / name for name in dataset.SOURCES}
    paths.update(provider_sources())
    require(set(paths) == set(manifest["sources"]), "Inherited corpus source inventory differs")
    for name, path in paths.items():
        require(digest(path.read_bytes()) == digest((source / "sources" / name).read_bytes()) == manifest["sources"][name],
                "Current or retained inherited source changed: " + name)
    cases = {case["case_id"]: case for case in dataset.development_cases()}
    require(len(cases) == 464, "Incomplete declared pilot inventory")
    definitions = json.loads((source / "contracts.json").read_bytes())
    release = json.loads((source / "release.json").read_bytes())
    require(release == stock_release(Path(dataset.__file__).parent), "Original release fixture metadata changed")
    model = LlamaCppModel(base_url="http://127.0.0.1:9", model_id="reference-formatting-only",
                         params={"temperature": 0, "max_tokens": 192})
    counts, seen, profiles, contexts = {}, set(), set(), set()
    try:
        require(dataset.contracts(model) == definitions, "Current inherited provider contract differs")
        for split in ("train", "validation"):
            records = [json.loads(line) for line in (source / (split + ".jsonl")).read_text().splitlines()]
            split_profiles = {record["profile_id"] for record in records}
            require(not profiles & split_profiles, "Inherited development profiles cross splits")
            profiles.update(split_profiles)
            for record in records:
                identity = record["case_id"]
                require(identity not in seen and record["case"] == cases.get(identity) and record["split"] == split,
                        "Duplicate, moved or changed pilot fixture")
                seen.add(identity)
                require(inspect_record(record, definitions[record["role"]], release) == record["evaluation"],
                        "Retained inherited semantic audit differs")
                if record["role"] == "equipment":
                    require(all(record["case"]["original_case_id"] not in canonical(record[field]).decode()
                                for field in ("messages", "wire_messages", "strands_messages")),
                            "Original condition-bearing identity entered model messages")
                if record["role"] == "equipment_inherited":
                    contexts.add(canonical(record["knowledge"]["context"]))
                    for source_record in record["knowledge"]["sources"].values():
                        require(source_record["fixture"] == dataset.source_fixture(record["case"],
                            omit_validity=source_record["omit_validity_counterexample"]),
                            "Source fixture differs from its declared development inventory")
                await dataset.replay(record, model, definitions[record["role"]], release)
            counts[split] = dataset.split_counts(records, definitions)
        require(seen == set(cases) and counts == manifest["counts"] and len(contexts) == 1,
                "Pilot coverage, public context or counts differ")
        require(counts["train"]["roles"] == {"equipment": 180, "reproduction": 60, "equipment_inherited": 108}
                and counts["validation"]["roles"] == {"equipment": 60, "reproduction": 20, "equipment_inherited": 36}
                and counts["train"]["assistant_targets"] == 1584
                and counts["validation"]["assistant_targets"] == 528
                and counts["train"]["knowledge_states"] == {"applicable": 36, "absent": 36, "conflicting": 36}
                and counts["validation"]["knowledge_states"] == {"applicable": 12, "absent": 12, "conflicting": 12}
                and counts["train"]["context_only_injected_actions"] == 6
                and counts["validation"]["context_only_injected_actions"] == 2, "Pilot allocation or balance changed")
    finally:
        await model.client.aclose()
    return {"passed": True, "source_manifest_sha256": digest((source / "manifest.json").read_bytes()),
            "counts": counts, "current_runtime_replay_verified": True, "independent_semantics_verified": True,
            "provider_formatting_verified": True, "context_only_bad_actions_verified": 8,
            "opaque_equipment_identifiers_verified": 240,
            "signed_fixture_source_notes_verified": 144, "human_review_claimed": False,
            "serving_template_parity_verified": False, "training_admitted": False, "qualification_passed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as destination:
        try:
            result = asyncio.run(audit(args.source))
        except Exception as exc:
            result = {"passed": False, "failure": type(exc).__name__ + ": " + str(exc), "training_admitted": False}
        destination.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
