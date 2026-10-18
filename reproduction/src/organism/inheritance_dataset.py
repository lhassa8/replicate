"""Development-only reader inheritance through the actual typed runtime tools.

Retain all 320 original scenarios, conceal equipment outcome labels in task IDs,
and add 144 cases within the 500-trajectory pilot allocation.
Signing keys exist only during fixture construction. Replay uses
a public-only identity handle with the actual ExperienceStore; it cannot sign
copy receipts and does not establish possession of a device's private key.
"""

import argparse
import asyncio
import base64
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from strands import Agent
from strands.models import LlamaCppModel

from .audit_runtime_dataset import require
from .capsule import canonical
from .check_template_parity import native_messages
from .diagnostic_agent import EquipmentTool
from .enrollment import Enrollment
from .equipment_baseline import solve
from .equipment_diagnostic_agent import EXPERIENCE_SPECS, EXPERIENCE_SYSTEM_PROMPT
from .equipment_experience import ExperiencedInterfaceSession, TOOL_NAME
from .equipment_interfaces import InterfaceDirectory, InterfaceSession
from .experience_issue import checker_identity, CHECKER_SOURCES, issue
from .experience_store import ExperienceStore
from .probe_experience import job as fixture_job, lab, RELEASE
from . import equipment_dataset as original
from . import runtime_dataset as legacy
from .runtime_rows import INHERITED_SCHEMA, INHERITED_CORPUS_SCHEMA, inspect_record, targets


DATASET_ID = "equipment-runtime-development-v3"
IDENTIFIER_POLICY = "opaque-equipment-identifiers-v1"
STATES = ("applicable", "absent", "conflicting")
SOURCE_SCOPE = "development_fixture"
SOURCES = tuple(sorted(set(original.SOURCES) | set(CHECKER_SOURCES) | {
    "inheritance_dataset.py", "audit_inheritance_dataset.py", "equipment_experience.py",
    "audit_inherited_diagnostics.py", "experience_store.py", "probe_experience.py",
}))
PROVENANCE = {
    "kind": "locally_authored_inherited_reader_runtime_reference",
    "teacher_model_used": False, "human_review_claimed": False,
    "real_peer_transport_used": False, "native_process_used": False,
    "source_kind": SOURCE_SCOPE, "source_procedure_semantic_review_claimed": False,
    "independent_physical_hosts_verified": False, "private_signing_keys_retained": False,
    "replay_identity_mode": "public_only_fixture_no_signer",
    "training_admitted": False, "qualification_data_present": False,
}


def inherited_cases():
    # Seed 3 includes the observed 0.12 ratio -> 12 percent error. The original
    # cases retain all five numeric seeds; no qualification family is imported.
    for base in original.equipment_cases():
        if base["seed"] != 3:
            continue
        for state in STATES:
            case = deepcopy(base)
            # Keep condition and knowledge labels in evaluator metadata only.
            # A model must inspect the tool result, not decode its task ID.
            identity = "inherit-data-" + legacy.digest(canonical(
                {"profile": case["profile_id"], "variation": case["variation"], "state": state, "seed": 3}))[:24]
            case["case_id"] = case["fixture"]["case_id"] = case["fixture"]["document"]["device_id"] = identity
            case["fixture"]["expected_reader"]["device_id"] = identity
            case["role"], case["knowledge_state"] = "equipment_inherited", state
            yield case


def opaque_equipment_case(original_case):
    """Change only identity fields; retain the old ID as evaluator metadata."""
    require(original_case["role"] == "equipment" and "original_case_id" not in original_case,
            "Expected an original equipment case")
    case = deepcopy(original_case)
    case["original_case_id"] = case["case_id"]
    identity = "equipment-data-" + legacy.digest(canonical({
        "policy": IDENTIFIER_POLICY, "original_case_id": case["original_case_id"],
    }))[:24]
    case["case_id"] = case["fixture"]["case_id"] = case["fixture"]["document"]["device_id"] = identity
    case["fixture"]["expected_reader"]["device_id"] = identity
    return case


def development_cases():
    for case in original.development_cases():
        yield opaque_equipment_case(case) if case["role"] == "equipment" else case
    yield from inherited_cases()


def contracts(model):
    definitions = original.contracts(model)
    agent = Agent(model=model, tools=[EquipmentTool(None, spec) for spec in EXPERIENCE_SPECS],
                  system_prompt=EXPERIENCE_SYSTEM_PROMPT, callback_handler=None, retry_strategy=None,
                  context_manager=False, background_tasks=False, load_tools_from_directory=False)
    specs = deepcopy(agent.tool_registry.get_all_tool_specs())
    request = model._format_request([], specs, EXPERIENCE_SYSTEM_PROMPT)
    body = {"system_prompt": EXPERIENCE_SYSTEM_PROMPT, "tools": request["tools"], "strands_specs": specs,
            "final_acknowledgement": "Diagnosis recorded."}
    definitions["equipment_inherited"] = {**body, "sha256": legacy.digest(canonical(body))}
    return definitions


def public_context(policy, owner, identities):
    checker, sources = checker_identity()
    return {"signed_enrollment": base64.b64encode(policy.signed).decode(),
            "owner_public": base64.b64encode(owner).decode(), "reader_device_id": identities[2],
            "checker_sha256": checker, "checker_sources": sources, "release_id": RELEASE}


def source_fixture(case, *, omit_validity=False):
    """Opposite-condition source, with exactly matching published capabilities."""
    condition = "fault" if case["variation"] == "nominal" else "nominal"
    source = next(item for item in original.equipment_cases()
                  if item["profile_id"] == case["profile_id"] and item["variation"] == condition
                  and item["seed"] == case["seed"])
    fixture = deepcopy(source["fixture"])
    identity = "source-" + case["case_id"] + ("-v0" if omit_validity else "-v1")
    fixture["case_id"] = fixture["document"]["device_id"] = fixture["expected_reader"]["device_id"] = identity
    if case["variation"] == "scaled_fault":
        data = fixture["document"]["sources"][0]
        data["fields"][0]["scale"] = 0.01
        raw = original.VALUES[case["profile_id"][-1]]["nominal"][case["seed"]][0]
        original.assign(data, fixture["expected_reader"]["value_selectors"][0],
                        original.number_spelling(raw * 100))
    if omit_validity:
        require(fixture["expected_reader"]["validity_selector"] is not None,
                "Conflict fixture needs a declared validity selector")
        fixture["expected_reader"]["validity_selector"] = None
    return fixture


def make_knowledge(case, context, policy, keys):
    packet = {"schema": "organism-inheritance-development-support-v1",
              "context": deepcopy(context), "records": {}, "sources": {}}
    for omit in ([] if case["knowledge_state"] == "absent" else
                 [False, True] if case["knowledge_state"] == "conflicting" else [False]):
        fixture = source_fixture(case, omit_validity=omit)
        raw = canonical(fixture["document"])
        with tempfile.TemporaryDirectory(prefix="inherit-source-") as temporary:
            root = Path(temporary)
            (root / (fixture["case_id"] + ".json")).write_bytes(raw)
            directory = InterfaceDirectory(root, clock=lambda: case["clock"])
            planning = InterfaceSession(directory, fixture["case_id"])
            solve(planning, fixture["case_id"])
            session = InterfaceSession(directory, fixture["case_id"])
            session.inspect_equipment(fixture["case_id"])
            configuration = deepcopy(planning.events[1]["arguments"])
            if omit:
                configuration["validity_selector"] = None
            session.configure_reader(**configuration)
            reading = session.read_equipment(fixture["case_id"])
            answer = deepcopy(planning.events[-1]["arguments"])
            answer["value"] = original.number_spelling(answer["value"])
            answer["evidence_id"] = reading["evidence_id"]
            session.submit_diagnosis(**answer)
            job = fixture_job(session)
        plan = {"schema": "organism-diagnostic-oracle-v1",
                **{key: value for key, value in fixture["expected_reader"].items() if key != "device_id"},
                "operation": {"kind": "scalar"}}
        signed, validation = issue(raw, job, plan, policy=policy, producer_key=keys[0], verifier_key=keys[1],
            producer_individual_id="a" * 32, release_id=context["release_id"],
            created=case["clock"], expires=case["clock"] + 3600, source_clock=case["clock"], live=False)
        identity = json.loads(signed)["note_id"]
        packet["records"][identity] = base64.b64encode(signed).decode()
        packet["sources"][identity] = {
            "fixture": fixture, "job": job, "reviewed_source_plan": plan, "validation": validation,
            "omit_validity_counterexample": omit,
        }
    return packet


class PublicReplayIdentity:
    """Local fixture-only identity; deliberately has no signing operation."""
    def __init__(self, public):
        self.public = Ed25519PublicKey.from_public_bytes(public)

    def public_key(self):
        return self.public


def replay_store(root, case, knowledge):
    context = knowledge["context"]
    policy = Enrollment(base64.b64decode(context["signed_enrollment"], validate=True),
                        base64.b64decode(context["owner_public"], validate=True))
    store = ExperienceStore(root, policy, PublicReplayIdentity(policy.peer_key(context["reader_device_id"])),
        release_id=context["release_id"], allowed_checkers={context["checker_sha256"]},
        require_live_source=False, clock=lambda: case["clock"])
    for identity, encoded in knowledge["records"].items():
        require(store.add(base64.b64decode(encoded, validate=True)) == identity, "Saved note identity changed")
    return store


async def execute_case(case, model, contract, release, knowledge, *, supplied_actions=None, final=None):
    with tempfile.TemporaryDirectory(prefix="inherit-data-") as temporary:
        root = Path(temporary)
        observations = root / "observations"
        observations.mkdir()
        raw = canonical(case["fixture"]["document"])
        path = observations / (case["case_id"] + ".json")
        path.write_bytes(raw)
        directory = InterfaceDirectory(observations, clock=lambda: case["clock"])
        store = replay_store(root / "experience", case, knowledge)
        session = ExperiencedInterfaceSession(directory, case["case_id"], store)
        tools = {spec["name"]: EquipmentTool(session, spec) for spec in EXPERIENCE_SPECS}
        messages = [{"role": "user", "content": [{"text": "Diagnose the equipment " + case["case_id"] + "."}]}]
        actions = []

        async def invoke(action):
            name, args = action["tool"], action["arguments"]
            require(name in tools, "Unknown inherited reference action")
            call = {"name": name, "input": deepcopy(args),
                    "toolUseId": "data-" + case["case_id"] + "-" + str(len(actions))}
            messages.append({"role": "assistant", "content": [{"toolUse": call}]})
            events = [event async for event in tools[name].stream(call, {})]
            require(len(events) == 1 and events[0].tool_result["status"] == "success",
                    "Inherited tool wrapper failed")
            messages.append({"role": "user", "content": [{"toolResult": deepcopy(events[0].tool_result)}]})
            actions.append({"tool": name, "arguments": deepcopy(args)})

        if supplied_actions is None:
            # The fixed reference planner sees only the public descriptor and
            # observation, through a separate actual reader session.
            planning = InterfaceSession(directory, case["case_id"])
            solve(planning, case["case_id"])
            planned = [{key: deepcopy(event[key]) for key in ("tool", "arguments")} for event in planning.events]
            if planned[-1]["arguments"]["value"] is not None:
                planned[-1]["arguments"]["value"] = original.number_spelling(planned[-1]["arguments"]["value"])
            await invoke({"tool": TOOL_NAME, "arguments": {"device_id": case["case_id"]}})
            prepared = session.events[-1]["result"]["status"] == "prepared_from_experience"
            for action in planned[2 if prepared else 1:]:
                await invoke(action)
        else:
            for action in supplied_actions:
                await invoke(action)
        acknowledgement = contract["final_acknowledgement"] if final is None else final
        messages.append({"role": "assistant", "content": [{"text": acknowledgement}]})
        request = model._format_request(messages, contract["strands_specs"], contract["system_prompt"])
        require(request["tools"] == contract["tools"] and path.read_bytes() == raw,
                "Inherited contract or read-only input changed")
        return {"actions": actions, "events": deepcopy(session.events),
                "inheritance": deepcopy(session.inheritance), "strands_messages": messages,
                "wire_messages": request["messages"], "messages": native_messages(request["messages"]),
                "submission": deepcopy(session.submission)}


def invariant_trace(trace):
    """Compare semantic replay exactly; retain but do not equate measured time."""
    result = deepcopy(trace)
    for attempt in result["attempts"]:
        elapsed = attempt.pop("elapsed_seconds")
        require(type(elapsed) in (int, float) and 0 <= elapsed < float("inf"), "Invalid measured lookup time")
    return result


async def trajectory(case, model, contract, release, context, policy, keys):
    if case["role"] != "equipment_inherited":
        return await original.trajectory(case, model, contract, release)
    knowledge = make_knowledge(case, context, policy, keys)
    actual = await execute_case(case, model, contract, release, knowledge)
    record = {"schema": INHERITED_SCHEMA, "case_id": case["case_id"], "role": case["role"],
              "split": case["split"], "profile_id": case["profile_id"], "case": deepcopy(case),
              "tool_contract_sha256": contract["sha256"], "provenance": deepcopy(PROVENANCE),
              "knowledge": knowledge, **actual}
    record["evaluation"] = inspect_record(record, contract, release)
    return record


async def replay(record, model, contract, release):
    if record["role"] != "equipment_inherited":
        return await original.replay(record, model, contract, release)
    actual = await execute_case(record["case"], model, contract, release, record["knowledge"],
        supplied_actions=record["actions"], final=record["messages"][-1]["content"])
    for name, value in actual.items():
        require(invariant_trace(record[name]) == invariant_trace(value) if name == "inheritance"
                else record[name] == value, "Inherited runtime replay differs: " + name)
    checked = inspect_record(record, contract, release)
    require(checked == record["evaluation"], "Inherited reference semantics changed")
    return checked


def split_counts(records, definitions):
    return {
        "trajectories": len(records), "profiles": sorted({record["profile_id"] for record in records}),
        "roles": {role: sum(record["role"] == role for record in records) for role in definitions},
        "assistant_targets": sum(len(list(targets(record))) for record in records),
        "context_only_injected_actions": sum(record["role"] == "equipment"
            and record["case"]["inject_configuration_error"] for record in records),
        "knowledge_states": {state: sum(record["role"] == "equipment_inherited"
            and record["case"]["knowledge_state"] == state for record in records) for state in STATES},
    }


async def build(output):
    require(version("strands-agents") == "1.57.1", "Pinned Strands version required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    release = legacy.stock_release(root)
    policy, owner, keys, identities = lab()
    context = public_context(policy, owner, identities)
    model = LlamaCppModel(base_url="http://127.0.0.1:9", model_id="reference-formatting-only",
                         params={"temperature": 0, "max_tokens": 192})
    groups = {"train": [], "validation": []}
    try:
        definitions = contracts(model)
        for case in development_cases():
            record = await trajectory(case, model, definitions[case["role"]], release, context, policy, keys)
            await replay(record, model, definitions[case["role"]], release)
            groups[case["split"]].append(record)
    finally:
        await model.client.aclose()
    for split, records in groups.items():
        with (output / (split + ".jsonl")).open("x") as target:
            for record in records:
                target.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    for name, value in (("contracts.json", definitions), ("release.json", release)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    (output / "sources").mkdir()
    paths = {name: root / name for name in SOURCES}
    paths.update(legacy.provider_sources())
    source_hashes = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        (output / "sources" / name).write_bytes(raw)
        source_hashes[name] = legacy.digest(raw)
    manifest = {
        "dataset_id": DATASET_ID, "schema": INHERITED_CORPUS_SCHEMA,
        "identifier_policy": IDENTIFIER_POLICY,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "development_references_through_current_runtime_tools",
        "training_admitted": False, "optimization_started": False,
        "qualification_data_present": False, "human_review_claimed": False,
        "counts": {split: split_counts(records, definitions) for split, records in groups.items()},
        "strands_version": version("strands-agents"), "sources": source_hashes,
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": legacy.digest(path.read_bytes())}
                  for path in sorted(output.iterdir()) if path.is_file()},
        "split_scope": "Original six training and two validation equipment families; original admission splits.",
        "limitations": [
            "No model inference, optimizer, authenticated transport or native device execution is invoked.",
            "All source notes are declared fixture evidence signed by separate ephemeral identities on one host.",
            "An alternative reader omitting validity is a conflicting source counterexample, never an applied target.",
            "Public-only local replay cannot sign receipts or prove possession of a device private key.",
            "Measured lookup times are retained; replay compares semantic fields and does not claim equal timings.",
            "The 144 new cases share seed 3; original 320 cases retain their previous numeric variations.",
            "All equipment task IDs are opaque; original equipment IDs are retained only as evaluator metadata.",
            "Human review, experiment freeze, native resource checks and model qualification remain pending.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(build(args.output))
    print(json.dumps({"dataset_id": result["dataset_id"], "counts": result["counts"],
                      "training_admitted": False}, indent=2))
