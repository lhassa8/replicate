"""Replay a narrow, previously admitted development subset with current tools.

No new observations or answers are derived from the frozen twenty-case suite.
This prepares candidate references only; no model inference or optimizer runs.
"""

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from strands import Agent
from strands.models import LlamaCppModel

from .audit_runtime_dataset import canonical, digest, require
from .check_template_parity import native_messages
from .diagnostic_agent import EquipmentTool
from .equipment_baseline import solve
from .equipment_dataset import number_spelling
from .equipment_diagnostic_agent import SPECS as EQUIPMENT_SPECS, SYSTEM_PROMPT as EQUIPMENT_PROMPT
from .equipment_interfaces import InterfaceDirectory, InterfaceSession
from .export_task_inputs import identity
from .reproduction_agent import ReproductionSession, ReproductionTool, SPECS as REPRODUCTION_SPECS
from .reproduction_agent import SYSTEM_PROMPT as REPRODUCTION_PROMPT
from .reproduction_protocol import PROFILE as REPRODUCTION_PROFILE, NO_PLATFORM
from .runtime_dataset import FixturePeer, execute_case as legacy_execute, contracts as legacy_contracts
from .temperature_measurement import PROFILE as TEMPERATURE_PROFILE


SCHEMA = "organism-first-lineage-runtime-trajectory-v1"
CORPUS_SCHEMA = "organism-first-lineage-runtime-corpus-v1"
DATASET_ID = "first-lineage-runtime-development-v6-01"
PARENT_RECIPE = "organism/training-recipe.current-v5.refinement-01.draft.json"
PARENT_FILES = {
    "train": ("organism/data/equipment-runtime-development-v5-refinement-01/train.jsonl",
              "db3183fd07ab30d5d6d58e3e7cbc2b1069444f4104868b3d61708aff05e30a82"),
    "validation": ("organism/data/equipment-runtime-development-v5-refinement-01/validation.jsonl",
                   "624f75edd24b9855a09527324c3190545f0b6d70f0a094344fb7d29f5ed687ac"),
}
PROVENANCE = {
    "kind": "previously_admitted_development_observations_replayed_through_versioned_tools",
    "teacher_model_used": False, "model_inference_used": False, "human_review_claimed": False,
    "real_peer_transport_used": False, "native_process_used": False,
    "training_admitted": False, "qualification_data_present": False,
}
SOURCES = (
    "first_lineage_dataset.py", "audit_first_lineage_dataset.py", "runtime_rows.py",
    "equipment_dataset.py", "equipment_interfaces.py", "equipment_baseline.py",
    "equipment_diagnostic_agent.py", "temperature_measurement.py", "audit_equipment_diagnostics.py",
    "diagnostic_agent.py", "equipment.py", "runtime_dataset.py", "audit_runtime_dataset.py",
    "reproduction_agent.py", "reproduction_protocol.py", "discovery.py", "check_template_parity.py",
    "capsule.py", "native_target_codec.py",
)


def selected_parents(root, split):
    path, expected = PARENT_FILES[split]
    source = Path(root) / path
    require(identity(source)["sha256"] == expected, "Admitted parent observations changed")
    records = [json.loads(line) for line in source.read_text().splitlines()]
    return [record for record in records if record["role"] in ("diagnostic", "reproduction")
            or (record["role"] == "equipment"
                and record["case"]["fixture"]["document"]["task"]["quantity"] == "temperature")]


def contracts(model):
    result = {"diagnostic": legacy_contracts(model)["diagnostic"]}
    for role, tool_class, specs, prompt, final, profile in (
        ("equipment", EquipmentTool, EQUIPMENT_SPECS, EQUIPMENT_PROMPT,
         "Diagnosis recorded.", TEMPERATURE_PROFILE),
        ("reproduction", ReproductionTool, REPRODUCTION_SPECS, REPRODUCTION_PROMPT,
         "Choice recorded.", REPRODUCTION_PROFILE),
    ):
        agent = Agent(model=model, tools=[tool_class(None, spec) for spec in specs],
                      system_prompt=prompt, callback_handler=None, retry_strategy=None,
                      context_manager=False, background_tasks=False, load_tools_from_directory=False)
        registered = deepcopy(agent.tool_registry.get_all_tool_specs())
        request = model._format_request([], registered, prompt)
        body = {"system_prompt": prompt, "tools": request["tools"], "strands_specs": registered,
                "final_acknowledgement": final, "tool_profile": profile}
        result[role] = {**body, "sha256": digest(canonical(body))}
    return result


async def execute_case(case, model, contract, release, *, supplied_actions=None):
    if case["role"] == "diagnostic":
        return await legacy_execute(case, model, contract, release, supplied_actions=supplied_actions)
    with tempfile.TemporaryDirectory(prefix="first-lineage-reference-") as temporary:
        if case["role"] == "equipment":
            raw = canonical(case["fixture"]["document"])
            (Path(temporary) / (case["case_id"] + ".json")).write_bytes(raw)
            directory = InterfaceDirectory(Path(temporary), clock=lambda: case["clock"])
            session = InterfaceSession(directory, case["case_id"], canonical_temperature=True)
            tools = {spec["name"]: EquipmentTool(session, spec) for spec in EQUIPMENT_SPECS}
            user = "Diagnose the equipment " + case["case_id"] + "."
        else:
            require(case["role"] == "reproduction", "Unknown first-lineage role")
            peer = FixturePeer(case)
            book = SimpleNamespace(policy=peer, endpoint=peer.endpoint)
            session = ReproductionSession(peer, book, case["peer_id"], release,
                                          tool_profile=REPRODUCTION_PROFILE)
            tools = {spec["name"]: ReproductionTool(session, spec) for spec in REPRODUCTION_SPECS}
            user = "Consider discovered device " + case["peer_id"] + "."
        messages = [{"role": "user", "content": [{"text": user}]}]
        actions = []

        async def invoke(name, arguments):
            require(name in tools, "Unknown reference tool")
            call = {"name": name, "input": deepcopy(arguments),
                    "toolUseId": "data-" + case["case_id"] + "-" + str(len(actions))}
            messages.append({"role": "assistant", "content": [{"toolUse": call}]})
            results = [event async for event in tools[name].stream(call, {})]
            require(len(results) == 1 and results[0].tool_result["status"] == "success",
                    "Reference tool failed")
            result = deepcopy(results[0].tool_result)
            messages.append({"role": "user", "content": [{"toolResult": result}]})
            actions.append({"tool": name, "arguments": deepcopy(arguments)})
            return json.loads(result["content"][0]["text"])

        if supplied_actions is not None:
            for action in supplied_actions:
                await invoke(action["tool"], action["arguments"])
        elif case["role"] == "equipment":
            planning = InterfaceSession(directory, case["case_id"], canonical_temperature=True)
            solve(planning, case["case_id"])
            planned = [{"tool": item["tool"], "arguments": deepcopy(item["arguments"])}
                       for item in planning.events]
            if planned[-1]["arguments"]["value"] is not None:
                planned[-1]["arguments"]["value"] = number_spelling(planned[-1]["arguments"]["value"])
            if case["inject_configuration_error"]:
                invalid = deepcopy(planned[1])
                invalid["arguments"]["value_selectors"] = invalid["arguments"]["value_selectors"][0]
                planned.insert(1, invalid)
            for action in planned:
                await invoke(action["tool"], action["arguments"])
        else:
            observed = await invoke("inspect_candidate", {"device_id": case["peer_id"]})
            establish = (observed["owner_mode"] == "enabled" and observed["available"]
                         and observed["platform"] in observed["supported_platforms"])
            await invoke("choose_child_action", {
                "action": "establish" if establish else "defer",
                "platform": observed["platform"] if establish else NO_PLATFORM,
                "evidence_id": observed["evidence_id"],
            })
        messages.append({"role": "assistant", "content": [{"text": contract["final_acknowledgement"]}]})
        request = model._format_request(messages, contract["strands_specs"], contract["system_prompt"])
        require(request["tools"] == contract["tools"], "Current provider changed the recorded tools")
        return {"actions": actions, "events": deepcopy(session.events),
                "strands_messages": messages, "wire_messages": request["messages"],
                "messages": native_messages(request["messages"]),
                "submission": deepcopy(session.submission if case["role"] == "equipment" else session.choice)}


async def build(root, output):
    from .audit_first_lineage_dataset import inspect_record, audit

    root, output = Path(root).resolve(), Path(output).resolve()
    require(version("strands-agents") == "1.57.1", "Pinned Strands version required")
    recipe = json.loads((root / PARENT_RECIPE).read_bytes())
    release_item = recipe["inputs"]["release"]
    release_path = root / release_item["path"]
    require(identity(release_path) == {k: release_item[k] for k in ("bytes", "sha256")},
            "Parent public release metadata changed")
    release = json.loads(release_path.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    model = LlamaCppModel(base_url="http://127.0.0.1:9", model_id="reference-formatting-only",
                         params={"temperature": 0, "max_tokens": 192})
    counts = {}
    try:
        definitions = contracts(model)
        for name, value in (("contracts.json", definitions),):
            (output / name).write_text(json.dumps(value, indent=2) + "\n")
        (output / "release.json").write_bytes(release_path.read_bytes())
        for split in ("train", "validation"):
            roles, target_count = Counter(), 0
            with (output / (split + ".jsonl")).open("x") as stream:
                for parent in selected_parents(root, split):
                    case = deepcopy(parent["case"])
                    role = case["role"]
                    actual = await execute_case(case, model, definitions[role], release)
                    mask = [True] * (len(actual["events"]) + 1)
                    if role == "equipment" and case["inject_configuration_error"]:
                        mask[1] = False
                    record = {
                        "schema": SCHEMA, **{key: parent[key] for key in
                                            ("case_id", "role", "split", "profile_id")},
                        "case": case, "parent_record_sha256": digest(canonical(parent)),
                        "tool_contract_sha256": definitions[role]["sha256"],
                        "supervised_assistants": mask, "provenance": PROVENANCE, **actual,
                    }
                    record["evaluation"] = inspect_record(record, definitions[role], release)
                    replay = await execute_case(case, model, definitions[role], release,
                                                supplied_actions=record["actions"])
                    require(replay == actual, "Actual tool replay differs from the reference")
                    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                    roles[role] += 1
                    target_count += sum(mask)
            counts[split] = {"trajectories": sum(roles.values()), "assistant_targets": target_count,
                             "roles": dict(roles)}
    finally:
        await model.client.aclose()
    snapshots = output / "sources"
    snapshots.mkdir()
    sources = {}
    for name in SOURCES:
        raw = (root / "organism" / name).read_bytes()
        (snapshots / name).write_bytes(raw)
        sources[name] = digest(raw)
    manifest = {
        "schema": CORPUS_SCHEMA, "dataset_id": DATASET_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "development_references_through_current_runtime_tools",
        "files": {name: identity(output / name) for name in
                  ("train.jsonl", "validation.jsonl", "contracts.json", "release.json")},
        "counts": counts, "sources": sources, "strands_version": version("strands-agents"),
        "parent_recipe": {"path": PARENT_RECIPE, **identity(root / PARENT_RECIPE)},
        "target_codec": "granite-json-scalar-targets-v1",
        "training_admitted": False, "optimization_started": False,
        "qualification_data_present": False, "human_review_claimed": False,
        "teacher_model_used": False, "model_generations": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = await audit(output)
    return manifest, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    manifest, result = asyncio.run(build(args.root, args.output))
    args.audit_output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"counts": manifest["counts"], "audit": result}, indent=2))
