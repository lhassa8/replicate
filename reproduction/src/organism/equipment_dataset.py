"""Mixed development references through current equipment and admission tools.

The reference policy sees published equipment data. Private fixture truths
are used only by the independent inspector. No model or optimizer is invoked.
"""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

from strands import Agent
from strands.models import LlamaCppModel

from .audit_runtime_dataset import require
from .capsule import canonical
from .check_template_parity import native_messages
from .diagnostic_agent import EquipmentTool
from .equipment_baseline import solve
from .equipment_development import case as base_case
from .equipment_diagnostic_agent import SPECS, SYSTEM_PROMPT
from .equipment_interfaces import InterfaceDirectory, InterfaceSession
from . import runtime_dataset as legacy
from .runtime_rows import EQUIPMENT_SCHEMA, MIXED_SCHEMA, inspect_record, targets


DATASET_ID = "equipment-runtime-development-v1"
VARIATIONS = ("nominal", "fault", "scaled_fault", "stale", "invalid", "missing")
SEEDS = range(5)
# Explicit raw/canonical pairs are fixture truth, never given to the policy.
VALUES = {
    "a": {"nominal": [(12, 12), (18, 18), (24, 24), (34, 34), (39, 39)],
          "fault": [(41, 41), (55, 55), (65, 65), (80, 80), (100, 100)]},
    "b": {"nominal": [(59, 15), (68, 20), (77, 25), (95, 35), (104, 40)],
          "fault": [(113, 45), (122, 50), (140, 60), (176, 80), (194, 90)]},
    "c": {"nominal": [(0.5, 0.5), (1, 1), (2, 2), (3, 3), (4.5, 4.5)],
          "fault": [(4.6, 4.6), (5, 5), (6, 6), (8, 8), (12, 12)]},
    "d": {"nominal": [(0.02, 0.508), (0.04, 1.016), (0.06, 1.524), (0.1, 2.54), (0.15, 3.81)],
          "fault": [(0.2, 5.08), (0.25, 6.35), (0.3, 7.62), (0.35, 8.89), (0.5, 12.7)]},
    "e": {"nominal": [(0, 0), (0.005, 0.5), (0.01, 1), (0.03, 3), (0.05, 5)],
          "fault": [(0.06, 6), (0.08, 8), (0.1, 10), (0.12, 12), (0.3, 30)]},
    "f": {"nominal": [(0, 0), (1, 1), (2, 2), (0, 0), (2, 2)],
          "fault": [(3, 3), (4, 4), (5, 5), (9, 9), (20, 20)]},
    "g": {"nominal": [(283.15, 10), (288.15, 15), (293.15, 20), (308.15, 35), (312.15, 39)],
          "fault": [(318.15, 45), (328.15, 55), (338.15, 65), (353.15, 80), (373.15, 100)]},
    "h": {"nominal": [(0, 0), (1, 1), (2, 2), (4, 4), (5, 5)],
          "fault": [(6, 6), (8, 8), (10, 10), (15, 15), (30, 30)]},
}
SOURCES = (
    "equipment_dataset.py", "audit_equipment_dataset.py", "runtime_rows.py",
    "equipment_development.py", "equipment_interfaces.py", "equipment_baseline.py",
    "equipment_diagnostic_agent.py", "audit_equipment_diagnostics.py", "audit_application.py",
    "diagnostic_agent.py", "equipment.py", "capsule.py", "runtime_dataset.py",
    "audit_runtime_dataset.py", "reproduction_agent.py", "discovery.py", "check_template_parity.py",
)


def number_spelling(value):
    value = round(value, 6)
    return int(value) if value == int(value) else value


def assign(source, selector, value):
    if source["format"] == "register_map":
        bank, address = selector.split(":")
        source["payload"][bank][address] = value
    else:
        parts = [part.replace("~1", "/").replace("~0", "~") for part in selector[1:].split("/")]
        current = source["payload"]
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1]] = value


def equipment_cases():
    for profile in VALUES:
        for variation in VARIATIONS:
            for seed in SEEDS:
                moment = 1700000000 + seed * 1000
                fixture = base_case(profile, "fault" if variation == "scaled_fault" else variation, moment)
                identity = f"equipment-data-{profile}-{variation}-{seed}"
                fixture["case_id"] = fixture["document"]["device_id"] = identity
                fixture["expected_reader"]["device_id"] = identity
                ordinary = variation == "nominal"
                raw, expected = VALUES[profile]["nominal" if ordinary else "fault"][seed]
                source = fixture["document"]["sources"][0]
                if variation != "missing":
                    if variation == "scaled_fault":
                        source["fields"][0]["scale"] = 0.01
                        raw = number_spelling(float(Decimal(str(raw)) / Decimal("0.01")))
                    assign(source, fixture["expected_reader"]["value_selectors"][0], raw)
                fixture["document"]["sequence"] = seed + 1
                fixture["variation"] = variation
                fixture["expected_value"] = None if variation in ("stale", "invalid", "missing") else expected
                fixture["expected_status"] = "unknown" if fixture["expected_value"] is None else "normal" if ordinary else "high"
                yield {
                    "case_id": identity, "role": "equipment",
                    "split": "train" if fixture["split"] == "development_train" else "validation",
                    "profile_id": fixture["profile"], "family_id": fixture["document"]["family"],
                    "variation": variation, "seed": seed, "clock": moment, "fixture": fixture,
                    "inject_configuration_error": variation == "nominal" and seed == 4,
                }


def development_cases():
    yield from equipment_cases()
    yield from (case for case in legacy.development_cases() if case["role"] == "reproduction")


def contracts(model):
    reproduction = legacy.contracts(model)["reproduction"]
    agent = Agent(model=model, tools=[EquipmentTool(None, spec) for spec in SPECS],
                  system_prompt=SYSTEM_PROMPT, callback_handler=None, retry_strategy=None,
                  context_manager=False, background_tasks=False, load_tools_from_directory=False)
    specs = deepcopy(agent.tool_registry.get_all_tool_specs())
    request = model._format_request([], specs, SYSTEM_PROMPT)
    body = {"system_prompt": SYSTEM_PROMPT, "tools": request["tools"], "strands_specs": specs,
            "final_acknowledgement": "Diagnosis recorded."}
    return {"equipment": {**body, "sha256": legacy.digest(canonical(body))}, "reproduction": reproduction}


async def execute_case(case, model, contract, release, *, supplied_actions=None, final=None):
    if case["role"] == "reproduction":
        return await legacy.execute_case(case, model, contract, release, supplied_actions=supplied_actions, final=final)
    with tempfile.TemporaryDirectory(prefix="equipment-data-") as temporary:
        root = Path(temporary)
        raw = canonical(case["fixture"]["document"])
        path = root / (case["case_id"] + ".json")
        path.write_bytes(raw)
        directory = InterfaceDirectory(root, clock=lambda: case["clock"])
        if supplied_actions is None:
            planning = InterfaceSession(directory, case["case_id"])
            solve(planning, case["case_id"])
            planned = [{"tool": event["tool"], "arguments": deepcopy(event["arguments"])} for event in planning.events]
            if planned[-1]["arguments"]["value"] is not None:
                planned[-1]["arguments"]["value"] = number_spelling(planned[-1]["arguments"]["value"])
            if case["inject_configuration_error"]:
                invalid = deepcopy(planned[1])
                invalid["arguments"]["value_selectors"] = invalid["arguments"]["value_selectors"][0]
                planned.insert(1, invalid)
        else:
            planned = supplied_actions
        session = InterfaceSession(directory, case["case_id"])
        tools = {spec["name"]: EquipmentTool(session, spec) for spec in SPECS}
        messages = [{"role": "user", "content": [{"text": "Diagnose the equipment " + case["case_id"] + "."}]}]
        actions = []
        for action in planned:
            name, args = action["tool"], action["arguments"]
            require(name in tools, "Unknown equipment reference action")
            call = {"name": name, "input": deepcopy(args), "toolUseId": "data-" + case["case_id"] + "-" + str(len(actions))}
            messages.append({"role": "assistant", "content": [{"toolUse": call}]})
            events = [event async for event in tools[name].stream(call, {})]
            require(len(events) == 1 and events[0].tool_result["status"] == "success", "Equipment tool wrapper failed")
            messages.append({"role": "user", "content": [{"toolResult": deepcopy(events[0].tool_result)}]})
            actions.append({"tool": name, "arguments": deepcopy(args)})
        acknowledgement = contract["final_acknowledgement"] if final is None else final
        messages.append({"role": "assistant", "content": [{"text": acknowledgement}]})
        request = model._format_request(messages, contract["strands_specs"], contract["system_prompt"])
        require(request["tools"] == contract["tools"] and path.read_bytes() == raw, "Tool contract or read-only input changed")
        return {"actions": actions, "events": deepcopy(session.events), "strands_messages": messages,
                "wire_messages": request["messages"], "messages": native_messages(request["messages"]),
                "submission": deepcopy(session.submission)}


async def trajectory(case, model, contract, release):
    if case["role"] == "reproduction":
        return await legacy.trajectory(case, model, contract, release)
    actual = await execute_case(case, model, contract, release)
    mask = [True] * (len(actual["events"]) + 1)
    if case["inject_configuration_error"]:
        mask[1] = False
    record = {
        "schema": EQUIPMENT_SCHEMA, "case_id": case["case_id"], "role": "equipment",
        "split": case["split"], "profile_id": case["profile_id"], "case": deepcopy(case),
        "tool_contract_sha256": contract["sha256"], "supervised_assistants": mask,
        "provenance": {"kind": "locally_authored_runtime_tool_reference_with_declared_error_context",
                       "teacher_model_used": False, "human_review_claimed": False,
                       "real_peer_transport_used": False, "native_process_used": False,
                       "training_admitted": False, "qualification_data_present": False},
        **actual,
    }
    record["evaluation"] = inspect_record(record, contract, release)
    return record


async def replay(record, model, contract, release):
    actual = await execute_case(record["case"], model, contract, release, supplied_actions=record["actions"],
                                final=record["messages"][-1]["content"])
    require(all(record[name] == value for name, value in actual.items()), "Equipment/admission replay differs")
    checked = inspect_record(record, contract, release)
    require(checked == record["evaluation"], "Reference semantics changed")
    return checked


async def build(output):
    require(version("strands-agents") == "1.57.1", "Pinned Strands version required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    release = legacy.stock_release(root)
    model = LlamaCppModel(base_url="http://127.0.0.1:9", model_id="reference-formatting-only",
                         params={"temperature": 0, "max_tokens": 192})
    groups = {"train": [], "validation": []}
    try:
        definitions = contracts(model)
        for case in development_cases():
            record = await trajectory(case, model, definitions[case["role"]], release)
            await replay(record, model, definitions[case["role"]], release)
            groups[case["split"]].append(record)
    finally:
        await model.client.aclose()
    counts = {}
    for split, records in groups.items():
        with (output / (split + ".jsonl")).open("x") as target:
            for record in records:
                target.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        counts[split] = {
            "trajectories": len(records), "profiles": sorted({record["profile_id"] for record in records}),
            "roles": {role: sum(record["role"] == role for record in records) for role in definitions},
            "assistant_targets": sum(len(list(targets(record))) for record in records),
            "context_only_injected_actions": sum(record["role"] == "equipment"
                                               and record["case"]["inject_configuration_error"] for record in records),
        }
    for name, value in (("contracts.json", definitions), ("release.json", release)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    snapshots = output / "sources"
    snapshots.mkdir()
    sources = {name: root / name for name in SOURCES}
    sources.update(legacy.provider_sources())
    source_hashes = {}
    for name, path in sources.items():
        raw = path.read_bytes()
        (snapshots / name).write_bytes(raw)
        source_hashes[name] = legacy.digest(raw)
    manifest = {
        "dataset_id": DATASET_ID, "created_at": datetime.now(timezone.utc).isoformat(), "schema": MIXED_SCHEMA,
        "scope": "development_references_through_current_runtime_tools",
        "training_admitted": False, "optimization_started": False, "qualification_data_present": False,
        "human_review_claimed": False, "counts": counts, "strands_version": version("strands-agents"),
        "sources": source_hashes,
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": legacy.digest(path.read_bytes())}
                  for path in sorted(output.iterdir()) if path.is_file()},
        "split_scope": "Six development equipment families train, two development families validate; retained admission profiles keep their previous split.",
        "limitations": [
            "No model, authenticated peer transport, native process, child establishment or optimizer is invoked.",
            "All references are authored policy examples, not human-reviewed or teacher-model demonstrations.",
            "Eight injected invalid configuration calls are context only and must never be supervised targets.",
            "The fixed policy solves this development corpus; no model advantage or blind qualification is established.",
            "Related values, timestamps and profiles do not constitute independent physical environments.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(build(args.output))["counts"], indent=2))
