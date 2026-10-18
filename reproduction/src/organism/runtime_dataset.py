"""Development references through the actual diagnostic/reproduction tools.

No model, transport, child process or optimizer is invoked. These deterministic
references remain separate from admitted teacher data and sealed qualification.
"""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from strands import Agent
from strands.models import LlamaCppModel
from strands.tools.registry import ToolRegistry

from .audit_runtime_dataset import inspect_record
from .capsule import canonical
from .check_template_parity import native_messages
from .diagnostic_agent import EquipmentTool, SPECS as DIAGNOSTIC_SPECS, SYSTEM_PROMPT as DIAGNOSTIC_PROMPT
from .equipment import EquipmentDirectory, EquipmentSession
from .reproduction_agent import (ReproductionSession, ReproductionTool,
                                 RAW_SPECS as REPRODUCTION_SPECS,
                                 RAW_SYSTEM_PROMPT as REPRODUCTION_PROMPT)


SCHEMA = "organism-runtime-trajectory-v1"
DATASET_ID = "runtime-development-v2"
ROLES = {
    "diagnostic": {"system_prompt": DIAGNOSTIC_PROMPT, "specs": list(DIAGNOSTIC_SPECS),
                   "final": "Assessment recorded."},
    "reproduction": {"system_prompt": REPRODUCTION_PROMPT, "specs": list(REPRODUCTION_SPECS),
                     "final": "Choice recorded."},
}
PROFILES = (
    ("a", "train", "C", 10, 40, 30),
    ("b", "train", "F", 5, 55, 60),
    ("c", "train", "K", 0, 65, 120),
    ("d", "train", "C", -20, 30, 15),
    ("e", "train", "F", 20, 80, 300),
    ("f", "train", "K", 15, 45, 90),
    ("g", "validation", "F", -10, 35, 45),
    ("h", "validation", "K", 25, 70, 180),
)
DIAGNOSTIC_VARIATIONS = ("normal", "low", "high", "lower_boundary", "upper_boundary", "stale")
ADMISSION_VARIATIONS = ("available", "occupied", "stopped", "retired", "unavailable")
SOURCES = ("runtime_dataset.py", "audit_runtime_dataset.py", "check_template_parity.py",
           "diagnostic_agent.py", "equipment.py", "reproduction_agent.py", "discovery.py")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def stock_release(root):
    # Only public signed release metadata is needed; no model payload is read.
    path = Path(root) / "artifacts/complete-stock-capsule-20261014-01/capsule/.manifest.json"
    return json.loads(path.read_bytes())


def development_cases():
    for profile, split, unit, lower, upper, maximum_age in PROFILES:
        for variation in DIAGNOSTIC_VARIATIONS:
            for seed in range(5):
                case_id = "thermal-" + digest(f"runtime-v1/{profile}/{variation}/{seed}".encode())[:20]
                values = {"normal": lower + (upper - lower) * (seed + 1) / 6,
                          "low": lower - 5 * (seed + 1), "high": upper + 5 * (seed + 1),
                          "lower_boundary": lower, "upper_boundary": upper,
                          "stale": upper + 10 * (seed + 1)}
                celsius = values[variation]
                raw = celsius if unit == "C" else (celsius * 9 / 5 + 32 if unit == "F" else celsius + 273.15)
                now = 1_700_000_000 + seed * 1000
                age = maximum_age + seed + 1 if variation == "stale" else seed + 1
                yield {
                    "case_id": case_id, "role": "diagnostic", "split": split, "profile_id": "temperature-" + profile,
                    "variation": variation, "seed": seed, "clock": now,
                    "observation": {"schema": "organism-temperature-v1", "device_id": case_id,
                                    "sensor_id": "temperature-" + profile, "unit": unit, "value": round(raw, 6),
                                    "normal_range_celsius": [lower, upper], "maximum_observation_age_seconds": maximum_age,
                                    "observed_unix_seconds": now - age, "sequence": seed + 1},
                }
        for architecture in ("x86_64", "arm64"):
            for index, variation in enumerate(ADMISSION_VARIATIONS):
                case_id = "birth-" + digest(f"runtime-v1/{profile}/{architecture}/{variation}".encode())[:20]
                owner = variation if variation in ("stopped", "retired") else "enabled"
                yield {
                    "case_id": case_id, "role": "reproduction", "split": split, "profile_id": "admission-" + profile,
                    "variation": variation, "seed": index,
                    "peer_id": digest((case_id + "/peer").encode()), "platform": "linux-" + architecture,
                    "owner_mode": owner, "available": variation == "available", "revision": 10 + index,
                }


def contracts(model):
    result = {}
    for role, definition in ROLES.items():
        tool_class = EquipmentTool if role == "diagnostic" else ReproductionTool
        # Registration adds schema descriptions. Calling the raw SPECS directly
        # does not reproduce the contract sent by the live Agent.
        agent = Agent(model=model, tools=[tool_class(None, spec) for spec in definition["specs"]],
                      system_prompt=definition["system_prompt"], callback_handler=None, retry_strategy=None,
                      context_manager=False, background_tasks=False, load_tools_from_directory=False)
        specs = deepcopy(agent.tool_registry.get_all_tool_specs())
        request = model._format_request([], specs, definition["system_prompt"])
        body = {"system_prompt": definition["system_prompt"], "tools": request["tools"],
                "strands_specs": specs, "final_acknowledgement": definition["final"]}
        result[role] = {**deepcopy(body), "sha256": digest(canonical(body))}
    return result


def provider_sources():
    from strands.tools import tools
    return {"strands_llamacpp.py": Path(inspect.getsourcefile(LlamaCppModel)),
            "strands_registry.py": Path(inspect.getsourcefile(ToolRegistry)),
            "strands_tools.py": Path(tools.__file__),
            "strands_agent.py": Path(inspect.getsourcefile(Agent))}


class FixturePeer:
    """Exact public observation schema, without claiming authenticated transport."""

    def __init__(self, case):
        self.case = case
        self.policy_id = digest((case["profile_id"] + "/synthetic-policy").encode())

    def platform(self, identity):
        if identity != self.case["peer_id"]:
            raise ValueError("Reference attempted to inspect another fixture peer")
        return self.case["platform"]

    def endpoint(self, identity):
        self.platform(identity)
        return identity, "127.0.0.1", 9

    async def observe(self, endpoint):
        if endpoint != self.endpoint(self.case["peer_id"]):
            raise ValueError("Reference changed its fixture endpoint")
        value = {"device_id": self.case["peer_id"], "policy_id": self.policy_id,
                 **{key: self.case[key] for key in ("platform", "owner_mode", "available", "revision")},
                 "observation_valid_seconds": 30}
        return {**value, "admission_ticket": "development-fixture-with-no-admission-authority",
                "transport_receipt": {"sequence": 1, "digest": digest(canonical(value))}}


async def execute_case(case, model, contract, release, *, supplied_actions=None, final=None):
    role = case["role"]
    if role not in ROLES:
        raise ValueError("Unsupported runtime-data role")
    with tempfile.TemporaryDirectory(prefix="organism-runtime-data-") as temporary:
        root = Path(temporary)
        if role == "diagnostic":
            raw = canonical(case["observation"])
            (root / (case["case_id"] + ".json")).write_bytes(raw)
            session = EquipmentSession(EquipmentDirectory(root, clock=lambda: case["clock"]), case["case_id"])
            tools = {spec["name"]: EquipmentTool(session, spec) for spec in DIAGNOSTIC_SPECS}
            user = "Assess the device " + case["case_id"] + "."
        else:
            client = FixturePeer(case)
            book = SimpleNamespace(policy=client, endpoint=client.endpoint)
            session = ReproductionSession(client, book, case["peer_id"], release)
            tools = {spec["name"]: ReproductionTool(session, spec) for spec in REPRODUCTION_SPECS}
            user = "Consider discovered device " + case["peer_id"] + "."
        messages = [{"role": "user", "content": [{"text": user}]}]
        actions = []

        async def invoke(name, arguments):
            if name not in tools:
                raise ValueError("Unknown reference tool")
            call = {"name": name, "input": deepcopy(arguments),
                    "toolUseId": "data-" + case["case_id"] + "-" + str(len(actions))}
            messages.append({"role": "assistant", "content": [{"toolUse": call}]})
            events = [event async for event in tools[name].stream(call, {})]
            if len(events) != 1 or events[0].tool_result["status"] != "success":
                raise ValueError("Reference tool did not produce its bounded successful result")
            result = events[0].tool_result
            messages.append({"role": "user", "content": [{"toolResult": deepcopy(result)}]})
            actions.append({"tool": name, "arguments": deepcopy(arguments)})
            return json.loads(result["content"][0]["text"])

        if supplied_actions is not None:
            for action in supplied_actions:
                await invoke(action["tool"], action["arguments"])
        elif role == "diagnostic":
            capability = await invoke("inspect_device", {"device_id": case["case_id"]})
            reading = await invoke("read_measurement", {"device_id": case["case_id"],
                                                       "sensor_id": capability["sensors"][0]["sensor_id"]})
            sensor = capability["sensors"][0]
            if reading["age_seconds"] > sensor["maximum_observation_age_seconds"]:
                status, temperature = "unknown", None
            else:
                value = reading["value"]
                temperature = value if reading["unit"] == "C" else ((value - 32) / 1.8 if reading["unit"] == "F" else value - 273.15)
                temperature = round(temperature, 6)
                # llama.cpp renders integral numeric tool arguments as integers
                # in subsequent history. Teach the same valid JSON spelling.
                if temperature == int(temperature):
                    temperature = int(temperature)
                low, high = sensor["normal_range_celsius"]
                status = "low" if temperature < low else ("high" if temperature > high else "normal")
            await invoke("submit_assessment", {"device_id": case["case_id"], "status": status,
                                               "temperature_celsius": temperature,
                                               "evidence_ids": [capability["evidence_id"], reading["evidence_id"]]})
        else:
            observed = await invoke("inspect_candidate", {"device_id": case["peer_id"]})
            establish = (observed["owner_mode"] == "enabled" and observed["available"]
                         and observed["platform"] in observed["supported_platforms"])
            await invoke("choose_child_action", {"action": "establish" if establish else "defer",
                                                 "platform": observed["platform"] if establish else None,
                                                 "evidence_id": observed["evidence_id"]})
        final = contract["final_acknowledgement"] if final is None else final
        messages.append({"role": "assistant", "content": [{"text": final}]})
        request = model._format_request(messages, contract["strands_specs"], contract["system_prompt"])
        if request["tools"] != contract["tools"]:
            raise ValueError("Provider formatting changed the recorded tool contract")
        return {"actions": actions, "events": deepcopy(session.events), "strands_messages": messages,
                "wire_messages": request["messages"], "messages": native_messages(request["messages"]),
                "submission": deepcopy(session.submission if role == "diagnostic" else session.choice)}


async def trajectory(case, model, contract, release):
    result = await execute_case(case, model, contract, release)
    record = {"schema": SCHEMA, "case_id": case["case_id"], "role": case["role"], "split": case["split"],
              "profile_id": case["profile_id"], "case": deepcopy(case), "tool_contract_sha256": contract["sha256"],
              "provenance": {"kind": "locally_authored_deterministic_runtime_tool_reference",
                             "teacher_model_used": False, "human_review_claimed": False,
                             "real_peer_transport_used": False, "native_process_used": False,
                             "training_admitted": False, "qualification_data_present": False},
              **result}
    record["evaluation"] = inspect_record(record, contract, release)
    return record


async def replay(record, model, contract, release):
    actual = await execute_case(record["case"], model, contract, release,
                                supplied_actions=record["actions"], final=record["messages"][-1]["content"])
    for name, value in actual.items():
        if record[name] != value:
            raise ValueError("Runtime trajectory differs from actual tool replay: " + name)
    evaluation = inspect_record(record, contract, release)
    if evaluation != record["evaluation"]:
        raise ValueError("Runtime trajectory evaluation changed")
    return evaluation


async def build(output):
    if version("strands-agents") != "1.57.1":
        raise ValueError("Runtime references require the pinned Strands provider")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    release = stock_release(root)
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
        counts[split] = {"trajectories": len(records), "profiles": sorted({record["profile_id"] for record in records}),
                         "roles": {role: sum(record["role"] == role for record in records) for role in ROLES},
                         "assistant_targets": sum(sum(message["role"] == "assistant" for message in record["messages"])
                                                  for record in records)}
    for name, value in (("contracts.json", definitions), ("release.json", release)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    snapshots = output / "sources"
    snapshots.mkdir()
    source_hashes = {}
    for name in SOURCES:
        raw = (root / name).read_bytes()
        (snapshots / name).write_bytes(raw)
        source_hashes[name] = digest(raw)
    for name, provider in provider_sources().items():
        (snapshots / name).write_bytes(provider.read_bytes())
        source_hashes[name] = digest(provider.read_bytes())
    manifest = {
        "dataset_id": DATASET_ID, "created_at": datetime.now(timezone.utc).isoformat(),
        "schema": SCHEMA, "scope": "development_references_through_current_runtime_tools",
        "training_admitted": False, "optimization_started": False, "qualification_data_present": False,
        "counts": counts, "strands_version": version("strands-agents"), "sources": source_hashes,
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": digest(path.read_bytes())}
                  for path in sorted(output.iterdir()) if path.is_file()},
        "split_scope": "Disjoint parameter profiles inside the existing APIs; validation is development material.",
        "limitations": ["No model, peer transport, native process, child establishment or optimizer is invoked.",
                        "The deterministic reference solves these protocol cases; model benefit is not established.",
                        "Profiles share the same temperature and admission APIs; they are not withheld firmware/API families.",
                        "Repeated boundary values and related profiles do not constitute independent physical environments.",
                        "Teacher/human provenance review and the full experiment freeze are still required for training admission."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(build(args.output))["counts"], indent=2))


if __name__ == "__main__":
    main()
