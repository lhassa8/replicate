"""Render candidate prompts on the owned Mac server, without generating tokens."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
import uuid

import httpx

from .audit_runtime_dataset import require
from .runtime_rows import audit, INHERITED_CORPUS_SCHEMA, MIXED_SCHEMA, STARTUP_CORPUS_SCHEMA, targets
from .check_template_parity import native_messages
from .inference_server import InferenceServer, RuntimeConfig, digest
from .runtime_dataset import contracts
from strands.models import LlamaCppModel


async def retained_contracts(root, *, equipment=False, inherited=False, startup=False):
    """Compare retained structured requests; argument string order is reported."""
    model = LlamaCppModel(base_url="http://127.0.0.1:9", model_id="reference-formatting-only")
    receipts = []
    try:
        if equipment or inherited or startup:
            from .equipment_dataset import contracts as equipment_contracts
            definitions = equipment_contracts(model)
            selections = [("equipment", "equipment-interface-preflight-20261014-02/" + name + ".json")
                          for name in ("equipment-a-nominal", "equipment-b-fault", "equipment-c-fault",
                                       "equipment-d-fault", "equipment-e-fault", "equipment-f-fault",
                                       "equipment-g-stale", "equipment-h-invalid")]
            if inherited or startup:
                from .inheritance_dataset import contracts as inherited_contracts
                definitions = inherited_contracts(model)
                selections += [("equipment_inherited", "inherited-diagnostic-live-20261014-01/" + name + ".json")
                               for name in ("equipment-a-nominal", "equipment-e-fault")]
            if startup:
                definitions.update({"diagnostic": contracts(model)["diagnostic"]})
                selections.append(("diagnostic", "application-preflight-20261014-03/dev-normal-k.json"))
        else:
            definitions = contracts(model)
            selections = [("diagnostic", "application-preflight-20261014-03/dev-normal-k.json")]
        selections += [
            ("reproduction", "reproduction-decision-preflight-20261014-01/available.json"),
            ("reproduction", "reproduction-decision-preflight-20261014-01/retired.json"),
        ]
        for role, relative in selections:
            path = root / "organism/evidence" / relative
            trace = json.loads(path.read_bytes())["job"]["record"]["trace"]
            require(len(trace["model_calls"]) == len(trace["http_requests"]) > 0, "Incomplete retained call capture")
            for index, (call, http) in enumerate(zip(trace["model_calls"], trace["http_requests"])):
                formatted = model._format_request(call["messages"], call["tools"], call["system_prompt"])
                body = http["body"]
                require(definitions[role]["tools"] == body["tools"] == formatted["tools"]
                        and definitions[role]["system_prompt"] == call["system_prompt"], "Live tool contract differs")
                require(native_messages(formatted["messages"]) == native_messages(body["messages"]),
                        "Retained structured message content differs")
                receipts.append({"source": relative, "sha256": digest(path), "call": index, "role": role,
                                 "registered_tool_contract_equal": True, "structured_messages_equal": True,
                                 "literal_formatted_messages_equal": formatted["messages"] == body["messages"]})
    finally:
        await model.client.aclose()
    return receipts


async def run(root, source, output, *, runtime_config=None):
    require(platform.system() == "Darwin", "This is a local Mac formatting probe")
    checked = await audit(source)
    manifest = json.loads((source / "manifest.json").read_bytes())
    first_lineage = manifest["schema"] == "organism-first-lineage-runtime-corpus-v1"
    # The v6 tool protocols are deliberately different from the historical raw
    # traces. Its current registered contracts are checked by the replay audit;
    # the actual new requests are rendered and tokenized below. This does not
    # claim a live model has already succeeded with the new reproduction tool.
    live = [] if first_lineage else await retained_contracts(
        root, equipment=manifest["schema"] == MIXED_SCHEMA,
        inherited=manifest["schema"] == INHERITED_CORPUS_SCHEMA,
        startup=manifest["schema"] == STARTUP_CORPUS_SCHEMA)
    output.mkdir(parents=True, exist_ok=False)
    experiment = json.loads((root / "organism/experiment-contract.draft.json").read_bytes())
    config = runtime_config or RuntimeConfig(
        root / "models/granite-3b/granite-base-Q4_K_M.gguf", experiment["model"]["local_export_sha256"],
        experiment["model"]["local_export_bytes"], root / "tools/llama.cpp/b9850/llama-b9850/llama-server",
        experiment["runtime"]["development_binary_sha256"], uuid.uuid4().hex, "d" * 64, uuid.uuid4().hex,
        "e" * 64, False, False)
    require(isinstance(config, RuntimeConfig) and config.native is False
            and config.trained_for_organism is False, "Formatting requires an explicit local stock runtime")
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "scope": "Mac candidate-prompt rendering only",
              "source_manifest_sha256": checked["source_manifest_sha256"], "source": str(source.resolve()),
              "runtime_audit": checked, "retained_live_contract_comparison": live, "requests": 0, "failure": None,
              "model_sha256": config.model_sha256, "server_sha256": config.executable_sha256,
              "model_path": str(config.model), "model_bytes": config.model_bytes,
              "model_generations": 0, "native_execution_verified": False, "training_admitted": False,
              "qualification_passed": False}
    if first_lineage:
        report["versioned_contract_scope"] = (
            "Current registered tools verified by full runtime reference replay; "
            "new reproduction protocol has no live model-success claim.")
    snapshot = output / "sources"
    snapshot.mkdir()
    report["sources"] = {}
    for name in ("probe_runtime_templates.py", "inference_server.py", "runtime_environment.py",
                 "check_template_parity.py", "runtime_rows.py"):
        path = root / "organism" / name
        (snapshot / name).write_bytes(path.read_bytes())
        report["sources"][name] = digest(path)
    server = InferenceServer(config, output)
    started = time.monotonic()
    try:
        async with asyncio.timeout(300):
            report["server_start"] = await server.start()
            definitions = json.loads((source / "contracts.json").read_bytes())
            async with httpx.AsyncClient(base_url=server.base_url, headers=server.headers, trust_env=False,
                                         follow_redirects=False, timeout=20) as client:
                with (output / "server-prompts.jsonl").open("x") as destination:
                    for split in ("train", "validation"):
                        for line in (source / (split + ".jsonl")).read_text().splitlines():
                            record = json.loads(line)
                            for step, index, message in targets(record):
                                body = {"messages": record["wire_messages"][:index],
                                        "tools": definitions[record["role"]]["tools"]}
                                response = await client.post("/apply-template", json=body)
                                response.raise_for_status()
                                prompt = response.json()["prompt"]
                                response = await client.post("/tokenize", json={
                                    "content": prompt, "add_special": False, "parse_special": True})
                                response.raise_for_status()
                                entry = {"case_id": record["case_id"], "role": record["role"], "split": split,
                                         "decision_index": step, "request": body, "prompt": prompt,
                                         "tokens": response.json()["tokens"]}
                                destination.write(json.dumps(entry, separators=(",", ":")) + "\n")
                                report["requests"] += 1
                            if report["requests"] % 100 == 0:
                                print(json.dumps({"rendered": report["requests"]}), flush=True)
            require(report["requests"] == sum(item["assistant_targets"] for item in checked["counts"].values()),
                    "Missing candidate prompt renderings")
            report["server_prompts_sha256"] = digest(output / "server-prompts.jsonl")
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        try:
            report["shutdown"] = await server.stop()
        except Exception as exc:
            report["shutdown"] = {"owned_server_stopped": False, "failure": type(exc).__name__ + ": " + str(exc)}
        report["seconds"] = round(time.monotonic() - started, 3)
        report["passed"] = report["failure"] is None and report["shutdown"]["owned_server_stopped"]
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = asyncio.run(run(root, args.source, args.output))
    print(json.dumps({key: result[key] for key in ("passed", "requests", "seconds", "failure", "shutdown")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
