"""Replay setup-model receipts and match executed actions to model requests."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from urllib.parse import urlparse

from .probe_local import digest_file, write_json
from .probe_setup import MODEL_CALL_LIMIT, selected_cases
from .setup_reference import SYSTEM_PROMPT
from .setup_simulator import SetupSession, TOOLS, evaluate_setup


def audit_call_records(receipt: dict, port: int, alias: str):
    requests = receipt["requests"]
    if not (
        len(requests) == len(receipt["model_calls"]) == len(receipt["model_responses"])
        == len(receipt["model_streams"])
        and 0 < len(requests) <= MODEL_CALL_LIMIT
    ):
        raise ValueError("Model call accounting differs")
    results_so_far = []
    event_index = 0
    for index, request in enumerate(requests):
        url = urlparse(request["url"])
        body = request["body"]
        if (url.scheme, url.hostname, url.port, url.path) != ("http", "127.0.0.1", port, "/v1/chat/completions"):
            raise ValueError("Inference endpoint mismatch")
        if body["model"] != alias or body["temperature"] != 0 or body["max_tokens"] != 192:
            raise ValueError("Inference settings changed")
        if body["tools"] != TOOLS:
            raise ValueError("Live tools differ from the training contract")
        if body["messages"][0] != {"role": "system", "content": SYSTEM_PROMPT}:
            raise ValueError("System prompt differs from the training contract")
        observed = []
        for message in body["messages"]:
            if message["role"] == "tool":
                content = message["content"]
                text = content if isinstance(content, str) else "".join(c["text"] for c in content)
                observed.append(json.loads(text))
        if observed != results_so_far:
            raise ValueError("Model received missing or altered tool observations")
        response = receipt["model_responses"][index]
        for block in (response.get("message") or {}).get("content", []):
            if "toolUse" in block:
                if event_index >= len(receipt["tool_events"]):
                    raise ValueError("Model action has no executed event")
                event = receipt["tool_events"][event_index]
                use = block["toolUse"]
                if event["tool"] != use["name"] or event["arguments"] != use["input"]:
                    raise ValueError("Model action differs from executed event")
                results_so_far.append(event["result"])
                event_index += 1
        usage = [e["metadata"]["usage"] for e in receipt["model_streams"][index] if "metadata" in e]
        if len(usage) != 1:
            raise ValueError("Missing or duplicate token usage")
        u = usage[0]
        if not (0 < u["inputTokens"] <= 4096 and 0 <= u["outputTokens"] <= 192):
            raise ValueError("Model token budget exceeded")
        if u["totalTokens"] != u["inputTokens"] + u["outputTokens"]:
            raise ValueError("Inconsistent token usage")
    if event_index != len(receipt["tool_events"]):
        raise ValueError("Executed event has no model action")


def audit(directory: Path) -> dict:
    run = json.loads((directory / "report.json").read_text())
    selection = json.loads((directory / "fixture-selection.json").read_text())
    if selection["cases"] != [asdict(c) for c in selected_cases()]:
        raise ValueError("Scheduled fixtures changed")
    command = run["command"]
    port = int(command[command.index("--port") + 1])
    alias = command[command.index("--alias") + 1]
    results = []
    for case in selected_cases():
        path = directory / f"{case.case_id}.json"
        receipt = json.loads(path.read_text())
        if receipt["case"] != asdict(case):
            raise ValueError("Case truth changed")
        session = SetupSession(case)
        for event in receipt["tool_events"]:
            actual = session.call(event["tool"], event["arguments"])
            if actual != event["result"]:
                raise ValueError("Tool observation differs from independent replay")
        if session.evaluator_snapshot() != receipt["final_state"]:
            raise ValueError("Final state differs from independent replay")
        expected = evaluate_setup(case, session)
        expected["passed"] = expected["passed"] and receipt["failure"] is None
        if expected != receipt["evaluation"]:
            raise ValueError("Claimed evaluation differs from replay")
        audit_call_records(receipt, port, alias)
        results.append({
            "case_id": case.case_id, "receipt_sha256": digest_file(path),
            "evaluation": expected, "replay_passed": True,
        })
    count = sum(r["evaluation"]["passed"] for r in results)
    if (
        run["passed_development_cases"] != count or run["total_development_cases"] != len(results)
        or len(run["cases"]) != len(results) or not run.get("owned_server_stopped")
    ):
        raise ValueError("Run summary or owned-process cleanup differs")
    for summary, reconstructed in zip(run["cases"], results):
        if summary["case_id"] != reconstructed["case_id"] or summary["evaluation"] != reconstructed["evaluation"]:
            raise ValueError("Run case summary differs")
    result = {
        "scope": "development receipt reconstruction only",
        "passed": True, "report_sha256": digest_file(directory / "report.json"),
        "cases": results, "stock_successes": count,
        "qualification_passed": False, "native_linux_verified": False,
    }
    write_json(directory / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))
