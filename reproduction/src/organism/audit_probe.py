"""Reconstruct a development preflight from recorded observations and actions."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from organism.diagnostics import (
    DEVELOPMENT_CASES, DiagnosticSession, evaluate_submission,
)


def audit_case(receipt: dict, expected_url: str, expected_alias: str) -> dict:
    cases = {c.case_id: c for c in DEVELOPMENT_CASES}
    if receipt.get("case_id") not in cases:
        raise ValueError("Unknown development case")
    case = cases[receipt["case_id"]]
    session = DiagnosticSession(case)
    allowed = {"inspect_device", "read_measurement", "submit_assessment"}
    recorded_actions = []
    for event in receipt["tool_events"]:
        name = event["tool"]
        if name not in allowed:
            raise ValueError("Undeclared tool in receipt")
        actual = getattr(session, name)(**event["arguments"])
        if actual != event["result"]:
            raise ValueError("Recorded tool observation differs from fixture replay")
        recorded_actions.append({"name": name, "input": event["arguments"]})
    if session.submission != receipt["submission"]:
        raise ValueError("Recorded submission differs from replay")
    evaluation = evaluate_submission(case, session)
    if receipt["failure"] is not None:
        evaluation["passed"] = False
    if evaluation != receipt["evaluation"]:
        raise ValueError("Reported evaluation differs from independent reconstruction")

    requests = receipt["requests"]
    responses = receipt["model_responses"]
    if not (len(requests) == len(responses) == len(receipt["model_calls"]) <= 4):
        raise ValueError("Request/response count or model-call budget mismatch")
    parsed_expected = urlsplit(expected_url)
    model_actions = []
    usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    for request, response in zip(requests, responses):
        parsed = urlsplit(request["url"])
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port != parsed_expected.port
            or parsed.path != "/v1/chat/completions"
            or parsed.username is not None
            or request["method"] != "POST"
        ):
            raise ValueError("Unexpected inference destination")
        body = request["body"]
        if (
            body["model"] != expected_alias or body["temperature"] != 0
            or body["max_tokens"] != 192
        ):
            raise ValueError("Inference settings differ from the preflight contract")
        message = response["message"]
        if message is None:
            if response["exception"] is None:
                raise ValueError("Response and exception are both absent")
            continue
        for content in message["content"]:
            if "toolUse" in content:
                tool = content["toolUse"]
                model_actions.append({"name": tool["name"], "input": tool["input"]})
        current = message["metadata"]["usage"]
        if (
            any(type(current.get(k)) is not int or current[k] < 0 for k in usage)
            or current["inputTokens"] + current["outputTokens"] != current["totalTokens"]
            or current["inputTokens"] > 4096 or current["outputTokens"] > 192
        ):
            raise ValueError("Invalid token accounting or exceeded token budget")
        for key in usage:
            usage[key] += current[key]
    if model_actions != recorded_actions:
        raise ValueError("Model-requested actions differ from recorded tool executions")
    return {
        "case_id": case.case_id, "evaluation": evaluation, "usage": usage,
        "model_calls": len(requests), "tool_calls": len(recorded_actions),
        "receipt_reconstructed": True,
    }


def audit_run(directory: Path) -> dict:
    report = json.loads((directory / "report.json").read_text())
    command = report["command"]
    alias = command[command.index("--alias") + 1]
    port = int(command[command.index("--port") + 1])
    if "--lora" in command or command[command.index("--host") + 1] != "127.0.0.1":
        raise ValueError("Expected an owned loopback-only stock-model server")
    for flag in ("--offline", "--no-kv-offload", "--no-op-offload"):
        if flag not in command:
            raise ValueError("Missing required preflight server option")
    if report["qualification_passed"] is not False or report["trained_for_organism"] is not False:
        raise ValueError("Preflight must not claim qualification or fine-tuning")
    rows = []
    files = {}
    for case in DEVELOPMENT_CASES:
        path = directory / f"{case.case_id}.json"
        raw = path.read_bytes()
        files[path.name] = hashlib.sha256(raw).hexdigest()
        rows.append(audit_case(json.loads(raw), f"http://127.0.0.1:{port}", alias))
    passed = sum(row["evaluation"]["passed"] for row in rows)
    if passed != report["passed_development_cases"] or len(rows) != report["total_development_cases"]:
        raise ValueError("Run totals differ from reconstructed case results")
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "SDK/tool receipts and deterministic development-fixture replay",
        "os_level_egress_independently_verified": False,
        "native_linux_qualification": False,
        "trained_model_qualification": False,
        "passed_development_cases": passed,
        "total_development_cases": len(rows),
        "all_receipts_reconstructed": True,
        "case_receipt_sha256": files,
        "report_sha256": hashlib.sha256((directory / "report.json").read_bytes()).hexdigest(),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(Path(__file__).with_name("diagnostics.py").read_bytes()).hexdigest(),
        "cases": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_run(args.directory.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in (
        "all_receipts_reconstructed", "passed_development_cases",
        "total_development_cases",
    )}))
