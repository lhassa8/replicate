"""Reconstruct the local adapter comparison from its plan and raw responses.

The original run's score is preserved. This separate audit handles the saved
template newline after EOS, which serving cannot generate after stopping.
"""

import argparse
import hashlib
import json
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def matches_served_target(expected, actual, stop_type, eos):
    require(expected.count(eos) == 1, "Expected one declared end token in the prepared target")
    end = expected.index(eos)
    # The prepared Granite template adds a newline after the supervised EOS.
    require(expected[end + 1:] == [198], "Unexpected post-EOS template suffix")
    if stop_type != "eos":
        return False
    if actual and actual[-1] == eos:
        actual = actual[:-1]
    return actual == expected[:end]


def audit(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    plan_path = directory / "development-plan.json"
    plan = json.loads(plan_path.read_text())
    require(identity(plan_path) == report["development_plan"], "Development plan changed")
    require(report["passed"] is True and report["failure"] is None, "Development run is incomplete or failed")
    require(report["frozen_merged_export_comparison"] is False and report["qualification_passed"] is False,
            "Development scope changed")
    require(len(plan["rows"]) == 8, "Development decision count differs")
    eos = json.loads((directory / "base-config/config.json").read_text())["eos_token_id"]
    arms = {}
    for arm, scale in (("stock-control", 0), ("trained-adapter", 1)):
        record = json.loads((directory / arm / "report.json").read_text())
        require(record == report["arms"][arm] and record["failure"] is None
                and record["shutdown"]["owned_server_stopped"] is True, "Arm receipt or shutdown differs")
        adapters = record["loaded_adapters"]
        expected_adapter = {"id": 0, "path": str(directory.resolve() / "trained-adapter-f32.gguf"),
                            "scale": float(scale)}
        require(len(adapters) == 1 and all(adapters[0].get(key) == value for key, value in expected_adapter.items())
                and set(adapters[0]) <= set(expected_adapter) | {"task_name", "prompt_prefix"}
                and all(adapters[0].get(key, "") == "" for key in ("task_name", "prompt_prefix")),
                "Loaded adapter scale or task metadata differs")
        require(len(record["requests"]) == len(plan["rows"]), "Arm decision count differs")
        decisions = []
        for row, observed in zip(plan["rows"], record["requests"]):
            require(observed["case_id"] == row["case_id"] and observed["decision_index"] == row["decision_index"],
                    "Arm row identity differs")
            request = observed["request"]
            require(request["prompt"] == row["input_ids"][:row["prompt_tokens"]]
                    and request["lora"] == [{"id": 0, "scale": scale}] and request["cache_prompt"] is False
                    and request["temperature"] == 0 and request["n_predict"] == 192,
                    "Serving input or generation settings differ")
            path = directory / arm / observed["response_file"]
            require(path.parent == directory / arm and identity(path) == observed["response"], "Response changed")
            body = json.loads(path.read_text())
            expected = row["input_ids"][row["prompt_tokens"]:]
            require(expected == observed["expected_tokens"], "Reference target changed")
            exact = matches_served_target(expected, body["tokens"], body["stop_type"], eos)
            decisions.append({"case_id": row["case_id"], "decision_index": row["decision_index"],
                              "role": row["role"], "target_kind": row["target_kind"],
                              "original_unadjusted_score": observed["exact_target_tokens"],
                              "exact_served_target": exact, "response": identity(path)})
        arms[arm] = {"decisions": decisions, "exact_served_targets": sum(item["exact_served_target"] for item in decisions),
                     "total": len(decisions), "peak_sampled_rss_bytes": record["peak_sampled_rss_bytes"],
                     "owned_server_stopped": True}
    return {
        "schema": "organism-adapter-development-score-audit-v1",
        "run_report": identity(directory / "report.json"), "plan": identity(plan_path), "arms": arms,
        "metric": "Exact prepared target tokens before EOS; require an EOS stop; omit only terminal EOS from returned tokens",
        "correction": "Original scorer compared the unreachable post-EOS template newline; original scores remain retained",
        "teacher_forced_prefixes": True, "tools_executed": False,
        "frozen_merged_export_comparison": False, "qualification_passed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.directory.resolve())
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
