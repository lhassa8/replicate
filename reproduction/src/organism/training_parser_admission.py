"""Reconstruct the parser evidence for every exact first-lineage training target.

This runs inside the isolated training package without executing the Mac parser
binary. It checks the recorded native parser output against source actions and
the exact decoded supervised bytes; the local raw parser run remains retained.
"""

import hashlib
import json
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def content_identity(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def same_content(receipt, path):
    return {key: receipt[key] for key in ("bytes", "sha256")} == content_identity(path)


def equivalent(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            equivalent(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            equivalent(a, b) for a, b in zip(actual, expected))
    if type(expected) in (int, float):
        return type(actual) in (int, float) and actual == expected
    return type(actual) is type(expected) and actual == expected


def keyed(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        require(row["id"] not in result, "Duplicate native parser evidence")
        result[row["id"]] = row
    return result


def verify(paths, recipe, rows):
    from tokenizers import Tokenizer

    codec = "granite-json-scalar-targets-v1"
    report = json.loads(paths["native_parser_report"].read_bytes())
    binding = json.loads(paths["token_parser_binding"].read_bytes())
    token_report = json.loads(paths["token_report"].read_bytes())
    require(report["schema"] == "organism-native-target-roundtrip-v1"
            and report["passed"] is True and report["failures"] == []
            and report["serving_parser_roundtrip_verified"] is True
            and report["runtime_commit"] == "4f31eedb0ccf546b7e8d6bb243b170f12522f54d"
            and report["model_generations"] == 0
            and report["target_codec"] == token_report["target_codec"] == codec
            and same_content(report["source_manifest"], paths["source_manifest"]),
            "Native parser report does not bind the candidate")
    require(binding["schema"] == "organism-native-token-parser-binding-v1"
            and binding["passed"] is True and binding["target_codec"] == codec
            and binding["all_prompt_tokens_masked"] is True
            and binding["all_target_bytes_match_passing_parser_inputs"] is True
            and binding["serving_prefix_and_row_reconstruction_passed"] is True
            and same_content(binding["native_parser_report"], paths["native_parser_report"])
            and same_content(binding["row_audit"], paths["row_audit"])
            and same_content(binding["token_report"], paths["token_report"]),
            "Parser report and final token rows are not bound")
    for key in ("native_parser_input", "native_parser_output"):
        matches = [entry for entry in report["inputs"]
                   if Path(entry["path"]).name == paths[key].name]
        require(len(matches) == 1 and same_content(matches[0], paths[key]),
                "Raw native parser evidence changed")
    for split in ("train", "validation"):
        matches = [entry for entry in binding["token_rows"]
                   if Path(entry["path"]).name == paths["tokens_" + split].name]
        require(len(matches) == 1 and same_content(matches[0], paths["tokens_" + split]),
                "Bound parser token split changed")
    inputs, outputs = keyed(paths["native_parser_input"]), keyed(paths["native_parser_output"])
    tokenizer = Tokenizer.from_file(str(paths["tokenizer/tokenizer.json"]))
    contracts = json.loads(paths["contracts"].read_bytes())
    expected = {}
    for split in ("train", "validation"):
        for line in paths["source_" + split].read_text().splitlines():
            record = json.loads(line)
            assistants = [message for message in record["messages"] if message["role"] == "assistant"]
            mask = record["supervised_assistants"]
            require(len(mask) == len(assistants) and all(type(value) is bool for value in mask),
                    "Invalid parser-reference supervision mask")
            for ordinal, (target, selected) in enumerate(zip(assistants, mask)):
                if selected:
                    key = f"{split}/{record['case_id']}/{ordinal}"
                    require(key not in expected, "Duplicate parser-reference action")
                    expected[key] = (target, contracts[record["role"]]["tools"])
    seen = set()
    for split, group in rows.items():
        for row in group:
            key = f"{split}/{row['case_id']}/{row['decision_index']}"
            require(key not in seen and key in expected and key in inputs and key in outputs,
                    "Missing or repeated exact parser target")
            target, tools = expected[key]
            prefix = row["prompt_tokens"]
            require(row["labels"] == [-100] * prefix + row["input_ids"][prefix:],
                    "Prompt tokens entered parser-bound supervision")
            text = tokenizer.decode(row["input_ids"][prefix:], skip_special_tokens=False)
            require(text.count("<|im_end|>") == 1, "Incomplete parser-bound target")
            body, end = text.split("<|im_end|>")
            require(not end.strip() and body == inputs[key]["text"] and tools == inputs[key]["tools"],
                    "Final target bytes or tools differ from actual parser input")
            result = outputs[key]
            require(result["parsed"] is True and result["format"] == "peg-native",
                    "Native runtime did not parse the target")
            message = result["message"]
            actual_calls, wanted_calls = message.get("tool_calls", []), target.get("tool_calls", [])
            require(len(actual_calls) == len(wanted_calls), "Native parser changed the tool count")
            for actual, wanted in zip(actual_calls, wanted_calls):
                require(actual["function"]["name"] == wanted["function"]["name"]
                        and equivalent(json.loads(actual["function"]["arguments"]),
                                       wanted["function"]["arguments"]),
                        "Native parser changed the source action or argument types")
            require((message.get("content") or "").strip() == (target.get("content") or "").strip(),
                    "Native parser changed the final acknowledgement")
            seen.add(key)
    require(seen == set(expected) == set(inputs) == set(outputs)
            and len(seen) == sum(recipe["counts"].values())
            == report["targets"] == binding["complete_supervised_targets"] == 974,
            "Native parser evidence does not cover the full candidate")
    return {"passed": True, "complete_targets": len(seen), "exact_bytes_verified": True,
            "source_action_semantics_verified": True, "live_parser_rerun_claimed": False}
