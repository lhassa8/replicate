"""Compare server prompts with the exact saved candidate training prefixes."""

import argparse
import json
import os
from pathlib import Path

from .audit_runtime_dataset import require
from .runtime_rows import inspect_record, targets
from .check_template_parity import native_messages
from .tokenize_dataset import completion_tokens, sha256


def check(directory, token_directory, tokenizer_path, source):
    record_inspector, record_targets = inspect_record, targets
    source_manifest = json.loads((source / "manifest.json").read_bytes())
    if source_manifest["schema"] == "organism-refinement-runtime-corpus-v1":
        from .audit_refinement_dataset import inspect_record as inspect_refined, targets as refined_targets, verify_dataset
        verify_dataset(source)
        record_inspector, record_targets = inspect_refined, refined_targets
    capture = json.loads((directory / "report.json").read_bytes())
    data = json.loads((token_directory / "report.json").read_bytes())
    target_codec = data.get("target_codec", "upstream-native-v1")
    require(target_codec in ("upstream-native-v1", "granite-json-scalar-targets-v1"),
            "Unsupported assistant target codec")
    require(target_codec == source_manifest.get("target_codec", "upstream-native-v1"),
            "Source and token target codecs differ")
    receipt = json.loads((tokenizer_path / "receipt.json").read_bytes())
    require(capture["passed"] is True and capture["shutdown"]["owned_server_stopped"] is True
            and capture["source_manifest_sha256"] == data["source_manifest_sha256"], "Capture and token dataset are not bound")
    require(data["status"] == "serving_prefix_and_native_targets_verified_not_admitted", "Incomplete native token preparation")
    require(data["serving_capture_report_sha256"] == sha256(directory / "report.json")
            and data["serving_prompts_sha256"] == capture["server_prompts_sha256"], "Token rows use a different server capture")
    require(sha256(directory / "server-prompts.jsonl") == capture["server_prompts_sha256"], "Server capture hash changed")
    require(sha256(tokenizer_path / "receipt.json") == data["tokenizer_receipt_sha256"], "Tokenizer receipt changed")
    for item in receipt["files"]:
        path = tokenizer_path / item["name"]
        require(path.parent.resolve() == tokenizer_path.resolve() and sha256(path) == item["sha256"], "Tokenizer file changed")
    rows = {}
    for split, stats in data["splits"].items():
        require(split in ("train", "validation"), "Unexpected token split")
        path = token_directory / (split + ".tokens.jsonl")
        require(sha256(path) == stats["sha256"], "Token rows changed")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (split, row["case_id"], row["decision_index"])
            require(key not in rows, "Duplicate token target")
            rows[key] = row
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.resolve(), local_files_only=True)
    manifest = json.loads((source / "manifest.json").read_bytes())
    require(sha256(source / "manifest.json") == capture["source_manifest_sha256"], "Reference source changed")
    for name, expected in manifest["files"].items():
        require(Path(name).name == name and sha256(source / name) == expected["sha256"], "Reference file changed")
    contracts = json.loads((source / "contracts.json").read_bytes())
    release = json.loads((source / "release.json").read_bytes())
    references = {}
    for split in ("train", "validation"):
        for line in (source / (split + ".jsonl")).read_text().splitlines():
            record = json.loads(line)
            require(record["split"] == split, "Reference split changed")
            contract = contracts[record["role"]]
            require(record_inspector(record, contract, release) == record["evaluation"], "Reference semantic audit changed")
            for step, index, message in record_targets(record):
                key = (split, record["case_id"], step)
                require(key not in references, "Duplicate reference")
                references[key] = (record, index, contract)
    entries, seen = [], set()
    for line in (directory / "server-prompts.jsonl").read_text().splitlines():
        server = json.loads(line)
        key = (server["split"], server["case_id"], server["decision_index"])
        require(key in rows and key in references and key not in seen, "Extra or duplicate server prompt")
        seen.add(key)
        row = rows[key]
        body = server["request"]
        record, index, contract = references[key]
        require(body == {"messages": record["wire_messages"][:index], "tools": contract["tools"]},
                "Serving prompt belongs to another reference")
        for field in ("case_id", "profile_id", "role", "tool_contract_sha256"):
            require(row[field] == record[field], "Token row reference binding differs")
        target = record["messages"][index]
        kind = "tool" if target.get("tool_calls") else "final"
        require(row["target_kind"] == kind, "Target kind changed")
        full, _, boundary, target_text = completion_tokens(
            tokenizer, record["messages"][:index], target, contract["tools"],
            target_kind=kind, target_codec=target_codec)
        target_ids = full[boundary:]
        native = tokenizer.apply_chat_template(native_messages(body["messages"]), tools=body["tools"], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False,
                                               truncate_history_thinking=True)
        ids = tokenizer.encode(native, add_special_tokens=False)
        text_equal, tokens_equal = native == server["prompt"], ids == server["tokens"]
        prefix_equal = row["input_ids"][:row["prompt_tokens"]] == server["tokens"]
        prefix_text_equal = tokenizer.decode(row["input_ids"][:row["prompt_tokens"]]) == server["prompt"]
        target_equal = row["input_ids"][row["prompt_tokens"]:] == target_ids
        row_equal = (row["input_ids"] == server["tokens"] + target_ids
                     and row["labels"] == [-100] * len(server["tokens"]) + target_ids
                     and row["prompt_tokens"] == len(server["tokens"])
                     and len(row["input_ids"]) <= 4096 and len(target_ids) <= 192)
        joined_equal = tokenizer.encode(server["prompt"] + target_text, add_special_tokens=False) == row["input_ids"]
        server_tokenization_equal = tokenizer.encode(server["prompt"], add_special_tokens=False) == server["tokens"]
        entries.append({"case_id": key[1], "split": key[0], "decision_index": key[2], "target_kind": row["target_kind"],
                        "unmodified_native_text_equal": text_equal, "unmodified_native_tokens_equal": tokens_equal,
                        "training_prefix_equal": prefix_equal, "training_prefix_text_equal": prefix_text_equal,
                        "native_target_equal": target_equal, "entire_row_and_mask_verified": row_equal,
                        "tokens": row["prompt_tokens"],
                        "passed": prefix_equal and prefix_text_equal and target_equal and row_equal and joined_equal and server_tokenization_equal})
    require(seen == set(rows) == set(references) and len(entries) == capture["requests"], "Missing prompt comparison")
    return {"passed": bool(entries) and all(item["passed"] for item in entries), "requests": entries,
            "target_codec": target_codec, "serving_parser_roundtrip_verified": False,
            "capture_report_sha256": sha256(directory / "report.json"), "token_report_sha256": sha256(token_directory / "report.json"),
            "checker_sha256": sha256(Path(__file__)), "training_admitted": False, "qualification_passed": False,
            "unmodified_native_prompt_parity_verified": all(item["unmodified_native_text_equal"] and
                                                            item["unmodified_native_tokens_equal"] for item in entries),
            "scope": "Captured serving prompt, exact full-precision native target, reference semantics and complete masked training row"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("directory", "tokens", "tokenizer", "output", "source"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    with args.output.open("x") as destination:
        try:
            result = check(args.directory, args.tokens, args.tokenizer, args.source)
        except Exception as exc:
            result = {"passed": False, "failure": type(exc).__name__ + ": " + str(exc), "training_admitted": False}
        destination.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "requests": len(result.get("requests", [])), "failure": result.get("failure")}))
    raise SystemExit(0 if result["passed"] else 1)
