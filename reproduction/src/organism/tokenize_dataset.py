"""Validate exact native tool-action targets without a training backend.

Run in the isolated tokenizer environment. Every row supervises one assistant
action and masks its entire prompt. No truncation, inferred token counts, or
qualification data is permitted. This produces candidate data, not trained
weights or a training-admission decision.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completion_tokens(tokenizer, prompt: list[dict], completion: dict, tools: list[dict], *,
                      target_kind="tool", target_codec="upstream-native-v1"):
    if completion.get("role") != "assistant" or target_kind not in ("tool", "final"):
        raise ValueError("Expected a declared assistant target kind")
    if target_codec not in ("upstream-native-v1", "granite-json-scalar-targets-v1"):
        raise ValueError("Unsupported assistant target codec")
    if target_kind == "final" and (completion.get("tool_calls") or not isinstance(completion.get("content"), str)
                                   or not completion["content"].strip()):
        raise ValueError("Final acknowledgement must be nonempty text without tool calls")
    options = {
        "tools": tools, "tokenize": True, "return_dict": False,
        "enable_thinking": False, "truncate_history_thinking": True,
    }
    prefix = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, **options)
    full = tokenizer.apply_chat_template(prompt + [completion], add_generation_prompt=False, **options)
    if full[:len(prefix)] != prefix:
        raise ValueError("Native assistant target is not an exact token continuation")
    target = full[len(prefix):]
    if not target or tokenizer.eos_token_id not in target:
        raise ValueError("Incomplete assistant target or missing end token")
    text = tokenizer.decode(target)
    if target_codec == "granite-json-scalar-targets-v1":
        from .native_target_codec import json_scalar_target
        text = json_scalar_target(text, completion)
        target = tokenizer.encode(text, add_special_tokens=False)
        full = prefix + target
        if tokenizer.encode(tokenizer.decode(prefix) + text, add_special_tokens=False) != full:
            raise ValueError("Corrected assistant target is not an exact token continuation")
        if not target or tokenizer.eos_token_id not in target:
            raise ValueError("Corrected assistant target lost its end token")
    if len(full) > 4096 or len(target) > 192:
        raise ValueError(f"Untruncated example exceeds contract: context={len(full)}, target={len(target)}")
    if target_kind == "tool" and ("<tool_call>" not in text or "</tool_call>" not in text):
        raise ValueError("Assistant target lost its complete native tool call")
    if target_kind == "final" and ("<tool_call>" in text or "</tool_call>" in text):
        raise ValueError("Final acknowledgement unexpectedly contains a native tool call")
    return full, [-100] * len(prefix) + target, len(prefix), text


def prepare(source: Path, tokenizer_path: Path, output: Path) -> dict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["qualification_data_present"] or manifest["scope"] != "development_only_in_memory_simulation":
        raise ValueError("Tokenizer preflight accepts development material only")
    for name, expected in manifest["files"].items():
        path = source / name
        if path.parent.resolve() != source.resolve() or sha256(path) != expected["sha256"]:
            raise ValueError("Dataset file hash or path validation failed")
    tools = json.loads((source / "tools.json").read_text())
    tokenizer_receipt = json.loads((tokenizer_path / "receipt.json").read_text())
    for item in tokenizer_receipt["files"]:
        path = tokenizer_path / item["name"]
        if path.parent.resolve() != tokenizer_path.resolve() or sha256(path) != item["sha256"]:
            raise ValueError("Pinned tokenizer source hash or path validation failed")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.resolve(), local_files_only=True)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_data_preflight_in_progress",
        "optimization_started": False, "training_admitted": False,
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "tokenizer_receipt_sha256": sha256(tokenizer_path / "receipt.json"),
        "repository": tokenizer_receipt["repository"], "revision": tokenizer_receipt["revision"],
        "versions": {name: version(name) for name in ("transformers", "tokenizers", "jinja2")},
        "tokenizer_class": type(tokenizer).__name__, "vocabulary_size": len(tokenizer),
        "loss_policy": "One assistant tool action per row; all prompt and observation tokens masked.",
        "context_limit": 4096, "completion_limit": 192, "truncation_allowed": False,
        "splits": {},
    }
    try:
        for split in ("train", "validation"):
            count = total_tokens = target_tokens = max_context = max_target = 0
            min_target = None
            with (output / f"{split}.tokens.jsonl").open("w") as destination:
                for line in (source / f"{split}.jsonl").read_text().splitlines():
                    trajectory = json.loads(line)
                    messages = trajectory["messages"]
                    for index, message in enumerate(messages):
                        if message["role"] != "assistant":
                            continue
                        full, labels, prefix_length, target_text = completion_tokens(
                            tokenizer, messages[:index], message, tools,
                        )
                        supervised = len(full) - prefix_length
                        row = {
                            "case_id": trajectory["case_id"], "family_id": trajectory["family_id"],
                            "decision_index": (index - 2) // 2,
                            "input_ids": full, "labels": labels,
                            "prompt_tokens": prefix_length,
                        }
                        destination.write(json.dumps(row, separators=(",", ":")) + "\n")
                        if count == 0:
                            (output / f"{split}.first-target.txt").write_text(target_text)
                        count += 1
                        total_tokens += len(full)
                        target_tokens += supervised
                        max_context = max(max_context, len(full))
                        max_target = max(max_target, supervised)
                        min_target = supervised if min_target is None else min(min_target, supervised)
            if count != manifest["counts"][split]["assistant_decisions"]:
                raise ValueError("Tokenized decision count differs from verified trajectory count")
            report["splits"][split] = {
                "decisions": count, "tokens": total_tokens, "supervised_tokens": target_tokens,
                "maximum_context_tokens": max_context, "maximum_target_tokens": max_target,
                "minimum_target_tokens": min_target,
                "sha256": sha256(output / f"{split}.tokens.jsonl"),
            }
        report["status"] = "exact_native_token_targets_verified_not_admitted"
        report["serving_template_parity_verified"] = False
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.tokenizer, args.output), indent=2))
