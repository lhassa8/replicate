"""Compare native training rendering/tokens to captured live server requests."""

import argparse
from copy import deepcopy
import difflib
import json
import os
from pathlib import Path


def native_messages(wire_messages):
    messages = deepcopy(wire_messages)
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            if any(item.get("type") != "text" for item in content):
                raise ValueError("Parity probe accepts text-only messages")
            message["content"] = "".join(item["text"] for item in content)
        elif content is None:
            message["content"] = ""
        for call in message.get("tool_calls", []):
            args = call["function"].get("arguments")
            if isinstance(args, str):
                call["function"]["arguments"] = json.loads(args)
    return messages


def check(directory: Path, tokenizer_path: Path) -> dict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.resolve(), local_files_only=True)
    selection = json.loads((directory / "fixture-selection.json").read_text())
    entries = []
    for case in selection["cases"]:
        receipt = json.loads((directory / f"{case['case_id']}.json").read_text())
        if len(receipt["requests"]) != len(receipt["server_rendered_requests"]):
            raise ValueError("Missing server request rendering")
        for index, (request, server) in enumerate(zip(receipt["requests"], receipt["server_rendered_requests"])):
            if "failure" in server:
                entries.append({"case_id": case["case_id"], "call": index, "passed": False, **server})
                continue
            body = request["body"]
            messages = native_messages(body["messages"])
            native = tokenizer.apply_chat_template(
                messages, tools=body["tools"], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            tokens = tokenizer.encode(native, add_special_tokens=False)
            same_text = native == server["prompt"]
            same_tokens = tokens == server["tokens"]
            entry = {
                "case_id": case["case_id"], "call": index, "passed": same_text and same_tokens,
                "text_equal": same_text, "tokens_equal": same_tokens,
                "native_tokens": len(tokens), "server_tokens": len(server["tokens"]),
            }
            if not same_text:
                entry["first_text_differences"] = list(difflib.unified_diff(
                    native.splitlines(), server["prompt"].splitlines(),
                    fromfile="native", tofile="server", n=1,
                ))[:20]
            entries.append(entry)
    result = {"passed": bool(entries) and all(e["passed"] for e in entries), "requests": entries}
    (directory / "template-parity.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.directory, args.tokenizer)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
