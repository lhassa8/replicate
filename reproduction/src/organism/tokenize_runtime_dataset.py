"""Prepare complete action and final-text targets for the actual runtime roles."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path

from .audit_runtime_dataset import require
from .runtime_rows import INHERITED_CORPUS_SCHEMA, STARTUP_CORPUS_SCHEMA, inspect_record, targets
from .tokenize_dataset import completion_tokens, sha256


def prepare(source, tokenizer_path, audit_path, output, serving_capture, *,
            target_codec="upstream-native-v1"):
    require(target_codec in ("upstream-native-v1", "granite-json-scalar-targets-v1"),
            "Unsupported assistant target codec")
    source, tokenizer_path, audit_path, output, serving_capture = map(
        Path, (source, tokenizer_path, audit_path, output, serving_capture))
    manifest = json.loads((source / "manifest.json").read_bytes())
    require(target_codec == manifest.get("target_codec", "upstream-native-v1"),
            "Target codec differs from the declared source corpus")
    audit = json.loads(audit_path.read_bytes())
    require(audit.get("passed") is True and audit.get("current_runtime_replay_verified") is True
            and audit.get("independent_semantics_verified") is True
            and audit.get("source_manifest_sha256") == sha256(source / "manifest.json"), "Missing matching runtime replay audit")
    require(manifest["scope"] == "development_references_through_current_runtime_tools"
            and all(manifest[key] is False for key in ("training_admitted", "optimization_started", "qualification_data_present")),
            "Only unadmitted runtime development references are accepted")
    require(set(manifest["files"]) == {"train.jsonl", "validation.jsonl", "contracts.json", "release.json"},
            "Unexpected dataset inventory")
    for name, expected in manifest["files"].items():
        path = source / name
        require(expected == {"sha256": sha256(path), "bytes": path.stat().st_size}, "Dataset file changed: " + name)
    contracts = json.loads((source / "contracts.json").read_bytes())
    release = json.loads((source / "release.json").read_bytes())
    capture = json.loads((serving_capture / "report.json").read_bytes())
    require(capture["passed"] is True and capture["source_manifest_sha256"] == sha256(source / "manifest.json")
            and capture["shutdown"]["owned_server_stopped"] is True, "Missing matching completed server capture")
    require(sha256(serving_capture / "server-prompts.jsonl") == capture["server_prompts_sha256"], "Server capture changed")
    server_rows = {}
    for line in (serving_capture / "server-prompts.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = (row["split"], row["case_id"], row["decision_index"])
        require(key not in server_rows, "Duplicate server prompt")
        server_rows[key] = row
    require(len(server_rows) == capture["requests"], "Server capture count differs")
    receipt = json.loads((tokenizer_path / "receipt.json").read_bytes())
    for item in receipt["files"]:
        path = tokenizer_path / item["name"]
        require(path.parent.resolve() == tokenizer_path.resolve() and sha256(path) == item["sha256"],
                "Pinned tokenizer file changed")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.resolve(), local_files_only=True)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "in_progress",
        "source_manifest_sha256": sha256(source / "manifest.json"), "runtime_audit_sha256": sha256(audit_path),
        "serving_capture_report_sha256": sha256(serving_capture / "report.json"),
        "serving_prompts_sha256": capture["server_prompts_sha256"],
        "tokenizer_receipt_sha256": sha256(tokenizer_path / "receipt.json"),
        "repository": receipt["repository"], "revision": receipt["revision"],
        "versions": {name: version(name) for name in ("transformers", "tokenizers", "jinja2")},
        "sources": {path.name: sha256(path) for path in
                    (Path(__file__), Path(__file__).with_name("tokenize_dataset.py"),
                     Path(__file__).with_name("audit_runtime_dataset.py"),
                     Path(__file__).with_name("runtime_rows.py"),
                     Path(__file__).with_name("audit_equipment_dataset.py"))},
        "context_limit": 4096, "completion_limit": 192, "truncation_allowed": False,
        "target_codec": target_codec,
        "serving_parser_roundtrip_verified": False,
        "loss_policy": "One complete assistant tool action or final acknowledgement per row; entire prompt masked.",
        "prompt_policy": "Use the pinned server's captured prompt text and verified token IDs; preserve the full native target.",
        "training_admitted": False, "optimization_started": False, "qualification_passed": False,
        "unmodified_native_prompt_parity_verified": False, "serving_prefix_parity_verified": False, "splits": {},
    }
    if target_codec == "granite-json-scalar-targets-v1":
        report["sources"]["native_target_codec.py"] = sha256(Path(__file__).with_name("native_target_codec.py"))
        report["prompt_policy"] = (
            "Use unchanged pinned serving prefixes and complete native actions with JSON scalar literals. "
            "Require a separate serving-parser semantic round-trip before admission.")
    if manifest["schema"] == "organism-first-lineage-runtime-corpus-v1":
        for name in ("first_lineage_dataset.py", "audit_first_lineage_dataset.py",
                     "reproduction_protocol.py", "audit_equipment_diagnostics.py", "temperature_measurement.py"):
            current = sha256(Path(__file__).with_name(name))
            require(current == manifest["sources"][name], "First-lineage reference checker changed")
            report["sources"][name] = current
    if manifest["schema"] in (INHERITED_CORPUS_SCHEMA, STARTUP_CORPUS_SCHEMA):
        # This path performs signature and independent source reconstruction,
        # so include those checker implementations in its provenance.
        for name in ("audit_inheritance_dataset.py", "audit_inherited_diagnostics.py", "experience.py",
                     "experience_issue.py", "diagnostic_oracle.py", "enrollment.py", "capsule.py",
                     "audit_equipment_diagnostics.py", "audit_application.py"):
            report["sources"][name] = sha256(Path(__file__).with_name(name))
        report["versions"]["cryptography"] = version("cryptography")
    if manifest["schema"] == STARTUP_CORPUS_SCHEMA:
        for name in ("startup_dataset.py", "audit_startup_dataset.py"):
            report["sources"][name] = sha256(Path(__file__).with_name(name))
    seen = set()
    try:
        for split in ("train", "validation"):
            stats = {"decisions": 0, "tokens": 0, "supervised_tokens": 0, "maximum_context_tokens": 0,
                     "maximum_target_tokens": 0, "unmodified_native_prefix_mismatches": 0,
                     "roles": {}, "target_kinds": {"tool": 0, "final": 0}}
            with (output / (split + ".tokens.jsonl")).open("x") as destination:
                for line in (source / (split + ".jsonl")).read_text().splitlines():
                    record = json.loads(line)
                    require(record["split"] == split, "Moved development split")
                    contract = contracts[record["role"]]
                    require(inspect_record(record, contract, release) == record["evaluation"], "Trajectory evaluation differs")
                    for decision, index, message in targets(record):
                        kind = "tool" if message.get("tool_calls") else "final"
                        native_full, _, native_prefix, target_text = completion_tokens(
                            tokenizer, record["messages"][:index], message, contract["tools"],
                            target_kind=kind, target_codec=target_codec)
                        key = (split, record["case_id"], decision)
                        require(key in server_rows and key not in seen, "Missing or duplicate matched server prompt")
                        seen.add(key)
                        server = server_rows[key]
                        require(server["request"] == {"messages": record["wire_messages"][:index], "tools": contract["tools"]},
                                "Server request differs from the actual trajectory")
                        prefix_ids = tokenizer.encode(server["prompt"], add_special_tokens=False)
                        require(prefix_ids == server["tokens"], "Native tokenizer and server prompt tokens differ")
                        target_ids = native_full[native_prefix:]
                        full = prefix_ids + target_ids
                        require(tokenizer.encode(server["prompt"] + target_text, add_special_tokens=False) == full,
                                "Target is not an exact continuation of the serving prompt")
                        require(len(full) <= 4096 and len(target_ids) <= 192, "Serving-prefix example exceeds limits")
                        prefix = len(prefix_ids)
                        labels = [-100] * prefix + target_ids
                        stats["unmodified_native_prefix_mismatches"] += int(native_full[:native_prefix] != prefix_ids)
                        row = {"case_id": record["case_id"], "profile_id": record["profile_id"], "role": record["role"],
                               "decision_index": decision, "target_kind": kind, "tool_contract_sha256": contract["sha256"],
                               "input_ids": full, "labels": labels, "prompt_tokens": prefix}
                        destination.write(json.dumps(row, separators=(",", ":")) + "\n")
                        stats["decisions"] += 1
                        stats["tokens"] += len(full)
                        stats["supervised_tokens"] += len(full) - prefix
                        stats["maximum_context_tokens"] = max(stats["maximum_context_tokens"], len(full))
                        stats["maximum_target_tokens"] = max(stats["maximum_target_tokens"], len(full) - prefix)
                        stats["roles"][record["role"]] = stats["roles"].get(record["role"], 0) + 1
                        stats["target_kinds"][kind] += 1
            require(stats["decisions"] == manifest["counts"][split]["assistant_targets"], "Target count changed")
            require(stats["target_kinds"]["final"] == manifest["counts"][split]["trajectories"], "Missing final acknowledgements")
            stats["sha256"] = sha256(output / (split + ".tokens.jsonl"))
            report["splits"][split] = stats
        require(seen == set(server_rows), "Unused or missing serving prompts")
        report["status"] = "serving_prefix_and_native_targets_verified_not_admitted"
        report["serving_prefix_parity_verified"] = True
        report["unmodified_native_prompt_parity_verified"] = not any(
            split["unmodified_native_prefix_mismatches"] for split in report["splits"].values())
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "tokenizer", "audit", "output", "serving-capture"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--target-codec",
                        choices=("upstream-native-v1", "granite-json-scalar-targets-v1"),
                        default="upstream-native-v1")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.tokenizer, args.audit, args.output, args.serving_capture,
                             target_codec=args.target_codec), indent=2))
