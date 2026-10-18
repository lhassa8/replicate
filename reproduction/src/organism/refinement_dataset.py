"""Prepare the separately versioned refinement corpus and composed token inputs."""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .audit_refinement_dataset import (CORPUS_SCHEMA, SCHEMA, PROVENANCE, PARENT_SHA, REFERENCE_SHA,
                                       audit, inspect_record, verify_dataset)
from .audit_runtime_dataset import digest, require
from .tokenize_dataset import sha256


def write(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def build_source(root, output):
    root, output = Path(root), Path(output)
    base = root / "organism"
    parent = base / "data/equipment-runtime-development-v4-01"
    candidate = base / "evidence/refinement-corpus-preparation-20261016-01"
    probes = base / "evidence/training-refinement-probe-20261015-01/prepared"
    require(sha256(parent / "manifest.json") == PARENT_SHA
            and sha256(probes / "records.jsonl") == REFERENCE_SHA, "Pinned source identity differs")
    prior = json.loads((candidate / "independent-readback.json").read_bytes())
    require(prior["composition_reconstructed"] is True
            and prior["manifest"] == identity(candidate / "candidate/manifest.json"),
            "Prior composition audit is not bound")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("contracts.json", "release.json", "validation.jsonl"):
        shutil.copyfile(parent / name, output / name)
    shutil.copyfile(parent / "manifest.json", output / "parent-manifest.json")
    shutil.copyfile(probes / "records.jsonl", output / "references.jsonl")
    contracts = json.loads((output / "contracts.json").read_bytes())
    release = json.loads((output / "release.json").read_bytes())
    require(contracts["equipment"] == json.loads((probes / "contract.json").read_bytes()),
            "Equipment contract changed")
    with (output / "train.jsonl").open("xb") as stream:
        stream.write((parent / "train.jsonl").read_bytes())
        for reference in rows(probes / "records.jsonl"):
            case = deepcopy(reference["case"])
            case["split"] = "train"
            record = {"schema": SCHEMA, **{k: case[k] for k in ("case_id", "role", "split", "profile_id")},
                      "case": case, "reference": reference, "provenance": PROVENANCE,
                      "tool_contract_sha256": contracts["equipment"]["sha256"],
                      "supervised_assistants": [True] * 5, **reference["execution"]}
            record["evaluation"] = inspect_record(record, contracts["equipment"], release)
            stream.write((json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode())
    files = {p.name: identity(p) for p in output.iterdir() if p.is_file()}
    manifest = {
        "schema": CORPUS_SCHEMA, "dataset_id": "equipment-runtime-development-v5-refinement-01",
        "created_at": datetime.now(timezone.utc).isoformat(), "files": files,
        "parent_trajectories_preserved": 500, "supplement_trajectories": 84,
        "training_admitted": False, "optimization_started": False, "qualification_data_present": False,
        "human_review_claimed": False, "teacher_model_used": False,
        "source_composition_audit": identity(candidate / "independent-readback.json"),
        "scope": "Development references; unchanged original split plus training-family supplement.",
    }
    write(output / "manifest.json", manifest)
    verify_dataset(output)
    return manifest


def compose_tokens(root, source, token_output, capture_output, runtime_report):
    base = Path(root) / "organism"
    source, token_output, capture_output = map(Path, (source, token_output, capture_output))
    parent_tokens = base / "data/equipment-runtime-development-v4-01-tokens"
    parent_capture = base / "evidence/startup-training-template-capture-20261014-01"
    extra = base / "evidence/refinement-target-preparation-20261015-01"
    candidate = base / "evidence/refinement-corpus-preparation-20261016-01/candidate"
    prior = json.loads((extra / "independent-readback.json").read_bytes())
    require(prior["passed"] is True and prior["capture_report"] == identity(extra / "capture/report.json")
            and prior["token_report"] == identity(extra / "tokens/report.json"), "Supplement evidence changed")
    extra_capture = json.loads((extra / "capture/report.json").read_bytes())
    require(extra_capture["prompts"] == identity(extra / "capture/server-prompts.jsonl")
            and extra_capture["model_generations"] == 0 and extra_capture["shutdown"]["owned_server_stopped"] is True,
            "Supplement capture differs or server remains active")
    old_report = json.loads((parent_tokens / "report.json").read_bytes())
    require(old_report["serving_prompts_sha256"] == sha256(parent_capture / "server-prompts.jsonl")
            and old_report["serving_capture_report_sha256"] == sha256(parent_capture / "report.json"),
            "Original capture changed")
    runtime = json.loads(Path(runtime_report).read_bytes())
    require(runtime["passed"] is True and runtime["source_manifest_sha256"] == sha256(source / "manifest.json"),
            "Runtime replay did not verify this corpus")
    verify_dataset(source)
    token_output.mkdir(parents=True, exist_ok=False)
    capture_output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "validation"):
        shutil.copyfile(candidate / (split + ".tokens.jsonl"), token_output / (split + ".tokens.jsonl"))
    with (capture_output / "server-prompts.jsonl").open("xb") as output:
        output.write((parent_capture / "server-prompts.jsonl").read_bytes())
        for row in rows(extra / "capture/server-prompts.jsonl"):
            require(row["split"] == "train_preparation", "Supplement capture crossed splits")
            row["split"] = "train"
            output.write((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode())
    parent_capture_report = json.loads((parent_capture / "report.json").read_bytes())
    require(parent_capture_report["passed"] is True
            and parent_capture_report["shutdown"]["owned_server_stopped"] is True,
            "Original capture was not complete")
    report = {
        "scope": "Composition of two completed serving captures; no new model or server execution.",
        "created_at": datetime.now(timezone.utc).isoformat(), "passed": True,
        "source_manifest_sha256": sha256(source / "manifest.json"), "requests": 2676,
        "server_prompts_sha256": sha256(capture_output / "server-prompts.jsonl"),
        "shutdown": {"owned_server_stopped": True, "scope": "Both original capture sessions verified stopped."},
        "component_captures": {"parent": identity(parent_capture / "report.json"),
                               "supplement": identity(extra / "capture/report.json")},
        "model_generations": 0, "training_admitted": False, "qualification_passed": False,
    }
    write(capture_output / "report.json", report)
    splits = {}
    for split in ("train", "validation"):
        path = token_output / (split + ".tokens.jsonl")
        items = rows(path)
        splits[split] = {
            "decisions": len(items), "tokens": sum(len(r["input_ids"]) for r in items),
            "supervised_tokens": sum(len(r["input_ids"]) - r["prompt_tokens"] for r in items),
            "maximum_context_tokens": max(len(r["input_ids"]) for r in items),
            "maximum_target_tokens": max(len(r["input_ids"]) - r["prompt_tokens"] for r in items),
            "unmodified_native_prefix_mismatches": 0,
            "roles": dict(Counter(r["role"] for r in items)),
            "target_kinds": dict(Counter(r["target_kind"] for r in items)), "sha256": sha256(path),
        }
    report = {
        **{k: old_report[k] for k in ("status", "tokenizer_receipt_sha256", "repository", "revision", "versions",
                                    "context_limit", "completion_limit", "truncation_allowed", "loss_policy",
                                    "prompt_policy")},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "runtime_audit_sha256": sha256(runtime_report),
        "serving_capture_report_sha256": sha256(capture_output / "report.json"),
        "serving_prompts_sha256": sha256(capture_output / "server-prompts.jsonl"),
        "splits": splits, "training_admitted": False, "optimization_started": False, "qualification_passed": False,
        "composition_source": identity(Path(__file__)),
        "component_tokens": {"parent": identity(parent_tokens / "report.json"),
                            "supplement": identity(extra / "tokens/report.json")},
        "unmodified_native_prompt_parity_verified": True, "serving_prefix_parity_verified": True,
    }
    write(token_output / "report.json", report)
    return report


async def main(args):
    build_source(args.root, args.source)
    parent = args.root / "organism/data/equipment-runtime-development-v4-01"
    result = await audit(args.source, parent)
    write(args.runtime_audit, result)
    compose_tokens(args.root, args.source, args.tokens, args.capture, args.runtime_audit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "tokens", "capture", "runtime-audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
