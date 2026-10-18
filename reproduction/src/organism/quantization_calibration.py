"""Training-only calibration for a separately versioned Q4_K_M export.

Calibration observes activations; it does not update model parameters. These
helpers do not admit a model, change numerical limits, or launch cloud work.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .gguf_export import canonical, identity, run_command
from .training_inputs import require, validate_row


ROLES = ("equipment", "equipment_inherited", "reproduction", "diagnostic")
FIRST_LINEAGE_ROLES = ("equipment", "reproduction", "diagnostic")
SETTINGS = {
    "context_tokens": 4096, "chunks": 4, "cases_per_role": 4,
    "batch_tokens": 512, "microbatch_tokens": 128, "threads": 4,
    "cache_type": "f32", "flash_attention": "off", "process_output": False,
    "quantization": "Q4_K_M", "selection_salt": "organism-q4-calibration-v1",
}
V4_PREFIX = {
    "kind": "unchanged_v4_training_prefix", "rows": 1680, "bytes": 26130191,
    "sha256": "572786faeed6132b86ad1fdef1b527cbaedefd41dbbcbf6fb341c9d809793d6a",
}
FIRST_LINEAGE_SOURCE = {
    "kind": "first_lineage_frozen_training", "rows": 716, "bytes": 10371097,
    "sha256": "9277a8e0dfc5d0275aa9c95ff89c976a16f56d25a8047bc70c7696be0518e5a3",
    "roles": list(FIRST_LINEAGE_ROLES),
}


def calibration_rows(admitted, *, preserve_v4_selection=False):
    """Keep the original calibration population for the explicit v5 comparison."""
    require(type(preserve_v4_selection) is bool, "Invalid calibration source selection")
    path = admitted["paths"]["tokens_train"]
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    require(len(lines) == admitted["recipe"]["counts"]["train"], "Training row count changed")
    rows = [json.loads(line) for line in lines]
    for row in rows:
        validate_row(row, vocab_size=admitted["config"]["vocab_size"],
                     eos_token_id=admitted["config"]["eos_token_id"])
    if preserve_v4_selection:
        require(admitted["recipe"].get("corpus") == "refinement_runtime" and len(rows) == 2100,
                "Original calibration selection requires the explicit v5 refinement corpus")
        prefix = b"".join(lines[:V4_PREFIX["rows"]])
        require(len(prefix) == V4_PREFIX["bytes"]
                and hashlib.sha256(prefix).hexdigest() == V4_PREFIX["sha256"],
                "Original calibration training prefix changed")
        rows = rows[:V4_PREFIX["rows"]]
    return rows


def select_rows(rows, *, roles=ROLES):
    """Pick full final training targets evenly by role, without outcome scores."""
    require(roles in (ROLES, FIRST_LINEAGE_ROLES)
            and rows and all(row["role"] in roles for row in rows), "Unexpected calibration roles")
    reference_indices = (0, max(range(len(rows)), key=lambda i: rows[i]["prompt_tokens"]), len(rows) - 1)
    excluded = sorted({rows[i]["case_id"] for i in reference_indices})
    cases, seen = {}, set()
    for row in rows:
        key = (row["case_id"], row["decision_index"])
        require(key not in seen and type(row["decision_index"]) is int and row["decision_index"] >= 0,
                "Duplicate or invalid training decision")
        seen.add(key)
        old = cases.get(row["case_id"])
        require(old is None or old["role"] == row["role"], "Training case changed roles")
        if old is None or old["decision_index"] < row["decision_index"]:
            cases[row["case_id"]] = row
    by_role = defaultdict(list)
    for case_id, row in cases.items():
        if case_id not in excluded:
            by_role[row["role"]].append(row)
    for role in roles:
        by_role[role].sort(key=lambda row: (
            hashlib.sha256((SETTINGS["selection_salt"] + "\0" + row["case_id"]).encode()).hexdigest(),
            row["case_id"]))
        require(len(by_role[role]) >= SETTINGS["cases_per_role"], "Too few independent training cases")
    selected = [by_role[role][index]
                for index in range(SETTINGS["cases_per_role"]) for role in roles]
    require(all(row["target_kind"] == "final" for row in selected), "Calibration omitted a final response")
    return selected, excluded


def calibration_payload(admitted, tokenizer, *, process_output=False, preserve_v4_selection=False):
    """Reproduce text from the exact admitted training tokens, never validation."""
    require(type(process_output) is bool, "Invalid output-calibration selection")
    settings = {**SETTINGS, "process_output": process_output}
    path = admitted["paths"]["tokens_train"]
    rows = calibration_rows(admitted, preserve_v4_selection=preserve_v4_selection)
    require(not preserve_v4_selection or process_output is True,
            "The matched v4 comparison requires output-inclusive calibration")
    first_lineage = admitted["recipe"].get("corpus") == "first_lineage_runtime"
    if first_lineage:
        require(process_output and not preserve_v4_selection
                and len(rows) == FIRST_LINEAGE_SOURCE["rows"]
                and identity(path) == {key: FIRST_LINEAGE_SOURCE[key] for key in ("bytes", "sha256")},
                "First-lineage calibration requires its exact frozen training population and output coverage")
    selected, excluded = select_rows(rows, roles=FIRST_LINEAGE_ROLES if first_lineage else ROLES)
    texts, inventory = [], []
    for row in selected:
        text = tokenizer.decode(row["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        require(tokenizer.encode(text, add_special_tokens=False) == row["input_ids"],
                "Training text does not round-trip to its admitted tokens")
        texts.append(text)
        inventory.append({
            "case_id": row["case_id"], "role": row["role"], "decision_index": row["decision_index"],
            "source_split": "train", "source_row_sha256": hashlib.sha256(canonical(row)).hexdigest(),
            "source_tokens": len(row["input_ids"]), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        })
    # Separators are explicit calibration formatting; source targets are unchanged.
    text = "\n\n".join(texts) + "\n"
    ids = tokenizer.encode(text, add_special_tokens=False)
    require(SETTINGS["context_tokens"] * SETTINGS["chunks"] <= len(ids) <= 131072
            and len(text.encode()) <= 2 * 1024**2, "Calibration data exceeds or undershoots its bound")
    plan = {
        "schema": ("organism-training-only-calibration-v4" if first_lineage else
                   "organism-training-only-calibration-v3" if preserve_v4_selection else
                   "organism-training-only-calibration-v2" if process_output
                   else "organism-training-only-calibration-v1"),
        "training_admission": admitted["receipt"], "training_tokens": identity(path),
        "settings": settings, "selected_rows": inventory, "excluded_reference_case_ids": excluded,
        "tokenization": {"add_special_tokens": False, "parse_special": True,
                         "token_count": len(ids), "token_ids_sha256": hashlib.sha256(canonical(ids)).hexdigest()},
        "format": "Decoded complete final training targets, separated by two newlines; one final newline",
        "validation_used": False, "qualification_used": False,
        "optimizer_steps": 0, "quality_selection_performed": False, "qualification_passed": False,
    }
    if preserve_v4_selection:
        plan["source_selection"] = dict(V4_PREFIX)
        full = [json.loads(line) for line in path.read_text().splitlines()]
        references = (0, max(range(len(full)), key=lambda i: full[i]["prompt_tokens"]), len(full) - 1)
        plan["current_reference_case_ids"] = sorted({full[i]["case_id"] for i in references})
        require(not set(plan["current_reference_case_ids"]) & {row["case_id"] for row in selected},
                "Matched calibration overlaps a current export reference")
    if first_lineage:
        plan["source_selection"] = {**FIRST_LINEAGE_SOURCE, "roles": list(FIRST_LINEAGE_ROLES)}
        plan["current_reference_case_ids"] = excluded
    return text.encode(), plan


def prepare_calibration(admitted, tokenizer, output, *, process_output=False, preserve_v4_selection=False):
    output = Path(output)
    require(not output.exists(), "Calibration output must be new")
    text, plan = calibration_payload(admitted, tokenizer, process_output=process_output,
                                     preserve_v4_selection=preserve_v4_selection)
    output.mkdir()
    (output / "calibration.txt").write_bytes(text)
    plan["text"] = identity(output / "calibration.txt")
    (output / "plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    return plan


def verify_calibration(admitted, tokenizer, directory):
    directory = Path(directory)
    require(not directory.is_symlink()
            and {path.name for path in directory.iterdir()} == {"calibration.txt", "plan.json"}
            and all(path.is_file() and not path.is_symlink() for path in directory.iterdir()),
            "Calibration inventory changed")
    actual = json.loads((directory / "plan.json").read_text())
    require(actual.get("schema") in ("organism-training-only-calibration-v1",
                                    "organism-training-only-calibration-v2",
                                    "organism-training-only-calibration-v3",
                                    "organism-training-only-calibration-v4"),
            "Unsupported calibration contract")
    preserved = actual["schema"] == "organism-training-only-calibration-v3"
    require(not preserved or actual.get("source_selection") == V4_PREFIX,
            "Matched calibration lacks its exact original source binding")
    if actual["schema"] == "organism-training-only-calibration-v4":
        require(actual.get("source_selection") == FIRST_LINEAGE_SOURCE,
                "First-lineage calibration lacks its frozen training binding")
    process_output = actual["schema"] != "organism-training-only-calibration-v1"
    require(actual.get("settings", {}).get("process_output") is process_output,
            "Calibration version and output selection differ")
    text, expected = calibration_payload(admitted, tokenizer, process_output=process_output,
                                         preserve_v4_selection=preserved)
    require((directory / "calibration.txt").read_bytes() == text, "Calibration text changed")
    expected["text"] = identity(directory / "calibration.txt")
    require(actual == expected, "Calibration plan changed")
    return actual


def imatrix_command(tools, model, text, target, *, context_tokens=4096, chunks=4, process_output=False):
    require(type(context_tokens) is int and 512 <= context_tokens <= 4096
            and context_tokens % 512 == 0 and type(chunks) is int and 2 <= chunks <= 4,
            "Invalid calibration command bounds")
    require(type(process_output) is bool, "Invalid output-calibration selection")
    name = tools.manifest.get("imatrix")
    require(name in tools.manifest["files"] and Path(name).name == "llama-imatrix"
            and not tools.path(name).is_symlink()
            and identity(tools.path(name)) == tools.manifest["files"][name],
            "Calibration executable is not bound")
    require(not Path(target).exists() and Path(model).is_file() and Path(text).is_file(),
            "Missing calibration input or existing output")
    command = [
        str(tools.path(name)), "--model", str(model), "--file", str(text), "--output-file", str(target),
        "--output-format", "gguf", "--ctx-size", str(context_tokens), "--chunks", str(chunks),
        "--batch-size", "512", "--ubatch-size", "128", "--threads", "4", "--threads-batch", "4",
        "--gpu-layers", "0", "--device", "none", "--no-kv-offload", "--no-op-offload",
        "--cache-type-k", "f32", "--cache-type-v", "f32", "--flash-attn", "off",
        "--no-ppl", "--parse-special", "--save-frequency", "0", "--output-frequency", "999999",
    ]
    if process_output:
        command.append("--process-output")
    return command


def inspect_imatrix(path, config, text, *, context_tokens=4096, chunks=4, process_output=False):
    import gguf
    import numpy as np

    path = Path(path)
    require(type(process_output) is bool, "Invalid output-calibration selection")
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 16 * 1024**2,
            "Importance matrix file exceeds its bound")
    reader = gguf.GGUFReader(path)
    fields = {name: field.contents() for name, field in reader.fields.items()}
    require(fields.get("general.type") == "imatrix"
            and fields.get("imatrix.chunk_count") == chunks
            and fields.get("imatrix.chunk_size") == context_tokens
            and fields.get("imatrix.datasets") == [str(text)], "Importance matrix metadata differs")
    expected = {}
    for layer in range(config["num_hidden_layers"]):
        for weight in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
            name = f"blk.{layer}.{weight}.weight"
            width = config["intermediate_size"] if weight == "ffn_down" else config["hidden_size"]
            expected[name + ".in_sum2"] = width
            expected[name + ".counts"] = 1
    if process_output:
        expected["output.weight.in_sum2"] = config["hidden_size"]
        expected["output.weight.counts"] = 1
    require(len(reader.tensors) == len(expected)
            and {tensor.name for tensor in reader.tensors} == set(expected),
            "Importance matrix omitted or added a weight")
    inventory = {}
    for tensor in reader.tensors:
        data = np.asarray(tensor.data).reshape(-1)
        require(tensor.tensor_type == gguf.GGMLQuantizationType.F32
                and tensor.n_elements == expected[tensor.name]
                and np.isfinite(data).all() and (data >= 0).all() and (data > 0).any(),
                "Invalid importance matrix activation data")
        if tensor.name.endswith(".counts"):
            require((data == np.floor(data)).all(), "Importance matrix count is fractional")
        inventory[tensor.name] = {"elements": int(tensor.n_elements), "minimum": float(data.min()),
                                  "maximum": float(data.max())}
    return {**identity(path), "chunks": chunks, "context_tokens": context_tokens,
            "tensors": inventory, "qualification_passed": False}


def collect_imatrix(tools, model, text, target, config, log, records, *,
                    context_tokens=4096, chunks=4, process_output=False, timeout_seconds=1200):
    before = {"model": identity(Path(model)), "text": identity(Path(text))}
    command = imatrix_command(tools, model, text, target, context_tokens=context_tokens, chunks=chunks,
                              process_output=process_output)
    run_command(command, Path(log), records, timeout_seconds=timeout_seconds,
                library_paths=tuple(tools.path(name) for name in tools.manifest["library_paths"]),
                output_path=Path(target), maximum_output_bytes=16 * 1024**2)
    require(before == {"model": identity(Path(model)), "text": identity(Path(text))},
            "Calibration input changed")
    return inspect_imatrix(target, config, text, context_tokens=context_tokens, chunks=chunks,
                           process_output=process_output)
