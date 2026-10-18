"""Reconstruct retained native responses independently of export scoring code."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load(path):
    return json.loads(path.read_text())


def reconstruct_response(directory, item, prompt, vocabulary):
    request = item["request"]
    require(request == {
        "prompt": prompt, "n_predict": 1, "n_probs": vocabulary, "temperature": 0,
        "cache_prompt": False, "return_tokens": True, "post_sampling_probs": False,
        "repeat_penalty": 1, "seed": 20261014, "grammar": 'root ::= "a"',
    }, "Native probability request changed")
    name = item["response_file"]
    require(Path(name).name == name and name.endswith(".json.gz"), "Invalid response path")
    with gzip.open(directory / name, "rb") as stream:
        raw = stream.read(24 * 1024**2 + 1)
    require(len(raw) == item["response_bytes"] <= 24 * 1024**2
            and hashlib.sha256(raw).hexdigest() == item["response_sha256"],
            "Raw response identity differs")
    body = json.loads(raw)
    require(body["tokens_evaluated"] == len(prompt) and body["content"] == "a"
            and body["tokens_predicted"] == 1 and body["stop"] is True
            and body["truncated"] is False
            and body["tokens"] == item["generated_tokens"] and len(body["tokens"]) == 1
            and len(body["completion_probabilities"]) == 1, "Native response semantics differ")
    settings = body["generation_settings"]
    require(settings["n_probs"] == vocabulary and settings["post_sampling_probs"] is False
            and settings["grammar"] == 'root ::= "a"' and settings["temperature"] == 0,
            "Native response probability settings differ")
    entries = body["completion_probabilities"][0]["top_logprobs"]
    require(len(entries) == vocabulary, "Incomplete probability vocabulary")
    values = np.full(vocabulary, np.nan, dtype=np.float64)
    for entry in entries:
        token, value = entry["id"], entry["logprob"]
        require(type(token) is int and 0 <= token < vocabulary and np.isnan(values[token])
                and type(value) in (int, float) and math.isfinite(value) and value <= 0,
                "Invalid or duplicate probability entry")
        values[token] = value
    require(np.isfinite(values).all()
            and np.array_equal(values, values.astype(np.float32).astype(np.float64)),
            "Native values do not preserve finite F32 probabilities")
    return values


def compare(reference, observed):
    """Independent formula; raw error is retained before KL normalization."""
    reference = np.asarray(reference, dtype=np.float64)
    require(reference.shape == observed.shape and np.isfinite(reference).all(), "Reference shape differs")
    shifted = reference - reference.max(axis=1, keepdims=True)
    reference = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    normalizers = np.exp(observed).sum(axis=1)
    normalized = observed - np.log(normalizers)[:, None]
    divergence = (np.exp(reference) * (reference - normalized)).sum(axis=1)
    require((divergence >= -1e-10).all(), "Negative divergence")
    unit = float(np.finfo(np.float32).eps / 2)
    product = (observed.shape[1] - 1) * unit
    gamma = product / (1 - product)
    allowance = gamma / (1 - gamma) + np.expm1(8 * unit * (1 + np.abs(observed).max(axis=1)))
    require((np.abs(normalizers - 1) <= allowance).all(), "Native normalization exceeds F32 allowance")
    return {
        "maximum_absolute_log_probability_error": float(np.abs(reference - observed).max()),
        "maximum_normalized_log_probability_error": float(np.abs(reference - normalized).max()),
        "mean_kl_divergence": float(np.maximum(divergence, 0).mean()),
        "per_row_kl_divergence": np.maximum(divergence, 0).tolist(),
        "top_one_agreement": int((reference.argmax(axis=1) == observed.argmax(axis=1)).sum()),
        "raw_normalizers": normalizers.tolist(),
        "normalization_error_allowances": allowance.tolist(),
    }


def pinned_parser(export, gguf_source):
    require(gguf_source is not None, "Pinned local GGUF source is required for tensor readback")
    gguf_source = Path(gguf_source).resolve()
    manifest = load(export / "toolchain.json")
    require(manifest["declared_commit"] == "4f31eedb0ccf546b7e8d6bb243b170f12522f54d",
            "GGUF source revision differs")
    source_prefix = manifest["source_root"] + "/gguf-py/"
    expected_sources = {name[len(source_prefix):]: value for name, value in manifest["files"].items()
                        if name.startswith(source_prefix) and name.endswith(".py")}
    actual_sources = {str(p.relative_to(gguf_source / "gguf-py")): identity(p)
                      for p in (gguf_source / "gguf-py" / "gguf").rglob("*.py")}
    require(len(expected_sources) > 10 and actual_sources == expected_sources,
            "Local GGUF parser differs from pinned export source")
    sys.path.insert(0, str(gguf_source / "gguf-py"))
    import gguf
    require(Path(gguf.__file__).resolve().is_relative_to(gguf_source / "gguf-py"),
            "Unexpected GGUF parser import")
    return gguf, len(actual_sources)


def precision_specification(report):
    if "quantization_precision" not in report:
        require(report.get("schema") != "organism-trained-gguf-pair-export-v2",
                "Versioned export omitted its precision specification")
        return None
    precision = report["quantization_precision"]
    require(isinstance(precision, dict) and isinstance(precision.get("specification"), dict),
            "Export precision specification is invalid")
    expected = {
        "schema": "organism-quantization-precision-v1", "profile": "q4-k-m-io-q8-v1",
        "family": "Q4_K_M",
        "overrides": {"output.weight": "Q8_0", "token_embd.weight": "Q8_0"},
    }
    if precision["specification"].get("profile") == "q4-k-m-io-q8-down-v-q6-v1":
        expected.update(
            schema="organism-quantization-precision-v2", profile="q4-k-m-io-q8-down-v-q6-v1",
            pattern_overrides={r"^blk\.[0-9]+\.(attn_v|ffn_down)\.weight$": "Q6_K"})
    elif precision["specification"].get("profile") == "q4-k-m-io-q8-attn-down-q6-v1":
        expected.update(
            schema="organism-quantization-precision-v2", profile="q4-k-m-io-q8-attn-down-q6-v1",
            pattern_overrides={r"^blk\.[0-9]+\.(attn_q|attn_k|attn_v|attn_output|ffn_down)\.weight$": "Q6_K"})
    require(report.get("schema") == "organism-trained-gguf-pair-export-v2"
            and report["quantization_precision"].get("specification") == expected,
            "Export precision specification or version differs")
    return expected


def precision_readback(export, report, job, gguf_source):
    """Independent tensor inspection; no producer validation code is imported."""
    spec = precision_specification(report)
    roles = (("attn_q", "attn_k", "attn_v", "attn_output", "ffn_down")
             if spec and spec["profile"] == "q4-k-m-io-q8-attn-down-q6-v1" else ("attn_v", "ffn_down"))
    require(spec is not None and job.get("schema") == "organism-cpu-export-job-v2"
            and job.get("quantization_precision") == spec and "calibration" in job,
            "Precision export does not match its prepared job")
    gguf, parser_files = pinned_parser(export, gguf_source)
    models = {}
    for arm in ("stock", "trained"):
        path = export / (arm + "-q4.gguf")
        require(identity(path) == {k: report["models"][arm + "-q4"][k] for k in ("bytes", "sha256")},
                "Precision model identity differs")
        reader = gguf.GGUFReader(path)
        require(reader.fields["general.file_type"].contents() == int(gguf.LlamaFileType.MOSTLY_Q4_K_M),
                "Actual model is outside the Q4_K_M family")
        types = {t.name: t.tensor_type.name for t in reader.tensors}
        require(len(types) == len(reader.tensors)
                and all(types.get(name) == kind for name, kind in spec["overrides"].items()),
                "Actual precision overrides differ")
        body_count = 0
        overridden = set()
        for tensor in reader.tensors:
            if tensor.name in spec["overrides"]:
                require(len(tensor.shape) == 2, "Override is not a matrix")
            elif len(tensor.shape) == 1:
                require(tensor.name.endswith("_norm.weight") and tensor.tensor_type.name == "F32",
                        "Actual normalization precision differs")
            else:
                require(len(tensor.shape) == 2 and tensor.name.startswith("blk.")
                        and tensor.tensor_type.name in ("Q4_K", "Q6_K"), "Actual body precision differs")
                if "pattern_overrides" in spec:
                    parts = tensor.name.split(".")
                    require(len(parts) == 4 and parts[0] == "blk" and parts[1].isdigit()
                            and parts[2] in ("attn_k", "attn_output", "attn_q", "attn_v",
                                             "ffn_down", "ffn_gate", "ffn_up")
                            and parts[3] == "weight", "Unexpected actual body weight")
                    selected = parts[2] in roles
                    require(tensor.tensor_type.name == ("Q6_K" if selected else "Q4_K"),
                            "Actual explicit body precision differs")
                    if selected:
                        overridden.add(tensor.name)
                body_count += 1
        require(body_count > 0, "Missing quantized body")
        if "pattern_overrides" in spec:
            field = reader.fields.get("granite.block_count")
            layers = field.contents() if field is not None else None
            require(type(layers) is int and 1 <= layers <= 128
                    and overridden == {f"blk.{layer}.{role}.weight" for layer in range(layers) for role in roles},
                    "Actual explicit body override coverage differs")
        models[arm] = {"specification": spec, "tensor_types_by_name": types}
    require(models["stock"] == models["trained"]
            and models == report["quantization_precision"]["models"],
            "Matched tensor precision or reported inventory differs")
    return {"specification": spec, "parser_files_verified": parser_files, "models": models}


def calibration_readback(export, report, prepared, gguf_source):
    gguf, parser_files = pinned_parser(export, gguf_source)
    spec = precision_specification(report)

    plan = load(export / "calibration-plan.json")
    require(plan == load(prepared / "plan.json") == report["calibration"]["plan"],
            "Calibration plan changed")
    require(identity(export / "calibration.txt") == plan["text"]
            and (export / "calibration.txt").read_bytes() == (prepared / "calibration.txt").read_bytes(),
            "Calibration text changed")
    require(plan["validation_used"] is False and plan["qualification_used"] is False
            and plan["optimizer_steps"] == 0 and plan["quality_selection_performed"] is False,
            "Calibration scope differs")
    require(plan.get("schema") in ("organism-training-only-calibration-v1",
                                  "organism-training-only-calibration-v2",
                                  "organism-training-only-calibration-v3",
                                  "organism-training-only-calibration-v4"),
            "Unsupported calibration contract")
    process_output = plan["schema"] != "organism-training-only-calibration-v1"
    if plan["schema"] == "organism-training-only-calibration-v3":
        require(plan.get("source_selection") == {
            "kind": "unchanged_v4_training_prefix", "rows": 1680, "bytes": 26130191,
            "sha256": "572786faeed6132b86ad1fdef1b527cbaedefd41dbbcbf6fb341c9d809793d6a",
        }, "Matched calibration parent binding differs")
    if plan["schema"] == "organism-training-only-calibration-v4":
        require(plan.get("source_selection") == {
            "kind": "first_lineage_frozen_training", "rows": 716, "bytes": 10371097,
            "sha256": "9277a8e0dfc5d0275aa9c95ff89c976a16f56d25a8047bc70c7696be0518e5a3",
            "roles": ["equipment", "reproduction", "diagnostic"],
        } and plan["training_tokens"] == {
            "bytes": 10371097,
            "sha256": "9277a8e0dfc5d0275aa9c95ff89c976a16f56d25a8047bc70c7696be0518e5a3",
        }, "First-lineage calibration training population differs")
        selected = plan["selected_rows"]
        require(len(selected) == 12 and len({row["case_id"] for row in selected}) == 12
                and [row["role"] for row in selected] == ["equipment", "reproduction", "diagnostic"] * 4
                and all(row["source_split"] == "train" for row in selected),
                "First-lineage calibration role coverage or split differs")
    if plan["schema"] in ("organism-training-only-calibration-v3", "organism-training-only-calibration-v4"):
        current = plan.get("current_reference_case_ids")
        require(isinstance(current, list) and len(current) == 3
                and all(isinstance(value, str) and value for value in current)
                and current == sorted(set(current))
                and current == sorted({r["case_id"] for r in report["development_reference_cases"]})
                and not set(current) & {r["case_id"] for r in plan["selected_rows"]},
                "Matched calibration overlaps current numerical references")
        require(plan["training_admission"] == report["admission"],
                "Matched calibration belongs to different training")
    settings = plan["settings"]
    require(settings.get("process_output") is process_output and settings == {
        "context_tokens": 4096, "chunks": 4, "cases_per_role": 4,
        "batch_tokens": 512, "microbatch_tokens": 128, "threads": 4,
        "cache_type": "f32", "flash_attention": "off", "process_output": process_output,
        "quantization": "Q4_K_M", "selection_salt": "organism-q4-calibration-v1",
    }, "Calibration settings or version differ")
    matrices = {}
    for arm in ("stock", "trained"):
        path = export / (arm + "-imatrix.gguf")
        receipt = report["calibration"]["matrices"][arm]
        require(identity(path) == {k: receipt[k] for k in ("bytes", "sha256")}
                and path.stat().st_size <= 16 * 1024**2, "Matrix identity differs")
        reader = gguf.GGUFReader(path)
        fields = {name: field.contents() for name, field in reader.fields.items()}
        require(fields["general.type"] == "imatrix"
                and fields["imatrix.chunk_count"] == settings["chunks"] == 4
                and fields["imatrix.chunk_size"] == settings["context_tokens"] == 4096,
                "Matrix coverage differs")
        model = gguf.GGUFReader(export / (arm + "-q4.gguf"))
        weights = {t.name: int(t.shape[0]) for t in model.tensors
                   if len(t.shape) == 2 and (t.name.startswith("blk.")
                                             or process_output and t.name == "output.weight")}
        require(not process_output or "output.weight" in weights, "Output projection is missing")
        expected = {name + suffix: width if suffix == ".in_sum2" else 1
                    for name, width in weights.items() for suffix in (".in_sum2", ".counts")}
        require(len(reader.tensors) == len(expected)
                and {t.name for t in reader.tensors} == set(expected), "Matrix weight inventory differs")
        inventory = {}
        for tensor in reader.tensors:
            data = np.asarray(tensor.data).reshape(-1)
            require(tensor.tensor_type == gguf.GGMLQuantizationType.F32
                    and tensor.n_elements == expected[tensor.name]
                    and np.isfinite(data).all() and (data >= 0).all() and (data > 0).any(),
                    "Invalid matrix observations")
            if tensor.name.endswith(".counts"):
                require((data == np.floor(data)).all(), "Fractional matrix count")
            inventory[tensor.name] = {
                "elements": int(tensor.n_elements), "minimum": float(data.min()),
                "maximum": float(data.max()),
            }
        require(inventory == receipt["tensors"], "Matrix summary differs")
        text = fields["imatrix.datasets"]
        require(type(text) is list and len(text) == 1 and Path(text[0]).name == "calibration.txt",
                "Matrix dataset binding differs")
        remote = Path(text[0]).parent
        collectors = [c for c in report["commands"] if Path(c["argv"][0]).name == "llama-imatrix"
                      and len(c["argv"]) > 2 and c["argv"][2] == str(remote / (arm + "-f32.gguf"))]
        require(len(collectors) == 1 and collectors[0]["return_code"] == 0,
                "Missing successful unique calibration command")
        expected_collection = [
            "--model", str(remote / (arm + "-f32.gguf")), "--file", text[0],
            "--output-file", str(remote / (arm + "-imatrix.gguf")), "--output-format", "gguf",
            "--ctx-size", "4096", "--chunks", "4", "--batch-size", "512", "--ubatch-size", "128",
            "--threads", "4", "--threads-batch", "4", "--gpu-layers", "0", "--device", "none",
            "--no-kv-offload", "--no-op-offload", "--cache-type-k", "f32", "--cache-type-v", "f32",
            "--flash-attn", "off", "--no-ppl", "--parse-special",
            "--save-frequency", "0", "--output-frequency", "999999",
        ]
        if process_output:
            expected_collection.append("--process-output")
        require(collectors[0]["argv"][1:] == expected_collection, "Calibration command binding differs")
        commands = [c for c in report["commands"] if Path(c["argv"][0]).name == "llama-quantize"
                    and str(c["argv"][-3]).endswith("/" + arm + "-q4.gguf")]
        require(len(commands) == 1 and commands[0]["return_code"] == 0,
                "Missing successful unique calibrated quantization command")
        argv = commands[0]["argv"]
        precision_args = ([] if spec is None else
                          ["--output-tensor-type", "q8_0", "--token-embedding-type", "q8_0"])
        if spec is not None and "pattern_overrides" in spec:
            pattern = (r"^blk\.[0-9]+\.(attn_q|attn_k|attn_v|attn_output|ffn_down)\.weight$"
                       if spec["profile"] == "q4-k-m-io-q8-attn-down-q6-v1"
                       else r"^blk\.[0-9]+\.(attn_v|ffn_down)\.weight$")
            precision_args += ["--tensor-type", pattern + "=q6_k"]
        require(argv[1:] == ["--imatrix", str(remote / (arm + "-imatrix.gguf")),
                            *precision_args,
                            str(remote / (arm + "-f32.gguf")), str(remote / (arm + "-q4.gguf")),
                            "Q4_K_M", "4"], "Quantization matrix binding differs")
        matrices[arm] = {**identity(path), "observed_weights": len(weights),
                         "chunks": fields["imatrix.chunk_count"],
                         "context_tokens": fields["imatrix.chunk_size"]}
    return {"parser_files_verified": parser_files,
            "plan": identity(export / "calibration-plan.json"),
            "text": identity(export / "calibration.txt"), "matrices": matrices}


def audit(directory, *, calibration=None, gguf_source=None, export_job=None):
    directory = Path(directory)
    export, merge = directory / "workload/export", directory / "workload/merge"
    report, merged = load(export / "report.json"), load(merge / "report.json")
    retention = load(directory / "retention.json")
    checked = {}

    def check_file(path):
        relative = str(path.relative_to(directory))
        require(not path.is_symlink() and identity(path) == retention["files"][relative],
                "Retained file identity differs: " + relative)
        checked[relative] = retention["files"][relative]

    # Verify all retained files, including both real model artifacts, before parsing.
    for name in retention["files"]:
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Invalid retained path")
        check_file(directory / name)
    require(report["limits"]["f32_maximum_log_probability_error"] == 0.002
            and report["limits"]["quantized_maximum_mean_kl"] == 0.10
            and merged["maximum_merge_logit_error"] == 0.0002, "Numerical limit changed")
    require(report["prompts"] == merged["prompts"] and len(report["prompts"]) == 3,
            "Reference prompts differ")
    require(all(c["source_split"] == "train" for c in report["development_reference_cases"]),
            "Unexpected reference split")
    arrays = {name: np.load(merge / (name + "-reference-logits.npy"), allow_pickle=False)
              for name in ("base", "adapter", "merged")}
    require(all(a.shape == arrays["base"].shape and a.dtype in (np.dtype("float32"), np.dtype("float64"))
                and np.isfinite(a).all() for a in arrays.values()), "Invalid reference arrays")
    merge_error = float(np.abs(arrays["adapter"].astype(np.float64)
                               - arrays["merged"].astype(np.float64)).max())
    require(abs(merge_error - merged["merge"]["comparison"]["maximum_absolute_logit_error"]) <= 1e-12,
            "Merge summary differs")
    arms, failures = {}, []
    for label in ("stock-f32", "trained-f32", "stock-q4", "trained-q4"):
        serving = load(export / label / "report.json")
        require(serving["failure"] is None and len(serving["requests"]) == 3
                and serving["shutdown"]["owned_server_stopped"] is True
                and serving["shutdown"]["server_return_code"] == 0, "Serving did not complete")
        rows = np.stack([reconstruct_response(export / label, item, prompt, arrays["base"].shape[1])
                         for item, prompt in zip(serving["requests"], report["prompts"])])
        require(np.array_equal(rows, np.load(export / (label + "-log-probabilities.npy"), allow_pickle=False)),
                "Response probabilities differ from saved array")
        comparison = compare(arrays["base" if label.startswith("stock") else "merged"], rows)
        stored = report["comparisons"][label]
        for key, value in comparison.items():
            original = stored["native_probability_accounting"][key] if key in (
                "raw_normalizers", "normalization_error_allowances") else stored[key]
            require(np.allclose(value, original, atol=1e-10, rtol=0), "Comparison differs: " + key)
        passed = (comparison["maximum_absolute_log_probability_error"] <= 0.002 if label.endswith("f32")
                  else comparison["mean_kl_divergence"] <= 0.10)
        if not passed:
            failures.append(label + (": F32 raw log-probability error" if label.endswith("f32")
                                     else ": Q4 mean divergence"))
        arms[label] = {**comparison, "passes_unchanged_limit": passed, "raw_responses": len(rows)}
    require(failures == report["numerical_failures"], "Failure inventory differs")
    result = {
        "schema": "organism-export-response-audit-v1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "source": identity(Path(__file__)), "retention": identity(directory / "retention.json"),
        "checked_files": len(checked), "merge_error": merge_error, "arms": arms,
        "reconstruction_passed": True, "numerical_gates_passed": merge_error <= 0.0002 and not failures,
        "qualification_passed": False,
    }
    if calibration is not None:
        result["calibration"] = calibration_readback(export, report, Path(calibration), gguf_source)
    if precision_specification(report) is not None:
        require(calibration is not None and export_job is not None,
                "Precision audit requires the prepared calibration and export job")
        job = load(Path(export_job))
        require(job["calibration"]["plan"] == identity(Path(calibration) / "plan.json")
                and job["calibration"]["text"] == identity(Path(calibration) / "calibration.txt"),
                "Precision job calibration binding differs")
        result["quantization_precision"] = precision_readback(export, report, job, gguf_source)
        result["export_job"] = identity(Path(export_job))
    else:
        require(export_job is None, "Unexpected precision job for an ordinary export")
    require(not report["passed"] or result["numerical_gates_passed"], "False reported numerical success")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--gguf-source", type=Path)
    parser.add_argument("--export-job", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output already exists")
    result = audit(args.directory, calibration=args.calibration, gguf_source=args.gguf_source,
                   export_job=args.export_job)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
