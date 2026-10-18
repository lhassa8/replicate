"""Create an unqualified F32 merge from a completed, admitted training worker.

This offline stage does not quantize, upload, sign a device capsule, or qualify
a model. The full native base must already be present on the export host.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from .adapter_export import admit_training_export, identity, merge_adapter
from .training_inputs import load, offline_environment, require, validate_row
from .train_specialist import prepare_loading_view, verify_native_base


def run(workspace, worker_output, base, output, *, device="cpu"):
    offline_environment()
    workspace, worker_output, base, output = (Path(path).resolve()
                                               for path in (workspace, worker_output, base, output))
    admitted = admit_training_export(workspace, worker_output)
    recipe, paths, config = admitted["recipe"], admitted["paths"], admitted["config"]
    require(not output.exists() and not output.is_relative_to(base) and not output.is_relative_to(workspace)
            and not output.is_relative_to(worker_output), "Output would overwrite an input bundle")
    require(shutil.disk_usage(output.parent).free >= recipe["base"]["expected_parameters"] * 4 + 1024**3,
            "Insufficient scratch space for a full-size F32 merge")
    metadata = load(paths["base_metadata"])
    native = verify_native_base(base, metadata, recipe, paths["loader_index"])
    # verify_native_base hashes every complete source file before inspecting its
    # tensor headers. The lower-level header inspector cannot set this flag.
    native["full_native_payload_verified"] = True
    rows = [json.loads(line) for line in paths["tokens_train"].read_text().splitlines()]
    require(len(rows) == recipe["counts"]["train"], "Prepared development row count changed")
    indices = list(dict.fromkeys((0, max(range(len(rows)), key=lambda index: rows[index]["prompt_tokens"]),
                                 len(rows) - 1)))
    prompts, cases = [], []
    for index in indices:
        row = rows[index]
        validate_row(row, vocab_size=config["vocab_size"], eos_token_id=config["eos_token_id"])
        prompts.append(row["input_ids"][:row["prompt_tokens"]])
        cases.append({"case_id": row["case_id"], "decision_index": row["decision_index"],
                      "prompt_tokens": row["prompt_tokens"], "source_split": "train"})
    output.mkdir(exist_ok=False)
    report = {
        "schema": "organism-trained-adapter-merge-run-v1", "started_at": datetime.now(timezone.utc).isoformat(),
        "passed": False, "failure": None, "admission": admitted["receipt"], "native_base": native,
        "development_reference_cases": cases, "prompts": prompts, "maximum_merge_logit_error": 0.0002,
        "device": device, "optimizer_steps_this_stage": 0, "deployment_export_complete": False,
        "quality_selection_performed": False, "qualification_passed": False,
        "source_hashes": {name: identity(Path(__file__).with_name(name))
                          for name in ("merge_trained_adapter.py", "adapter_export.py", "export_math.py")},
    }
    started = time.monotonic()
    (output / "request.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        view = prepare_loading_view(base, metadata, paths["loader_index"], output / "base-loading-view")
        references, report["merge"] = merge_adapter(
            view, admitted["adapter_directory"], config, recipe["adapter"], output / "merged", prompts,
            expected_parameters=recipe["base"]["expected_parameters"],
            maximum_logit_error=report["maximum_merge_logit_error"], device=device,
        )
        import numpy as np
        report["references"] = {}
        for label, values in references.items():
            path = output / (label + "-reference-logits.npy")
            np.save(path, values, allow_pickle=False)
            report["references"][path.name] = identity(path)
        # Carry the original pinned tokenizer, not a saved host path or trainer pickle.
        tokenizer = paths["tokenizer_receipt"].parent
        for item in load(paths["tokenizer_receipt"])["files"]:
            if item["name"] == "config.json":
                continue
            source = tokenizer / item["name"]
            require(identity(source) == {key: item[key] for key in ("bytes", "sha256")}, "Tokenizer input changed")
            shutil.copyfile(source, output / "merged" / item["name"])
        report["merged_files"] = {path.name: identity(path)
                                  for path in (output / "merged").iterdir() if path.is_file()}
        # Recheck original native weights and returned adapter after the operation.
        verify_native_base(base, metadata, recipe, paths["loader_index"])
        final_admission = admit_training_export(workspace, worker_output)
        require(final_admission["receipt"] == admitted["receipt"], "Training inputs changed during export")
        report["original_inputs_unchanged"] = True
        report["passed"] = True
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["seconds"] = round(time.monotonic() - started, 3)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "worker-output", "base", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    result = run(args.workspace, args.worker_output, args.base, args.output, device=args.device)
    print(json.dumps({name: result[name] for name in ("passed", "seconds", "deployment_export_complete",
                                                     "qualification_passed")}, indent=2))
