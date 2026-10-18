"""Package a verified trained export with both pinned native runtime variants."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .build_development_capsule import PIN_FILE, _assemble, validate_inputs
from .capsule import canonical, validate_body
from .export_task_inputs import ARMS, checked_pair, identity, require


def bind_inputs(workspace, stock_pins, admission, pair, training_recipe):
    """Replace only the model input after complete export admission."""
    workspace = Path(workspace).resolve()
    original = validate_inputs(stock_pins)
    require(identity(training_recipe) == admission["recipe"], "Training recipe identity differs")
    base = json.loads(Path(training_recipe).read_text())["base"]
    require(base["repository"] == original["model"]["source_model"]
            and base["revision"] == original["model"]["source_revision"]
            and base["expected_parameters"] == original["model"]["parameters"]
            and original["model"]["quantization"] == "Q4_K_M",
            "Capsule source model, revision or quantization differs from the admitted export")
    require(pair["numerical_gates_passed"] is True and pair["qualification_passed"] is False,
            "Trained capsule requires passed numerical export evidence")
    require(pair["total_stored_parameters"] == admission["base_parameters"]
            == original["model"]["parameters"], "Trained capsule parameter identity differs")
    model = pair["models"][ARMS[1]]
    source = Path(model["path"])
    require(source.is_absolute() and source.resolve().is_relative_to(workspace)
            and not source.is_symlink(), "Trained model must be a regular workspace input")
    require(identity(source) == {key: model[key] for key in ("bytes", "sha256")},
            "Trained model changed after export admission")
    pins = deepcopy(stock_pins)
    body = deepcopy(original)
    old_path = pins["model"]["path"]
    model_path = "model/granite-trained-Q4_K_M.gguf"
    pins["schema"] = "organism-trained-capsule-inputs-v1"
    pins["model"].update(path=model_path, trained_for_organism=True)
    pins["files"].pop(old_path)
    pins["files"][model_path] = {
        "source": str(source.relative_to(workspace)), **{key: model[key] for key in ("bytes", "sha256")},
    }
    body["model"] = deepcopy(pins["model"])
    body["files"].pop(old_path)
    body["files"][model_path] = {key: model[key] for key in ("bytes", "sha256")}
    provenance = {
        "schema": "organism-trained-capsule-provenance-v1",
        "training_admission": admission, "export_audit": pair["audit"], "export_workload": pair["workload"],
        "model": {key: model[key] for key in ("bytes", "sha256")},
        "total_stored_parameters": pair["total_stored_parameters"],
        "optimizer_steps_during_export": 0, "numerical_gates_passed": True,
        "task_quality_verified": False, "native_execution_verified": False, "qualification_passed": False,
    }
    if "quantization_precision" in pair:
        from .quantization_precision import validate_specification
        validate_specification(pair["quantization_precision"])
        provenance["schema"] = "organism-trained-capsule-provenance-v2"
        provenance.update({name: deepcopy(pair[name]) for name in (
            "quantization_precision", "response_audit", "export_job", "export_report")})
    pins["trained_model_provenance"] = provenance
    validate_body(body, require_trained=True)
    return pins, body, provenance


def build(workspace, output, *, export_directory, admission_path, training_recipe, copy_mode="bytes"):
    workspace, output = Path(workspace).resolve(), Path(output).absolute()
    admission_path = Path(admission_path).resolve()
    admission = json.loads(admission_path.read_text())
    pair = checked_pair(export_directory, admission)
    pins_path = workspace / "organism" / PIN_FILE
    stock_pins = json.loads(pins_path.read_text())
    pins, body, provenance = bind_inputs(workspace, stock_pins, admission, pair, training_recipe)
    receipt = _assemble(workspace, output, pins, canonical(pins), body,
                        copy_mode=copy_mode, trained_provenance=provenance)
    receipt.update(trained_builder=identity(Path(__file__)), original_stock_pins=identity(pins_path),
                   training_admission=identity(admission_path), export_pair=pair)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-directory", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--training-recipe", type=Path, required=True)
    parser.add_argument("--copy-mode", choices=("bytes", "mac_clone"), default="bytes")
    args = parser.parse_args()
    result = build(Path(__file__).resolve().parents[1], args.output,
                   export_directory=args.export_directory, admission_path=args.admission,
                   training_recipe=args.training_recipe, copy_mode=args.copy_mode)
    print(json.dumps({key: result[key] for key in (
        "release_id", "payload_components", "payload_bytes", "trained_for_organism", "qualification_passed")}))
