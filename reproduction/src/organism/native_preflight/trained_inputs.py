"""Prepare a trained native candidate catalog; never create a launch kit or device."""

import argparse
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from .. import device_resources, device_storage, network_budget
from ..capsule import canonical, verify_capsule_directory, verify_manifest
from ..safe_archive import safe_relative
from .startup_common import MAX_CATALOG, validate_payload_catalog


SCHEMA = "organism-native-trained-candidate-v1"
PROVENANCE_MEMBER = "model/TRAINING-AND-EXPORT.json"
MAX_NOTICES = 16 * 1024**2
CATALOG_KEYS = {
    "schema", "trained_for_organism", "qualification_passed", "release_id", "manifest_base64",
    "release_keys", "platforms", "files", "trained_provenance", "provenance_member",
    "required_device_profile", "launch_ready", "native_execution_verified",
    "notices_base64",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def profile():
    """Requirements to observe later, not claims about a currently running device."""
    return {
        "schema": "organism-native-trained-device-requirements-v1",
        "device_profile_memory_gib": 8,
        "resource_slice": device_resources.SLICE,
        "aggregate_memory_max_bytes": device_resources.MEMORY_MAX,
        "memory_swap_max_bytes": 0, "cpu_max_cores": 4,
        "aggregate_tasks_max": device_resources.TASKS_MAX,
        "storage_profile": device_storage.PROFILE,
        "total_organism_storage_max_bytes": device_storage.TOTAL_LIMIT,
        "network_budget": network_budget.PROFILE,
        "network_rate_bytes_per_second": network_budget.RATE,
        "network_burst_bytes": network_budget.BURST,
        "kernel_core_output_disabled": True,
        "actual_receiver_profile_observation_required": True,
        "complete_write_accounting_required": True,
        "offline_application_and_checker_required": True,
    }


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate candidate input field")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("Nonfinite candidate input value")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def check_identity(value):
    require(isinstance(value, dict) and set(value) == {"bytes", "sha256"}
            and type(value["bytes"]) is int and value["bytes"] > 0
            and isinstance(value["sha256"], str) and re.fullmatch("[0-9a-f]{64}", value["sha256"]),
            "Candidate evidence identity is invalid")


def validate_catalog(value):
    """Validate bindings and requirements without claiming the export audit was rerun."""
    require(isinstance(value, dict) and set(value) == CATALOG_KEYS,
            "Unexpected trained candidate fields")
    validate_payload_catalog(value, expected_schema=SCHEMA, expected_trained=True)
    require(value["launch_ready"] is False and value["native_execution_verified"] is False,
            "Candidate inputs cannot declare native execution or launch readiness")
    require(canonical(value["required_device_profile"]) == canonical(profile()),
            "Trained candidate device requirements differ")
    keys = {name: base64.b64decode(encoded, validate=True) for name, encoded in value["release_keys"].items()}
    manifest = verify_manifest(base64.b64decode(value["manifest_base64"], validate=True), keys,
                               require_trained=True, max_payload_bytes=3 * 1024**3)
    body = manifest["body"]
    require(isinstance(value["notices_base64"], str) and len(value["notices_base64"]) <= MAX_CATALOG,
            "Encoded trained notices exceed the catalog bound")
    actual_proof = provenance_bytes(base64.b64decode(value["notices_base64"], validate=True), body)
    proof = value["trained_provenance"]
    expected = {"schema", "training_admission", "export_audit", "export_workload", "model",
                "total_stored_parameters", "optimizer_steps_during_export", "numerical_gates_passed",
                "task_quality_verified", "native_execution_verified", "qualification_passed"}
    require(isinstance(proof, dict), "Trained provenance is not an object")
    precision_version = proof.get("schema") == "organism-trained-capsule-provenance-v2"
    if precision_version:
        expected |= {"quantization_precision", "response_audit", "export_job", "export_report"}
    require(isinstance(proof, dict) and set(proof) == expected
            and proof["schema"] in ("organism-trained-capsule-provenance-v1",
                                    "organism-trained-capsule-provenance-v2"),
            "Trained provenance schema differs")
    if precision_version:
        from ..quantization_precision import validate_specification
        validate_specification(proof["quantization_precision"])
        for name in ("response_audit", "export_job", "export_report"):
            check_identity(proof[name])
    require(proof["numerical_gates_passed"] is True
            and type(proof["optimizer_steps_during_export"]) is int
            and proof["optimizer_steps_during_export"] == 0
            and all(proof[key] is False for key in (
                "task_quality_verified", "native_execution_verified", "qualification_passed")),
            "Trained provenance claims differ")
    for name in ("export_audit", "export_workload", "model"):
        check_identity(proof[name])
    admission = proof["training_admission"]
    require(isinstance(admission, dict)
            and admission.get("schema") == "organism-trained-adapter-export-admission-v1"
            and admission.get("training_receipts_linked") is True
            and admission.get("required_native_precheck_linked") is True
            and admission.get("qualification_passed") is False
            and type(admission.get("optimizer_steps")) is int and admission["optimizer_steps"] > 0,
            "Completed training admission is required")
    for name in ("worker_input", "worker_completion", "training_completion", "recipe", "frozen_contract"):
        check_identity(admission[name])
    require(type(proof["total_stored_parameters"]) is int
            and body["model"]["parameters"] == proof["total_stored_parameters"] == admission["base_parameters"]
            and body["model"]["quantization"] == "Q4_K_M"
            and proof["model"] == body["files"][body["model"]["path"]],
            "Trained candidate model identity differs")
    member = value["provenance_member"]
    raw_proof = canonical(proof)
    require(member == {"path": PROVENANCE_MEMBER, "bytes": len(raw_proof),
                       "sha256": hashlib.sha256(raw_proof).hexdigest(),
                       "component": body["notices"], "component_identity": body["files"][body["notices"]]},
            "Signed provenance component binding differs")
    require(proof == actual_proof, "Catalog provenance differs from the signed notice bytes")
    require(len(canonical(value)) <= MAX_CATALOG, "Trained candidate catalog exceeds its bound")
    return value


def provenance_bytes(contents, body):
    """Verify the whole signed component before reading bounded, unique members."""
    expected = body["files"][body["notices"]]
    require(0 < len(contents) <= MAX_NOTICES
            and {"bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()} == expected,
            "Signed notices component changed or exceeds its bound")
    selected, seen, total = {}, set(), 0
    with tarfile.open(fileobj=io.BytesIO(contents), mode="r:gz") as archive:
        for index, entry in enumerate(archive):
            name = safe_relative(entry.name)
            require(index < 4096 and name not in seen and entry.isfile()
                    and 0 <= entry.size <= 1024**2, "Invalid signed notice member")
            seen.add(name)
            total += entry.size
            require(total <= MAX_NOTICES, "Expanded signed notices exceed their bound")
            if name in (PROVENANCE_MEMBER, "NOTICE-INDEX.json"):
                with archive.extractfile(entry) as stream:
                    raw = stream.read(entry.size + 1)
                require(len(raw) == entry.size, "Signed notice member length differs")
                selected[name] = raw
    require(set(selected) == {PROVENANCE_MEMBER, "NOTICE-INDEX.json"},
            "Signed trained provenance is missing")
    raw = selected[PROVENANCE_MEMBER]
    index = decode(selected["NOTICE-INDEX.json"])
    require(index.get("trained_model_provenance") == {
        "member": PROVENANCE_MEMBER, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
        "Signed notice index and training provenance differ")
    require(index["model"]["trained_for_organism"] is True
            and all(index["model"][key] == item for key, item in body["model"].items()),
            "Signed notice model differs from the capsule")
    proof = decode(raw)
    require(canonical(proof) == raw, "Signed provenance is noncanonical")
    return proof


def signed_provenance(capsule, body):
    from ..export_task_inputs import identity
    """Read the actual signed notices archive without trusting catalog assertions."""
    path = Path(capsule) / body["notices"]
    expected = body["files"][body["notices"]]
    require(expected["bytes"] <= MAX_NOTICES and identity(path) == expected,
            "Signed notices component changed or exceeds its bound")
    contents = path.read_bytes()
    proof = provenance_bytes(contents, body)
    require(identity(path) == expected, "Signed notices changed during inspection")
    return proof, contents


def bind_catalog(artifact, *, admission, pair, expected_provenance):
    """Pure local binding after the production caller independently checked the export."""
    from ..export_task_inputs import ARMS
    artifact = Path(artifact).resolve()
    trust = decode((artifact / "release-trust.json").read_bytes())
    require(trust["require_trained"] is True and trust["private_key_retained"] is False,
            "A separately prepared trained release is required")
    keys = {name: base64.b64decode(value, validate=True) for name, value in trust["public_keys"].items()}
    raw = (artifact / "capsule/.manifest.json").read_bytes()
    envelope = verify_capsule_directory(artifact / "capsule", raw, keys, require_trained=True,
                                        max_payload_bytes=3 * 1024**3)
    body = envelope["body"]
    require(trust["release_id"] == envelope["release_id"], "Trained release trust identity differs")
    require(pair["numerical_gates_passed"] is True and pair["qualification_passed"] is False
            and expected_provenance["training_admission"] == admission
            and expected_provenance["export_audit"] == pair["audit"]
            and expected_provenance["export_workload"] == pair["workload"],
            "Trained candidate export binding differs")
    model = pair["models"][ARMS[1]]
    require(body["files"][body["model"]["path"]] == {key: model[key] for key in ("bytes", "sha256")},
            "Capsule weights differ from the admitted trained export")
    proof, notices = signed_provenance(artifact / "capsule", body)
    require(proof == expected_provenance, "Signed provenance differs from actual admitted export")
    proof_raw = canonical(proof)
    value = {
        "schema": SCHEMA, "trained_for_organism": True, "qualification_passed": False,
        "launch_ready": False, "native_execution_verified": False,
        "release_id": envelope["release_id"], "manifest_base64": base64.b64encode(raw).decode(),
        "release_keys": deepcopy(trust["public_keys"]), "platforms": deepcopy(body["platforms"]),
        "files": deepcopy(body["files"]), "trained_provenance": proof,
        "provenance_member": {"path": PROVENANCE_MEMBER, "bytes": len(proof_raw),
                              "sha256": hashlib.sha256(proof_raw).hexdigest(),
                              "component": body["notices"], "component_identity": body["files"][body["notices"]]},
        "required_device_profile": profile(),
        "notices_base64": base64.b64encode(notices).decode(),
    }
    return validate_catalog(value)


def prepare(workspace, output, *, artifact, export_directory, admission_path, training_recipe):
    from ..build_development_capsule import PIN_FILE
    from ..build_trained_capsule import bind_inputs
    from ..export_task_inputs import checked_pair, identity
    workspace, output = Path(workspace).resolve(), Path(output).absolute()
    admission = decode(Path(admission_path).read_bytes())
    pair = checked_pair(export_directory, admission)
    pins = decode((workspace / "organism" / PIN_FILE).read_bytes())
    _, _, provenance = bind_inputs(workspace, pins, admission, pair, training_recipe)
    value = bind_catalog(artifact, admission=admission, pair=pair, expected_provenance=provenance)
    output.mkdir(parents=False, exist_ok=False)
    target = output / "trained-catalog.json"
    target.write_bytes(canonical(value))
    receipt = {
        "schema": SCHEMA, "catalog": identity(target), "release_id": value["release_id"],
        "training_admission": identity(admission_path), "training_recipe": identity(training_recipe),
        "export_pair": pair, "publisher": identity(Path(__file__)),
        "source_capsule": str(Path(artifact).resolve()), "signed_provenance_verified": True,
        "launch_kit_created": False, "resources_created": False, "native_execution_verified": False,
        "qualification_passed": False,
    }
    (output / "receipt.json").write_bytes(canonical(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--export-directory", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--training-recipe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[2], args.output, artifact=args.artifact,
                             export_directory=args.export_directory, admission_path=args.admission,
                             training_recipe=args.training_recipe)))
