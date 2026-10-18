"""Build the application and generic reviewer without evaluation cases or answers."""

import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tarfile


APPLICATION_MODULES = (
    "__init__", "application", "application_store", "capsule", "diagnostic_agent",
    "entrypoint", "equipment", "inference_server", "runtime_environment",
    "discovery", "discovery_book", "node_context", "enrollment", "peer_audit",
    "peer_identity", "peer_transport", "receiver", "receiver_state",
    "async_peer", "reproduction", "reproduction_agent", "reproduction_journal", "reproduction_evidence",
    "reproduction_protocol",
    "equipment_interfaces", "equipment_diagnostic_agent", "temperature_measurement",
    "experience", "experience_store", "experience_exchange",
    "equipment_experience",
    "source_journal",
    "experience_proposal", "review_inbox", "source_review_queue", "source_inventory",
    "native_source_reviewer", "review_reference", "experience_endorsement",
    "diagnostic_oracle", "audit_equipment_diagnostics", "audit_inherited_diagnostics", "audit_application",
)
EXTERNAL_IMPORTS = frozenset(("cryptography", "httpx", "jsonschema", "strands"))
MAX_SOURCE_BYTES = 1024 * 1024


def collect(source):
    files, external = {}, set()
    for module in APPLICATION_MODULES:
        path = source / (module + ".py")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_SOURCE_BYTES:
            raise ValueError("Application input must be a bounded regular source file")
        raw = path.read_bytes()
        if len(raw) > MAX_SOURCE_BYTES:
            raise ValueError("Application source exceeds its bound")
        tree = ast.parse(raw, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level != 1 or not node.module or node.module.split(".")[0] not in APPLICATION_MODULES:
                        raise ValueError("Application archive is missing an internal import")
                    continue
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [entry.name for entry in node.names]
            else:
                continue
            for name in names:
                if name == "organism" or name.startswith("organism."):
                    if name not in {"organism." + item for item in APPLICATION_MODULES}:
                        raise ValueError("Application archive is missing an internal import")
                else:
                    top = name.split(".")[0]
                    if top not in sys.stdlib_module_names:
                        if top not in EXTERNAL_IMPORTS:
                            raise ValueError("Application has an undeclared external import")
                        external.add(top)
        files["organism/" + module + ".py"] = raw
    return files, sorted(external)


def package(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Do not overwrite an existing application archive")
    files, external = collect(source)
    # Read and validate every source before creating the output. Add exactly
    # those bytes, so later source edits cannot silently change this archive.
    with output.open("xb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for name, raw in sorted(files.items()):
                    info = tarfile.TarInfo(name)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.size, info.mode = len(raw), 0o644
                    archive.addfile(info, io.BytesIO(raw))
    packed = output.read_bytes()
    return {
        "schema": "organism-development-application-v1",
        "scope": "Diagnostic and reproduction application component; incomplete organism release",
        "archive": str(output), "archive_bytes": len(packed),
        "archive_sha256": hashlib.sha256(packed).hexdigest(),
        "source_files": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                         for name, raw in sorted(files.items())},
        "expanded_bytes": sum(map(len, files.values())), "external_imports": external,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "entrypoint": "organism/entrypoint.py",
        "runtime_dependencies_carried_in_separate_component": True,
        "model_included": False, "development_evaluation_assets_included": False,
        "full_capsule": False, "native_execution_verified": False,
        "reproduction_controller_included": True,
        "experience_exchange_included": True,
        "live_experience_issuance_included": True,
        "local_source_reviewer_included": True,
        "review_requires_release_bound_references_and_owner_trust": True,
        "evaluation_cases_or_expected_answers_included": False,
        "autonomous_reproduction_verified": False, "qualification_passed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = package(args.source, output / "agent.tar.gz")
    (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"archive": result["archive"], "bytes": result["archive_bytes"],
                      "source_files": len(result["source_files"]), "qualification_passed": False}))


if __name__ == "__main__":
    main()
