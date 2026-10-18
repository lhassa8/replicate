"""Read back and prepare both real platforms without executing foreign code."""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .audit_preparation_archives import compare_archive
from .build_development_capsule import KNOWLEDGE_FILES, identity, write_archive
from .capsule import PLATFORMS, CapsuleError, canonical, verify_capsule_directory, verify_manifest
from .enrollment import Enrollment, key_id, public_bytes, sign_enrollment
from .entrypoint import validate_installation
from .package_application import package as package_application
from .package_self_test import package as package_self_test
from .preparation import PackagePreparer, ROLE_LIMITS
from .safe_archive import remove_new_tree


def archive_contents(path):
    contents = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if (not member.isfile() or member.name in contents or member.size > 1024**2
                    or len(contents) >= 1000 or member.uid or member.gid or member.mtime
                    or member.mode != 0o644):
                raise ValueError("Generated component contains unexpected entries")
            contents[member.name] = archive.extractfile(member).read()
            if sum(map(len, contents.values())) > 16 * 1024**2:
                raise ValueError("Generated component exceeds audit bound")
    return contents


def verify_notices(capsule, body):
    contents = archive_contents(capsule / body["notices"])
    index = json.loads(contents["NOTICE-INDEX.json"])
    expected = {}
    for item in index["runtime_notices"]:
        source = (item["component"], item["member"])
        if source in expected:
            raise ValueError("Duplicate original notice path")
        expected[source] = item
    observed = set()
    for platform in sorted(body["platforms"]):
        for name in body["platforms"][platform].values():
            with tarfile.open(capsule / name, "r:gz") as archive:
                for member in archive:
                    parts = member.name.casefold().split("/")
                    selected = (any(part.startswith(("license", "licence", "copyright", "notice", "copying"))
                                    for part in parts)
                                or (len(parts) > 1 and parts[-2].endswith(".dist-info") and parts[-1] == "metadata"))
                    if not member.isfile() or not selected:
                        continue
                    source = (name, member.name)
                    if source not in expected:
                        raise ValueError("Original runtime notice is missing from the retained index")
                    item = expected[source]
                    original = archive.extractfile(member).read()
                    if (contents[item["notice_file"]] != original or len(original) != item["bytes"]
                            or hashlib.sha256(original).hexdigest() != item["sha256"]):
                        raise ValueError("Retained notice differs from its original runtime archive")
                    observed.add(source)
    if observed != set(expected):
        raise ValueError("Retained notice inventory differs")
    if contents["model/LICENSE-Apache-2.0.txt"] != contents[index["model"]["standard_license_text_source"]["notice_file"]]:
        raise ValueError("Model's retained standard license text differs")
    card = index["model"]["card"]
    if (len(contents["model/README.upstream.md"]) != card["bytes"]
            or hashlib.sha256(contents["model/README.upstream.md"]).hexdigest() != card["sha256"]):
        raise ValueError("Retained model card differs")
    if set(contents) != ({"NOTICE-INDEX.json", "model/README.upstream.md", "model/LICENSE-Apache-2.0.txt"}
                         | {item["notice_file"] for item in expected.values()}):
        raise ValueError("Notice component has extra or missing content")
    return contents, {"original_notice_entries_compared": len(observed),
                      "unique_notice_files": len(contents), "redistribution_review_complete": False}


def make_policy(envelope, public_keys, platform):
    # Temporary local packaging identity only. No receiving listener, process
    # activation or network operation is created by this audit.
    owner = Ed25519PrivateKey.generate()
    device = public_bytes(Ed25519PrivateKey.generate().public_key())
    body = {
        "schema": "organism-enrollment-v1", "experiment_id": envelope["body"]["experiment_id"], "epoch": 1,
        "devices": {key_id(device): {"public_key": base64.b64encode(device).decode(), "platform": platform}},
        "release_keys": {name: base64.b64encode(raw).decode() for name, raw in public_keys.items()},
        "require_trained": False, "max_payload_bytes": 3 * 1024**3, "birth_seconds": 1200,
        "max_births_per_device": 1, "max_process_starts": 1,
    }
    return Enrollment(sign_enrollment(body, owner), public_bytes(owner.public_key()))


def run(root, artifact, output, *, temporary_root, selected_platform=None):
    if selected_platform is not None and selected_platform not in PLATFORMS:
        raise ValueError("Unknown native preparation platform")
    root, artifact, output = Path(root), Path(artifact), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    capsule = artifact / "capsule"
    manifest = (capsule / ".manifest.json").read_bytes()
    trust = json.loads((artifact / "release-trust.json").read_bytes())
    keys = {name: base64.b64decode(value, validate=True) for name, value in trust["public_keys"].items()}
    report = {"schema": "organism-complete-capsule-audit-v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Actual complete stock capsule; selected native installation trees prepared on Mac without execution",
              "selected_platform": selected_platform, "both_platforms_prepared": False,
              "passed": False, "native_execution_verified": False, "readiness_executed": False,
              "autonomous_reproduction_verified": False, "qualification_passed": False, "platforms": []}
    started = time.monotonic()
    try:
        envelope = verify_capsule_directory(capsule, manifest, keys, require_trained=False)
        body = envelope["body"]
        targets = sorted(body["platforms"]) if selected_platform is None else [selected_platform]
        if any(target not in body["platforms"] for target in targets):
            raise ValueError("Requested platform is absent from the complete capsule")
        report["requested_platforms"] = targets
        if trust["release_id"] != envelope["release_id"] or body["model"]["trained_for_organism"]:
            raise ValueError("Expected the explicit stock development release")
        try:
            verify_manifest(manifest, keys)
        except CapsuleError:
            report["trained_only_policy_rejected_stock"] = True
        else:
            raise ValueError("Stock release unexpectedly passed a trained-only policy")
        report["release_id"] = envelope["release_id"]
        report["manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
        report["real_model_identity"] = identity(capsule / body["model"]["path"])
        with (capsule / body["model"]["path"]).open("rb") as source:
            if source.read(4) != b"GGUF":
                raise ValueError("Real stock model is missing its GGUF header")
        with tempfile.TemporaryDirectory(prefix="organism-component-rebuild-", dir=temporary_root) as scratch:
            scratch = Path(scratch)
            package_application(root, scratch / "agent.tar.gz")
            package_self_test(root / "self_test/readiness.py", scratch / "self_test.tar.gz")
            knowledge = {name: (root / "capsule_content" / name).read_bytes() for name in KNOWLEDGE_FILES}
            write_archive(knowledge, scratch / "knowledge.tar.gz")
            notice_contents, report["notices"] = verify_notices(capsule, body)
            write_archive(notice_contents, scratch / "notices.tar.gz")
            for role in ("agent", "self_test", "knowledge", "notices"):
                if identity(scratch / (role + ".tar.gz")) != body["files"][body[role]]:
                    raise ValueError("Generated component rebuild differs: " + role)
            report["generated_components_reproduced_byte_for_byte"] = True
        for platform in targets:
            temporary = Path(tempfile.mkdtemp(prefix="organism-installation-audit-", dir=temporary_root))
            record = {"platform": platform, "passed": False, "native_execution_verified": False}
            report["platforms"].append(record)
            try:
                policy = make_policy(envelope, keys, platform)
                preparer = PackagePreparer(temporary / "prepared", policy)
                prepared = preparer.prepare(capsule, manifest, platform)
                index, loaded = validate_installation(prepared.path, owner_uid=os.geteuid())
                if loaded != envelope:
                    raise ValueError("Installed bootstrap read a different release")
                work = temporary / "work"
                work.mkdir(mode=0o700)
                spec = prepared.process_spec(work)
                spec.validate()
                if spec.readiness is None:
                    raise ValueError("Complete package failed to declare its carried checker")
                record["readiness_declared"] = True
                record["node_context_provisioned"] = False
                record["prepared_inventory_sha256"] = hashlib.sha256(canonical(index["inventory"])).hexdigest()
                record["prepared_file_bytes"] = sum(item.get("bytes", 0) for item in index["inventory"].values())
                record["prepared_entries"] = len(index["inventory"])
                record["native"] = index["native"]
                record["archive_comparisons"] = {}
                for role in ROLE_LIMITS:
                    name = body["platforms"][platform].get(role, body.get(role))
                    record["archive_comparisons"][role] = compare_archive(
                        prepared.capsule / name, prepared.path / "runtime" / role,
                        {"python": "python", "inference": "llama-b9850"}.get(role), platform)
                # Both opposite-platform archives and the actual model remain
                # in the prepared capsule, checked by the production verifier.
                verify_capsule_directory(prepared.capsule, manifest, keys, require_trained=False)
                record["complete_reproductive_payload_verified"] = True
                record["prepared_index"] = "installation-index-" + platform + ".json"
                (output / record["prepared_index"]).write_bytes((prepared.path / ".installation.json").read_bytes())
                record["passed"] = True
            finally:
                remove_new_tree(temporary)
                record["temporary_installation_removed"] = True
            print(json.dumps({"platform": platform, "complete_capsule_prepared": True,
                              "native_execution_verified": False}), flush=True)
        report["passed"] = (len(report["platforms"]) == len(targets)
                            and all(item["passed"] for item in report["platforms"]))
        report["both_platforms_prepared"] = report["passed"] and set(targets) == set(PLATFORMS)
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["audit_sha256"] = identity(Path(__file__))["sha256"]
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporary-root", type=Path, required=True)
    parser.add_argument("--platform", choices=sorted(PLATFORMS),
                        help="Prepare only this platform; still verify the complete two-platform capsule")
    args = parser.parse_args()
    report = run(Path(__file__).resolve().parent, args.artifact.resolve(), args.output.resolve(),
                 temporary_root=args.temporary_root.resolve(), selected_platform=args.platform)
    print(json.dumps({"passed": report["passed"], "failure": report.get("failure")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
