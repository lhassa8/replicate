"""Assemble a complete stock development capsule from pinned local inputs.

No network, device enrollment, execution of payloads, training or deployment.
The temporary signing key is never saved or placed in the capsule.
"""

import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from .capsule import canonical, sign_manifest, validate_body, verify_capsule_directory
from .package_application import package as package_application
from .package_self_test import package as package_self_test
from .safe_archive import safe_relative
from .review_reference import FILENAME as REVIEW_REFERENCES, ReviewReferences


PIN_FILE = "development-capsule-inputs-20261014.json"
MAX_NOTICE_BYTES = 16 * 1024**2
MAX_NOTICE_FILE_BYTES = 1024**2
KNOWLEDGE_FILES = ("OPERATING-REFERENCE.md", REVIEW_REFERENCES)


def identity(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024**2):
            value.update(block)
    return {"bytes": Path(path).stat().st_size, "sha256": value.hexdigest()}


def verified_copy(source, target, expected):
    """Hash exactly the opened regular input while making a private copy."""
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as original:
        info = os.fstat(original.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected["bytes"]:
            raise ValueError("Pinned input type or length differs")
        target.parent.mkdir(parents=True, exist_ok=True)
        value, count = hashlib.sha256(), 0
        with target.open("xb") as output:
            while block := original.read(1024**2):
                count += len(block)
                if count > expected["bytes"]:
                    raise ValueError("Pinned input grew during copy")
                value.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if count != expected["bytes"] or value.hexdigest() != expected["sha256"]:
            raise ValueError("Pinned input hash or length differs")


def write_archive(files, target):
    if not files or sum(map(len, files.values())) > MAX_NOTICE_BYTES:
        raise ValueError("Reference or notice component exceeds its bound")
    for name in files:
        safe_relative(name)
    # PAX preserves long original paths without platform-dependent timestamps.
    with target.open("xb") as output, gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name, raw in sorted(files.items()):
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode = len(raw), 0o644
                entry.uid = entry.gid = entry.mtime = 0
                entry.uname = entry.gname = ""
                archive.addfile(entry, io.BytesIO(raw))
    return identity(target)


def notice_member(name):
    parts = name.lower().split("/")
    return (any(part.startswith(("license", "licence", "copyright", "notice", "copying"))
                for part in parts)
            or (len(parts) >= 2 and parts[-2].endswith(".dist-info") and parts[-1] == "metadata"))


def collect_notices(source, payload, contents, index):
    """Retain notice bytes, not executable archive entries; deduplicate by hash."""
    total, count, matches = 0, 0, 0
    seen = set()
    with tarfile.open(source, "r:gz") as archive:
        for member in archive:
            count += 1
            total += max(0, member.size)
            if count > 30000 or total > 512 * 1024**2:
                raise ValueError("Pinned archive exceeds notice-scan bound")
            name = safe_relative(member.name.rstrip("/"))
            if name in seen:
                raise ValueError("Duplicate archive path in notice scan")
            seen.add(name)
            if not member.isfile() or not notice_member(name):
                continue
            if not 0 <= member.size <= MAX_NOTICE_FILE_BYTES:
                raise ValueError("Notice file exceeds its bound")
            with archive.extractfile(member) as handle:
                raw = handle.read(MAX_NOTICE_FILE_BYTES + 1)
            if len(raw) != member.size:
                raise ValueError("Notice file length differs")
            sha = hashlib.sha256(raw).hexdigest()
            content_path = "retained/" + sha + ".txt"
            contents.setdefault(content_path, raw)
            if sum(map(len, contents.values())) > MAX_NOTICE_BYTES:
                raise ValueError("Retained notice contents exceed their bound")
            index.append({"component": payload, "member": name, "notice_file": content_path,
                          "bytes": len(raw), "sha256": sha})
            matches += 1
    if not matches:
        raise ValueError("Runtime archive has no retained license or notice")


def validate_inputs(value):
    if value.get("schema") != "organism-stock-capsule-inputs-v1":
        raise ValueError("Unexpected development input schema")
    if value["model"]["trained_for_organism"] is not False:
        raise ValueError("This builder cannot declare a trained model")
    body = {"schema": "organism-capsule-v1", "experiment_id": value["experiment_id"],
            "model": value["model"], "platforms": value["platforms"],
            **{role: "common/" + role + ".tar.gz" for role in ("agent", "knowledge", "self_test", "notices")},
            "files": {name: {key: entry[key] for key in ("bytes", "sha256")}
                      for name, entry in value["files"].items()}}
    for role in ("agent", "knowledge", "self_test", "notices"):
        body["files"][body[role]] = {"bytes": 1, "sha256": "0" * 64}
    validate_body(body, require_trained=False)
    for entry in (*value["files"].values(), value["model_card"]):
        safe_relative(entry["source"])
    return body


def collect_knowledge(source):
    files = {name: (Path(source) / "capsule_content" / name).read_bytes() for name in KNOWLEDGE_FILES}
    return files, len(ReviewReferences(files[REVIEW_REFERENCES]))


def build(workspace, output, *, copy_mode="bytes"):
    """Keep the stock entry point separate from trained-export admission."""
    workspace, output = Path(workspace).resolve(), Path(output).absolute()
    source = workspace / "organism"
    pins_raw = (source / PIN_FILE).read_bytes()
    pins = json.loads(pins_raw)
    body = validate_inputs(pins)
    return _assemble(workspace, output, pins, pins_raw, body, copy_mode=copy_mode)


def _assemble(workspace, output, pins, pins_raw, body, *, copy_mode="bytes", trained_provenance=None):
    """Common packaging after the caller has admitted its exact model input."""
    workspace, output = Path(workspace).resolve(), Path(output).absolute()
    source = workspace / "organism"
    trained = trained_provenance is not None
    if body["model"]["trained_for_organism"] is not trained:
        raise ValueError("Capsule model and admitted provenance differ")
    validate_body(body, require_trained=trained)
    # Validate references before copying the multi-gigabyte runtime payloads.
    knowledge, review_procedures = collect_knowledge(source)
    if copy_mode not in ("bytes", "mac_clone"):
        raise ValueError("Unsupported development payload copy mode")
    copy_payload = verified_copy
    if copy_mode == "mac_clone":
        import platform
        from .capsule_clone import verified_clone
        if platform.system() != "Darwin":
            raise ValueError("Mac cloning is an authoring option only")
        copy_payload = verified_clone
    required_free = 256 * 1024**2
    if copy_mode == "bytes":
        required_free += sum(v["bytes"] for v in pins["files"].values())
    if shutil.disk_usage(output.parent).free < required_free:
        raise ValueError("Insufficient free space for the complete development capsule")
    output.mkdir(parents=False, exist_ok=False)
    pending = output / ".pending-capsule"
    pending.mkdir(mode=0o700)
    receipt = {"schema": ("organism-complete-trained-capsule-v1" if trained
                          else "organism-complete-stock-capsule-v1"),
               "started_at": datetime.now(timezone.utc).isoformat(), "complete_capsule_assembled": False,
               "trained_for_organism": trained, "native_execution_verified": False,
               "autonomous_reproduction_verified": False, "qualification_passed": False,
               "upstream_tensor_conversion_rederived": False, "redistribution_review_complete": False,
               "payload_copy_mode": copy_mode, "payload_copy_receipts": {},
               "payload_copy_helper_sha256": (identity(Path(__file__).with_name("capsule_clone.py"))["sha256"]
                                              if copy_mode == "mac_clone" else None),
               "input_pins_sha256": hashlib.sha256(pins_raw).hexdigest()}
    try:
        for name, expected in sorted(pins["files"].items()):
            path = workspace / expected["source"]
            if not path.resolve().is_relative_to(workspace):
                raise ValueError("Pinned source escapes the workspace")
            copied = copy_payload(path, pending / name, {key: expected[key] for key in ("bytes", "sha256")})
            if copied is not None:
                receipt["payload_copy_receipts"][name] = copied
        (pending / "common").mkdir()
        receipt["application"] = package_application(source, pending / body["agent"])
        receipt["self_test"] = package_self_test(source / "self_test/readiness.py", pending / body["self_test"])
        receipt["source_review_procedures"] = review_procedures
        receipt["knowledge_sources"] = {
            name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in knowledge.items()}
        write_archive(knowledge, pending / body["knowledge"])
        contents, index = {}, []
        for platform in sorted(pins["platforms"]):
            for name in sorted(pins["platforms"][platform].values()):
                collect_notices(pending / name, name, contents, index)
        card = pins["model_card"]
        card_source = workspace / card["source"]
        if identity(card_source) != {key: card[key] for key in ("bytes", "sha256")}:
            raise ValueError("Pinned model card changed")
        contents["model/README.upstream.md"] = card_source.read_bytes()
        # The pinned model card declares Apache-2.0. Retain its standard text
        # from the independently pinned Ubuntu common-license payload.
        apache = next(item for item in index
                      if item["component"] == pins["platforms"]["linux-x86_64"]["dependencies"]
                      and item["member"] == "supplemental/notices/common-licenses/Apache-2.0")
        contents["model/LICENSE-Apache-2.0.txt"] = contents[apache["notice_file"]]
        notice_index = {
            "schema": "organism-retained-notices-v1", "runtime_notices": index,
            "model": {**pins["model"], "card": card, "standard_license_text_source": apache,
                      "upstream_tensor_conversion_rederived": False},
            "scope": "Retained upstream license, copyright, notice and distribution metadata; review remains open",
            "project_authored_code": "Owner-controlled research code; no new public redistribution license granted",
            "redistribution_review_complete": False,
        }
        if trained:
            provenance_name = "model/TRAINING-AND-EXPORT.json"
            contents[provenance_name] = canonical(trained_provenance)
            notice_index["trained_model_provenance"] = {
                "member": provenance_name, "bytes": len(contents[provenance_name]),
                "sha256": hashlib.sha256(contents[provenance_name]).hexdigest(),
            }
            receipt["trained_model_provenance"] = notice_index["trained_model_provenance"]
        contents["NOTICE-INDEX.json"] = canonical(notice_index)
        write_archive(contents, pending / body["notices"])
        receipt["notice_entries"] = len(index)
        receipt["unique_notice_files"] = len(contents)
        for name in body["files"]:
            body["files"][name] = identity(pending / name)
        key = Ed25519PrivateKey.generate()
        manifest = sign_manifest(body, key, require_trained=trained)
        public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        trust = {hashlib.sha256(public).hexdigest(): public}
        del key
        (pending / ".manifest.json").write_bytes(manifest)
        envelope = verify_capsule_directory(pending, manifest, trust, require_trained=trained)
        (output / "release-trust.json").write_bytes(canonical({
            "schema": "organism-development-release-trust-v1",
            "scope": "Experimental release verification only; does not enroll or authorize any device",
            "release_id": envelope["release_id"], "require_trained": trained,
            "public_keys": {identity: base64.b64encode(raw).decode() for identity, raw in trust.items()},
            "private_key_retained": False,
        }))
        (output / "input-pins.json").write_bytes(pins_raw)
        (output / "notice-index.json").write_bytes(canonical(notice_index))
        os.rename(pending, output / "capsule")
        receipt.update(complete_capsule_assembled=True, release_id=envelope["release_id"],
                       signer_id=envelope["signer_id"], manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                       payload_bytes=sum(v["bytes"] for v in body["files"].values()),
                       payload_components=len(body["files"]), files=body["files"])
        receipt["application"]["archive"] = str(output / "capsule" / body["agent"])
    except BaseException as exc:
        receipt["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        receipt["builder_sha256"] = identity(Path(__file__))["sha256"]
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--copy-mode", choices=("bytes", "mac_clone"), default="bytes",
                        help="Mac clone mode requires cloning support; it never falls back to a full byte copy")
    args = parser.parse_args()
    receipt = build(Path(__file__).resolve().parents[1], args.output, copy_mode=args.copy_mode)
    print(json.dumps({key: receipt[key] for key in
                      ("release_id", "payload_components", "payload_bytes", "qualification_passed")}))


if __name__ == "__main__":
    main()
