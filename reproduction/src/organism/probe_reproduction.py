"""Live Mac stock-model choices against read-only TLS fixture receivers."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import tempfile
import time
import uuid

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

from .application_store import ApplicationStore
from .async_peer import AsyncPeerClient
from .discovery_book import DiscoveryBook, SCHEMA, sign_discovery_book
from .inference_server import InferenceServer, RuntimeConfig, digest
from .peer_identity import device_certificate, owner_certificate, write_identity
from .peer_transport import PeerServer
from .receiver import LocalReceiver
from .reproduction_agent import decide
from .reproduction_evidence import verify_choice
from .tests import test_receiver_state as fixtures


async def run(root, output):
    if platform.system() != "Darwin":
        raise ValueError("This public development probe requires the Mac")
    output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((root / "organism/experiment-contract.draft.json").read_bytes())
    f = fixtures.ReceiverStateTests(methodName="runTest")
    f.setUp()
    report = {"scope": "Actual stock 3B model chooses procedures from real TLS observations; receiving state and release metadata are explicit fixtures",
              "started_at": datetime.now(timezone.utc).isoformat(), "cases": [], "failure": None,
              "trained_for_organism": False, "native_execution_verified": False, "complete_capsule_verified": False,
              "child_established": False, "autonomous_reproduction_verified": False, "qualification_passed": False,
              "training_example_admitted": False}
    snapshots = output / "source-snapshot"
    snapshots.mkdir()
    files = {Path(__file__)}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).parent == root / "organism" and Path(path).suffix == ".py":
            files.add(Path(path))
    report["source_hashes"] = {}
    for source in sorted(files):
        raw = source.read_bytes()
        (snapshots / source.name).write_bytes(raw)
        report["source_hashes"][source.name] = hashlib.sha256(raw).hexdigest()
    fixture_raw = Path(fixtures.__file__).read_bytes()
    (snapshots / "fixture_receiver_state.py").write_bytes(fixture_raw)
    report["fixture_source_sha256"] = hashlib.sha256(fixture_raw).hexdigest()
    receiver = peer = inference = None
    started = time.monotonic()
    try:
        owner = owner_certificate(f.owner_key)
        identities = {}
        for identity, key in zip(f.identities, f.device_keys):
            certificate = device_certificate(f.owner_key, owner, key.public_key())
            identities[identity] = write_identity(f.root / ("identity-" + identity[:12]), key, certificate, owner)
        receiver = LocalReceiver(f.state, f.root / "fixture-capsules")
        peer = PeerServer(receiver, identities[f.device]).start()
        endpoints = {identity: {"address": "127.0.0.1", "port": 7400 + index}
                     for index, identity in enumerate(f.identities)}
        endpoints[f.device] = {"address": peer.server_address[0], "port": peer.server_address[1]}
        body = {"schema": SCHEMA, "policy_id": f.policy.policy_id, "epoch": 1,
                "lab_cidrs": ["127.0.0.0/24"], "endpoints": endpoints}
        book = DiscoveryBook(sign_discovery_book(body, f.owner_key, f.policy), f.policy, f.owner_public)
        client = AsyncPeerClient(f.policy, f.parent, identities[f.parent])
        envelope = json.loads(f.manifest)
        (output / "fixture-enrollment.json").write_bytes(f.policy.signed)
        (output / "fixture-book.json").write_bytes(book.signed)
        (output / "fixture-release.json").write_bytes(f.manifest)
        config = RuntimeConfig(root / "models/granite-3b/granite-base-Q4_K_M.gguf",
                               contract["model"]["local_export_sha256"], contract["model"]["local_export_bytes"],
                               root / "tools/llama.cpp/b9850/llama-b9850/llama-server",
                               contract["runtime"]["development_binary_sha256"],
                               uuid.uuid4().hex, f.parent, uuid.uuid4().hex, envelope["release_id"], False, False)
        report.update(model_sha256=config.model_sha256, runtime_sha256=config.executable_sha256,
                      process_token=config.start_token, local_device_id=f.parent, target_device_id=f.device,
                      release_id=envelope["release_id"], model_does_not_match_fixture_payload=True)
        (output / "request.json").write_text(json.dumps(report, indent=2) + "\n")
        with tempfile.TemporaryDirectory(prefix="or-choice-", dir="/private/tmp") as temporary:
            work = Path(temporary)
            store = ApplicationStore(work, config.start_token)
            inference = InferenceServer(config, work)
            try:
                report["inference"] = await inference.start()
                print(json.dumps({"model": "serving", "pid": inference.process.pid}), flush=True)
                # Predetermined public cases. There is no selection after seeing
                # a result, no transfer, and no child activation by this probe.
                for case_id, expected in (("available", "establish"), ("retired", "defer")):
                    if case_id == "retired":
                        f.state.owner_control("retire", "Predetermined public reproduction-decision fixture")
                    request_id = uuid.uuid4().hex
                    store.reserve(request_id, f.device)
                    case_started = time.monotonic()
                    await decide(inference, client, book, envelope, store, request_id, f.device)
                    job = store.get(request_id)
                    failure, choice = None, None
                    try:
                        choice = verify_choice(job, peer_id=f.device, start_token=config.start_token, envelope=envelope)
                        if choice["action"] != expected:
                            raise ValueError("Choice differs from the predetermined public fixture outcome")
                    except (ValueError, KeyError, TypeError) as exc:
                        failure = type(exc).__name__ + ": " + str(exc)
                    case = {"case_id": case_id, "expected_action": expected, "request_id": request_id,
                            "passed": failure is None, "failure": failure, "choice": choice,
                            "seconds": round(time.monotonic() - case_started, 3), "job": job}
                    (output / (case_id + ".json")).write_text(json.dumps(case, indent=2) + "\n")
                    report["cases"].append({key: case[key] for key in ("case_id", "passed", "seconds", "failure")})
                    print(json.dumps({"case": case_id, "passed": case["passed"], "failure": failure}), flush=True)
            finally:
                if inference is not None:
                    report["shutdown"] = await inference.stop()
                shutil.copytree(work, output / "work-evidence")
        report["receiver_birth_count"] = f.state.snapshot()["birth_count"]
        if report["receiver_birth_count"] != 0:
            raise ValueError("Read-only decision probe unexpectedly created a birth")
        (output / "receiver-transport-audit.json").write_text(json.dumps(peer.audit.records(), indent=2) + "\n")
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        if inference is not None and inference.process is not None and inference.process.returncode is None:
            report["shutdown"] = await inference.stop()
        if peer is not None:
            peer.close()
        f.doCleanups()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["maximum_child_peak_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["sources_unchanged_during_run"] = all(digest(root / "organism" / name) == expected
                                                for name, expected in report["source_hashes"].items())
    if not report["sources_unchanged_during_run"]:
        report["failure"] = report["failure"] or "Probe sources changed during the run"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(output / "report.json"), "failure": report["failure"],
                      "passes": sum(case["passed"] for case in report["cases"])}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = asyncio.run(run(Path(__file__).resolve().parents[1], args.output.resolve()))
    return 1 if report["failure"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
