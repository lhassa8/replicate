"""Owner seed provisioning through the real receiver; never an offspring claim."""

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import threading
import time
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ..capsule import canonical, verify_capsule_directory
from ..discovery_book import sign_discovery_book
from ..enrollment import Enrollment, key_id, public_bytes, sign_enrollment
from ..network_namespace import inspect_namespace, verify_process_namespace
from ..peer_identity import LabNetwork, device_certificate, owner_certificate, write_identity
from ..peer_transport import PeerClient, PeerServer
from ..receiver_service import (
    ReceiverService, apply_owner_control, configuration, native_backend, native_components, native_state,
)
from ..receiver_state import ReceiverState
from ..supervision_records import KINDS
from .common import write_json
from .startup_common import BASE, NS_PATH, OUTPUT, ROOT, validate_catalog


def private_json(path, value):
    if os.path.lexists(path):
        raise FileExistsError("Do not overwrite owner provisioning")
    pending = path.with_name("." + path.name + ".pending")
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(canonical(value))
            output.flush()
            os.fsync(output.fileno())
        if os.path.lexists(path):
            raise FileExistsError("Owner provisioning appeared during publication")
        os.rename(pending, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        pending.unlink(missing_ok=True)


def provision(catalog, architecture):
    """One ephemeral owner, seed identity and non-serving bootstrap identity."""
    BASE.mkdir(mode=0o755)
    # The root driver uses UMask=0077. Only these owner-controlled parent
    # directories must be traversable by the separate application account.
    BASE.chmod(0o755)
    owner = Ed25519PrivateKey.generate()
    keys = [Ed25519PrivateKey.generate() for _ in range(2)]
    ids = [key_id(public_bytes(key.public_key())) for key in keys]
    native = "linux-" + architecture
    envelope = json.loads(base64.b64decode(catalog["manifest_base64"]))
    policy_body = {
        "schema": "organism-enrollment-v1", "experiment_id": envelope["body"]["experiment_id"], "epoch": 1,
        "devices": {identity: {"public_key": base64.b64encode(public_bytes(key.public_key())).decode(),
                              "platform": native} for identity, key in zip(ids, keys)},
        "release_keys": catalog["release_keys"], "require_trained": False,
        "max_payload_bytes": 3 * 1024**3, "birth_seconds": 1200,
        "max_births_per_device": 1, "max_process_starts": 1,
    }
    public = public_bytes(owner.public_key())
    policy = Enrollment(sign_enrollment(policy_body, owner), public)
    ca = owner_certificate(owner, days=1)
    for label, key in zip(("device-identity", "bootstrap-identity"), keys):
        write_identity(BASE / label, key, device_certificate(owner, ca, key.public_key(), days=1), ca)
    book = sign_discovery_book({
        "schema": "organism-discovery-book-v1", "policy_id": policy.policy_id, "epoch": 1,
        "lab_cidrs": ["127.0.0.0/24"],
        "endpoints": {ids[0]: {"address": "127.0.0.1", "port": 5443},
                      ids[1]: {"address": "127.0.0.1", "port": 5444}},
    }, owner, policy)
    # The bootstrap identity has no listener and never runs an organism.
    # Private owner and release-signing keys are not installed on either device.
    private_json(BASE / "enrollment.json", json.loads(policy.signed))
    private_json(BASE / "discovery.json", json.loads(book))
    (BASE / "work").mkdir(mode=0o755)
    (BASE / "work").chmod(0o755)
    config = {
        "schema": "organism-receiver-service-v1", "trusted_owner_public_key": base64.b64encode(public).decode(),
        "device_id": ids[0], "bind_address": "127.0.0.1", "bind_port": 5443,
        "lab_cidrs": ["127.0.0.0/24"], "service_user": "organism", "service_group": "organism",
        "enrollment_file": str(BASE / "enrollment.json"), "state_directory": str(BASE / "state"),
        "capsule_store": str(BASE / "capsules"), "prepared_store": str(BASE / "prepared"),
        "work_root": str(BASE / "work"), "identity_directory": str(BASE / "device-identity"),
        "guard_python": str(Path("/usr/bin/python3").resolve()),
        "guard_script": str(Path(__file__).resolve().parents[1] / "start_guard.py"),
        "discovery_file": str(BASE / "discovery.json"), "context_root": str(BASE / "contexts"),
        "network_namespace": NS_PATH,
    }
    # Configuration is durable before any process can be admitted, so the
    # independent host cleanup can identify this device even if the driver dies.
    private_json(BASE / "receiver.json", config)
    ReceiverState.enroll(BASE / "state", policy, ids[0])
    private_json(BASE / "bootstrap.json", {"device_id": ids[1]})
    return config, public, policy, ids[1]


def health(socket_path):
    """Read-only stdlib RPC: the seed driver imports no application or planner."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5)
        connection.connect(str(socket_path))
        connection.sendall(b'{"operation":"health"}\n')
        raw = bytearray()
        while not raw.endswith(b"\n"):
            block = connection.recv(4096)
            if not block or len(raw) + len(block) > 65536:
                raise ValueError("Invalid application health response")
            raw.extend(block)
        return json.loads(raw)


def drive(service, client, endpoint, manifest, capsule, *, deadline_seconds=1200):
    """Exercise the unchanged service lifecycle after one owner seed transfer."""
    report = {"seed_provisioned_by_owner": True, "autonomous_reproduction_verified": False,
              "qualification_passed": False, "ready": False, "cleanup_verified": False}
    expected_release = json.loads(manifest)["release_id"]
    errors = []

    def serve():
        try:
            service.run()
        except BaseException as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))

    worker = threading.Thread(target=serve, name="native-receiver-owner", daemon=True)
    worker.start()
    birth = None
    try:
        if not service.started.wait(10) or not worker.is_alive():
            raise RuntimeError("Native receiving service did not start")
        deadline = time.monotonic() + deadline_seconds
        operation = uuid.uuid4().hex
        report["operation_id"] = operation
        report["transfer_receipt"] = client.transfer(endpoint, operation, manifest, capsule)
        while time.monotonic() < deadline:
            if errors or not worker.is_alive():
                raise RuntimeError("Receiving service stopped: " + "; ".join(errors))
            receipt = client.request(endpoint, "status", operation_id=operation)
            snapshot = service.state.snapshot()
            birth = snapshot["current_birth"]
            if receipt["status"] == "ready" and receipt["ready_evidence"]:
                observed = service.supervisor.backend.inspect(birth)
                admitted = service.supervisor.backend.records.get(birth, "check_admission")
                if (snapshot["owner_mode"] != "enabled" or birth["status"] != "ready"
                        or birth["ready_evidence"] != receipt["ready_evidence"] or admitted is None
                        or admitted["digest"] != receipt["ready_evidence"] or observed["state"] != "running"
                        or receipt.get("operation_id") != operation or birth["operation_id"] != operation
                        or receipt.get("device_id") != endpoint[0] or birth["device_id"] != endpoint[0]
                        or receipt.get("release_id") != expected_release or birth["release_id"] != expected_release
                        or not observed["native_observation"]):
                    raise RuntimeError("Current native readiness is not established")
                report.update(ready=True, ready_receipt=receipt, application_observation=observed,
                              ready_observed_at=datetime.now(timezone.utc).isoformat(),
                              ready_observed_clock=service.state.clock())
                break
            if receipt["status"] in ("aborted", "expired", "retired", "failed") or receipt["preparation_allowance_exhausted"]:
                raise RuntimeError("Seed preparation or admission failed")
            if birth and birth["process_token"]:
                checked = service.supervisor.backend.records.get(birth, "check_result")
                if checked and checked["record"]["observation"]["passed"] is False:
                    raise RuntimeError("Carried readiness checker failed: " + str(checked["record"]["observation"]["failure"]))
            service.stop_requested.wait(2)
        else:
            raise TimeoutError("Native seed did not become ready within its original allowance")
        work = service.workspaces.root / birth["individual_id"]
        report["application_health"] = health(work / "agent.sock")
        if (report["application_health"].get("native") is not True
                or report["application_health"].get("inference_ready") is not True
                or report["application_health"].get("start_token") != birth["process_token"]
                or report["application_health"].get("trained_for_organism") is not False
                or report["application_health"].get("discovery_running") is not True
                or report["application_health"].get("reproduction_running") is not True):
            raise RuntimeError("Native application health differs from the admitted seed")
        app = json.loads((work / ("application-" + birth["process_token"] + ".json")).read_bytes())
        intent = service.supervisor.backend.records.get(birth, "intent")["record"]["observation"]
        report["inference_namespace"] = verify_process_namespace(intent["network_namespace"], app["inference"]["pid"])
        report["readiness_records"] = {kind: service.supervisor.backend.records.get(birth, kind) for kind in KINDS}
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        # Fence admission even while the receiving loop is busy with a start.
        try:
            service.state.owner_control("retire", "Native owner seed startup check finished")
        except Exception as exc:
            report["cleanup_failure"] = type(exc).__name__ + ": " + str(exc)
        service.stop_requested.set()
        worker.join(timeout=45)
        report["receiver_loop_stopped"] = not worker.is_alive()
        if worker.is_alive():
            report["cleanup_failure"] = "Receiving loop did not stop; independent host cleanup is required"
        else:
            try:
                result = service.supervisor.reconcile()
                final = service.state.snapshot()
                current = final["current_birth"]
                report["retirement"] = result
                report["final_owner_mode"] = final["owner_mode"]
                report["final_process_token"] = current["process_token"] if current else None
                report["cleanup_verified"] = bool(final["owner_mode"] == "retired"
                                                   and (current is None or current["process_token"] is None)
                                                   and result.get("state") in ("absent", "idle")
                                                   and not report.get("cleanup_failure"))
                if birth and birth["process_token"]:
                    report["exit_record"] = service.supervisor.backend.records.get(birth, "exit")
            except Exception as exc:
                report["cleanup_failure"] = type(exc).__name__ + ": " + str(exc)
        report["passed"] = report["ready"] and report["cleanup_verified"] and not report.get("failure")
    return report


def retire_native():
    config, public = configuration(BASE / "receiver.json")
    state, host = native_state(config, public)
    return apply_owner_control(state, lambda: native_backend(state, host, config), "retire",
                               "Independent native host cleanup")


def run_native(architecture):
    if platform.system() != "Linux" or os.geteuid() != 0:
        raise RuntimeError("Native seed preparation requires an owner Linux process")
    if platform.machine() != {"arm64": "aarch64", "x86_64": "x86_64"}[architecture]:
        raise RuntimeError("Seed machine differs from the requested native architecture")
    binding = inspect_namespace(NS_PATH)
    verify_process_namespace(binding, os.getpid())
    if set(path.name for path in Path("/sys/class/net").iterdir()) != {"lo"}:
        raise RuntimeError("Native seed driver has an external network interface")
    catalog = validate_catalog(json.loads((ROOT / "catalog.json").read_bytes()))
    manifest = base64.b64decode(catalog["manifest_base64"])
    keys = {name: base64.b64decode(value) for name, value in catalog["release_keys"].items()}
    verify_capsule_directory(ROOT / "capsule", manifest, keys, require_trained=False)
    config, public, policy, bootstrap = provision(catalog, architecture)
    # Exercise the same configuration parser and native factory as the CLI.
    config, public = configuration(BASE / "receiver.json")
    state, host = native_state(config, public)
    supervisor, preparer, workspaces, contexts = native_components(config, state, host, public)
    network = LabNetwork(config["lab_cidrs"])
    peer = PeerServer(supervisor.receiver, config["identity_directory"],
                      bind=(config["bind_address"], config["bind_port"]), network=network)
    service = ReceiverService(supervisor, preparer, workspaces, peer, lab_cidrs=config["lab_cidrs"],
                              user="organism", group="organism", node_contexts=contexts)
    client = PeerClient(policy, bootstrap, BASE / "bootstrap-identity", network=network)
    report = drive(service, client, (state.device_id, "127.0.0.1", 5443), manifest, ROOT / "capsule")
    report.update(architecture=architecture, release_id=catalog["release_id"],
                  completed_at=datetime.now(timezone.utc).isoformat(), namespace=binding)
    write_json(OUTPUT / "seed-receipt.json", report)
    return report
