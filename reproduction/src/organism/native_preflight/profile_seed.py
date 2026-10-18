"""Owner seed test with a separate real receiver service and all device profiles.

The driver sends one seed through authenticated transport. It never creates an
application, substitutes a checker, or represents that seed as a descendant.
An independent host controller owns the mounts, namespaces, service and expiry.
"""

import base64
from datetime import datetime, timezone
import grp
import json
import os
from pathlib import Path
import platform
import pwd
import time
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .. import device_resources, device_storage, network_budget
from ..capsule import verify_capsule_directory
from ..discovery_book import sign_discovery_book
from ..enrollment import Enrollment, key_id, public_bytes, sign_enrollment
from ..network_namespace import inspect_namespace, verify_process_namespace
from ..peer_identity import LabNetwork, device_certificate, owner_certificate, write_identity
from ..peer_transport import PeerClient
from ..receiver_service import configuration, native_backend, native_state, apply_owner_control
from . import owner_mount_namespace
from ..receiver_state import ReceiverState
from ..supervision import Supervisor
from ..supervision_records import KINDS
from ..systemd_supervision import NativeLinuxHost, process_start_ticks
from .startup_seed import health, private_json
from .trained_inputs import validate_catalog
from . import device_capacity
from . import profile_census
from . import first_lineage_profile


CONFIG = device_storage.ROOT / "control/receiver.json"
BOOTSTRAP = device_storage.ROOT / "control/bootstrap"
RECEIVER_NAMESPACE = "/run/netns/organism-device"
BOOTSTRAP_NAMESPACE = "/run/netns/organism-bootstrap"
LAB = "10.73.0.0/30"
RECEIVER_ADDRESS = "10.73.0.1"
BOOTSTRAP_ADDRESS = "10.73.0.2"


def layout():
    result = {name: str(device_storage.ROOT / "data" / name)
              for name in device_storage.DATA_PATHS}
    result.update({name: str(device_storage.ROOT / "control" / name)
                   for name in device_storage.CONTROL_PATHS})
    return result


def provision(catalog, capsule, architecture, *, guard_python, guard_script,
              service_user="organism", service_group="organism", cases_sha256=None):
    """Provision a fresh owner seed; all volume/network creation is external."""
    if platform.system() != "Linux" or os.geteuid() != 0:
        raise RuntimeError("Combined seed provisioning requires the owner Linux host")
    if architecture not in ("x86_64", "arm64") or platform.machine() != {
            "x86_64": "x86_64", "arm64": "aarch64"}[architecture]:
        raise RuntimeError("Requested native architecture differs from the device")
    validate_catalog(catalog)
    first_lineage = first_lineage_profile.validate_selection(cases_sha256, architecture)
    manifest = base64.b64decode(catalog["manifest_base64"], validate=True)
    release_keys = {name: base64.b64decode(value, validate=True)
                    for name, value in catalog["release_keys"].items()}
    verify_capsule_directory(capsule, manifest, release_keys, require_trained=True)
    account = pwd.getpwnam(service_user)
    if account.pw_uid == 0 or grp.getgrnam(service_group).gr_gid != account.pw_gid:
        raise ValueError("The application requires a distinct unprivileged primary account")
    paths = layout()
    if any(os.path.lexists(path) for path in (CONFIG, BOOTSTRAP, *paths.values())):
        raise FileExistsError("Combined seed provisioning must not reuse owner state")
    device_capacity.observe()
    device_storage.admit(NativeLinuxHost(), paths)
    receiving = inspect_namespace(RECEIVER_NAMESPACE)
    sending = inspect_namespace(BOOTSTRAP_NAMESPACE)
    if (receiving["device"], receiving["inode"]) == (sending["device"], sending["inode"]):
        raise RuntimeError("Bootstrap and receiving service must use distinct namespaces")
    owner = Ed25519PrivateKey.generate()
    keys = [Ed25519PrivateKey.generate() for _ in range(2)]
    ids = [key_id(public_bytes(key.public_key())) for key in keys]
    body = {
        "schema": "organism-enrollment-v1", "experiment_id": json.loads(manifest)["body"]["experiment_id"],
        "epoch": 1, "devices": {
            identity: {"public_key": base64.b64encode(public_bytes(key.public_key())).decode(),
                       "platform": "linux-" + architecture}
            for identity, key in zip(ids, keys)},
        "release_keys": catalog["release_keys"], "require_trained": True,
        "max_payload_bytes": 3 * 1024**3, "birth_seconds": 1200,
        "max_births_per_device": 1,
        "max_process_starts": first_lineage_profile.COLD_STARTS if first_lineage else 1,
    }
    public = public_bytes(owner.public_key())
    policy = Enrollment(sign_enrollment(body, owner), public)
    private_json(Path(paths["enrollment_file"]), json.loads(policy.signed))
    ca = owner_certificate(owner, days=1)
    for path, key in zip((Path(paths["identity_directory"]), BOOTSTRAP), keys):
        write_identity(path, key, device_certificate(owner, ca, key.public_key(), days=1), ca)
    book = sign_discovery_book({
        "schema": "organism-discovery-book-v1", "policy_id": policy.policy_id, "epoch": 1,
        "lab_cidrs": [LAB],
        "endpoints": {identity: {"address": address, "port": 5443}
                      for identity, address in zip(ids, (RECEIVER_ADDRESS, BOOTSTRAP_ADDRESS))},
    }, owner, policy)
    private_json(Path(paths["discovery_file"]), json.loads(book))
    Path(paths["work_root"]).mkdir(mode=0o711)
    Path(paths["work_root"]).chmod(0o711)
    config = {
        "schema": "organism-receiver-service-v1", **paths,
        "trusted_owner_public_key": base64.b64encode(public).decode(),
        "device_id": ids[0], "bind_address": RECEIVER_ADDRESS, "bind_port": 5443,
        "lab_cidrs": [LAB], "service_user": service_user, "service_group": service_group,
        "guard_python": str(guard_python), "guard_script": str(guard_script),
        "resource_slice": device_resources.SLICE, "storage_profile": device_storage.PROFILE,
        "network_budget": network_budget.PROFILE, "network_namespace": RECEIVER_NAMESPACE,
    }
    # Publish owner control before enrolling or admitting any workload.
    private_json(CONFIG, config)
    configuration(CONFIG)
    ReceiverState.enroll(paths["state_directory"], policy, ids[0])
    private_json(BOOTSTRAP / "client.json", {"device_id": ids[1]})
    return config


def receiver_observation(host):
    reply = host.run(["systemctl", "show", device_resources.RECEIVER_UNIT,
                      "--property=MainPID", "--value"])
    if reply.returncode or not reply.stdout.strip().isdigit():
        raise RuntimeError("The independent receiver service PID is unavailable")
    pid = int(reply.stdout.strip())
    if pid == os.getpid():
        raise RuntimeError("The seed driver cannot serve as the receiving service")
    observation = device_resources.verify_receiver(host, pid=pid)
    if observation["native_observation"] is not True:
        raise RuntimeError("A fixture cannot establish a native receiver")
    binding = inspect_namespace(RECEIVER_NAMESPACE)
    return {"resources": observation,
            "namespace": verify_process_namespace(binding, pid)}


def wait_listener(client, endpoint, host, *, timeout=30):
    """Type=exec startup precedes TCP listen; only connection refusal is retryable."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        receiver_observation(host)
        try:
            return client.observe(endpoint)
        except ConnectionRefusedError:
            time.sleep(0.2)
    raise TimeoutError("The separate receiver did not accept an authenticated observation")


def require_start_possible(birth, maximum_starts):
    """Fail a terminal observation without consuming its remaining wait budget."""
    if birth and birth["process_token"] is None and birth["starts"] >= maximum_starts:
        raise RuntimeError("Native process-start allowance is exhausted")


def drive(catalog, capsule, *, deadline_seconds=1200, census_directory=Path("/var/lib/organism-profile-output"),
          cases=None, cases_sha256=None):
    """Run inside the separate bootstrap namespace after owner service start."""
    if type(deadline_seconds) is not int or not 1 <= deadline_seconds <= 1200:
        raise ValueError("Native seed deadline exceeds its original allowance")
    first_lineage = first_lineage_profile.validate_selection(cases_sha256, platform.machine())
    if (cases is not None) != first_lineage:
        raise ValueError("Native observer needs the frozen case file and its checksum")
    if first_lineage:
        first_lineage_profile.validate_cases(cases)
    mount_observation = owner_mount_namespace.join_receiver()
    config, public = configuration(CONFIG)
    state, host = native_state(config, public, admit_storage=True)
    backend = native_backend(state, host, config)
    supervisor = Supervisor(None, backend)
    report = {
        "schema": "organism-native-combined-seed-v1", "release_id": catalog.get("release_id"),
        "seed_provisioned_by_owner": True, "trained_for_organism": True,
        "ready": False, "cleanup_verified": False, "autonomous_reproduction_verified": False,
        "qualification_passed": False,
        "owner_mount_namespace": mount_observation,
    }
    birth = None
    try:
        validate_catalog(catalog)
        bootstrap_binding = inspect_namespace(BOOTSTRAP_NAMESPACE)
        verify_process_namespace(bootstrap_binding, os.getpid())
        receiver_binding = inspect_namespace(RECEIVER_NAMESPACE)
        if ((bootstrap_binding["device"], bootstrap_binding["inode"])
                == (receiver_binding["device"], receiver_binding["inode"])):
            raise RuntimeError("Bootstrap and receiving service must use distinct namespaces")
        report["bootstrap_namespace"] = bootstrap_binding
        report["host_capacity_before"] = device_capacity.observe()
        identity = json.loads((BOOTSTRAP / "client.json").read_bytes())["device_id"]
        policy = Enrollment(Path(config["enrollment_file"]).read_bytes(), public)
        client = PeerClient(policy, identity, BOOTSTRAP, network=LabNetwork(config["lab_cidrs"]))
        endpoint = (state.device_id, config["bind_address"], config["bind_port"])
        manifest = base64.b64decode(catalog["manifest_base64"], validate=True)
        report["receiver_before"] = receiver_observation(host)
        deadline = time.monotonic() + deadline_seconds
        operation = uuid.uuid4().hex
        report["operation_id"] = operation
        observation = wait_listener(client, endpoint, host)
        report["initial_peer_observation"] = observation
        report["transfer_receipt"] = client.transfer(endpoint, operation, manifest, capsule,
                                                    observation=observation)
        while time.monotonic() < deadline:
            receiver = receiver_observation(host)
            receipt = client.request(endpoint, "status", operation_id=operation)
            snapshot = state.snapshot()
            birth = snapshot["current_birth"]
            if receipt["status"] == "ready" and receipt["ready_evidence"]:
                observed = backend.inspect(birth)
                admitted = backend.records.get(birth, "check_admission")
                if (snapshot["owner_mode"] != "enabled" or birth["status"] != "ready"
                        or birth["ready_evidence"] != receipt["ready_evidence"] or admitted is None
                        or admitted["digest"] != receipt["ready_evidence"] or observed["state"] != "running"
                        or receipt.get("operation_id") != operation or birth["operation_id"] != operation
                        or receipt.get("device_id") != endpoint[0] or birth["device_id"] != endpoint[0]
                        or receipt.get("release_id") != catalog["release_id"]
                        or birth["release_id"] != catalog["release_id"] or not observed["native_observation"]):
                    raise RuntimeError("Current combined native readiness is not established")
                report.update(ready=True, ready_receipt=receipt, application_observation=observed,
                              receiver_ready=receiver, ready_observed_at=datetime.now(timezone.utc).isoformat(),
                              ready_observed_clock=state.clock())
                break
            if receipt["status"] in ("aborted", "expired", "retired", "failed") or receipt["preparation_allowance_exhausted"]:
                raise RuntimeError("Combined seed preparation or admission failed")
            require_start_possible(birth, state.policy.body["max_process_starts"])
            if birth and birth["process_token"]:
                checked = backend.records.get(birth, "check_result")
                if checked and checked["record"]["observation"]["passed"] is False:
                    raise RuntimeError("Carried readiness checker failed: " + str(checked["record"]["observation"]["failure"]))
            time.sleep(2)
        else:
            raise TimeoutError("Combined seed did not become ready within its original allowance")
        work = Path(config["work_root"]) / birth["individual_id"]
        report["application_health"] = health(work / "agent.sock")
        expected_health = {
            "native": True, "inference_ready": True, "start_token": birth["process_token"],
            "trained_for_organism": True, "discovery_running": True, "reproduction_running": True,
        }
        if any(report["application_health"].get(key) != value for key, value in expected_health.items()):
            raise RuntimeError("Trained application health differs from the admitted seed")
        app = json.loads((work / ("application-" + birth["process_token"] + ".json")).read_bytes())
        intent = backend.records.get(birth, "intent")["record"]["observation"]
        report["namespace"] = intent["network_namespace"]
        report["inference_namespace"] = verify_process_namespace(intent["network_namespace"], app["inference"]["pid"])
        report["readiness_records"] = {kind: backend.records.get(birth, kind) for kind in KINDS}
        report["host_capacity_ready"] = device_capacity.observe(
            expected_roles=report["host_capacity_before"]["roles"])
        roles = {
            "receiver": report["receiver_ready"]["resources"]["receiver"]["pid"],
            "application": report["application_observation"]["process"]["pid"],
            "inference": app["inference"]["pid"],
        }
        identities = {name: {"pid": pid, "start_ticks": int(process_start_ticks(
            (host.proc_root / str(pid) / "stat").read_text(), pid))} for name, pid in roles.items()}
        owner_mount_namespace.validate(
            mount_observation, identities["receiver"],
            storage=report["application_observation"]["device_storage"])
        report["owner_census"] = profile_census.request_ready(
            census_directory, release_id=catalog["release_id"], request_id=operation, roles=identities)
        if first_lineage:
            from .first_lineage_seed import complete
            birth = complete(
                report, birth, config=config, state=state, host=host, backend=backend,
                client=client, endpoint=endpoint, operation=operation, catalog=catalog,
                output=census_directory, cases=cases, first_roles=identities)
    except BaseException as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    finally:
        try:
            birth = state.snapshot()["current_birth"] or birth
            if birth and birth["process_token"]:
                report["readiness_records"] = {kind: backend.records.get(birth, kind) for kind in KINDS}
        except Exception as exc:
            report["evidence_failure"] = type(exc).__name__ + ": " + str(exc)
        try:
            report["retirement"] = apply_owner_control(
                state, lambda: backend, "retire", "Combined native seed check completed")
            reconciled = supervisor.reconcile()
            snapshot = state.snapshot()
            current = snapshot["current_birth"]
            report["final_owner_mode"] = snapshot["owner_mode"]
            report["cleanup_verified"] = bool(
                snapshot["owner_mode"] == "retired"
                and (current is None or current["process_token"] is None)
                and reconciled.get("state") in ("absent", "idle"))
            if birth and birth["process_token"]:
                report["exit_record"] = backend.records.get(birth, "exit")
        except Exception as exc:
            report["cleanup_failure"] = type(exc).__name__ + ": " + str(exc)
        report["passed"] = bool(report["ready"] and report["cleanup_verified"]
                               and not report.get("failure") and not report.get("evidence_failure"))
    # The independent host still stops the receiving service and removes its
    # namespaces/volumes; retiring an application is not whole-host cleanup.
    report["host_cleanup_verified"] = False
    return report
