"""Owner receiving-substrate provisioning and the single permitted seed delivery."""

import base64
from datetime import datetime, timezone
import grp
import json
import os
from pathlib import Path
import platform
import pwd
import shutil
import time
import uuid

from .. import device_storage, device_resources, network_budget
from ..capsule import verify_capsule_directory
from ..network_namespace import inspect_namespace, verify_process_namespace
from ..peer_identity import LabNetwork
from ..peer_transport import PeerClient
from ..receiver_service import configuration, native_backend, native_state
from ..receiver_state import ReceiverState
from ..systemd_supervision import NativeLinuxHost
from ..training_inputs import require
from . import device_capacity, lineage_enrollment, lineage_network, owner_mount_namespace
from .profile_seed import layout, wait_listener


CONFIG = device_storage.ROOT / "control/receiver.json"
NODE = device_storage.ROOT / "control/lineage-node.json"
BOOTSTRAP = device_storage.ROOT / "control/bootstrap"


def provision(catalog, bundle, slot, *, guard_python, guard_script,
              service_user="organism", service_group="organism"):
    """Initialize a fresh stopped receiver, before its independent service starts."""
    require(platform.system() == "Linux" and platform.machine() == "x86_64" and os.geteuid() == 0,
            "Lineage owner provisioning requires its admitted native x86-64 Linux host")
    node, policy, book = lineage_enrollment.verify(bundle, slot, catalog)
    account = pwd.getpwnam(service_user)
    require(account.pw_uid > 0 and grp.getgrnam(service_group).gr_gid == account.pw_gid,
            "The application requires its distinct unprivileged primary account")
    paths = layout()
    require(not any(os.path.lexists(path) for path in (CONFIG, NODE, BOOTSTRAP, *paths.values())),
            "Do not reuse lineage owner state")
    capacity = device_capacity.observe()
    storage = device_storage.admit(NativeLinuxHost(), paths)
    namespace = inspect_namespace("/run/netns/" + lineage_network.DEVICE_NAMESPACE)
    for name, field in (("enrollment.json", "enrollment_file"), ("discovery.json", "discovery_file")):
        lineage_enrollment.private_write(Path(paths[field]), (Path(bundle) / name).read_bytes())
    shutil.copytree(Path(bundle) / "identity", paths["identity_directory"])
    if slot == "seed":
        shutil.copytree(Path(bundle) / "bootstrap", BOOTSTRAP)
    lineage_enrollment.private_write(NODE, (Path(bundle) / "node.json").read_bytes())
    Path(paths["work_root"]).mkdir(mode=0o711)
    Path(paths["work_root"]).chmod(0o711)
    config = {
        "schema": "organism-receiver-service-v1", **paths,
        "trusted_owner_public_key": node["owner_public"], "device_id": node["device_id"],
        "bind_address": node["bind_address"], "bind_port": 5443,
        "lab_cidrs": [lineage_network.LAB], "service_user": service_user, "service_group": service_group,
        "guard_python": str(guard_python), "guard_script": str(guard_script),
        "resource_slice": device_resources.SLICE, "storage_profile": device_storage.PROFILE,
        "network_budget": network_budget.PROFILE,
        "network_namespace": "/run/netns/" + lineage_network.DEVICE_NAMESPACE,
    }
    lineage_enrollment.private_write(CONFIG, json.dumps(config, indent=2).encode() + b"\n")
    configuration(CONFIG)
    state = ReceiverState.enroll(paths["state_directory"], policy, node["device_id"])
    stopped = state.owner_control("stop", "Fresh lineage device awaits the owner availability event")
    snapshot = state.snapshot()
    require(snapshot["owner_mode"] == "stopped" and snapshot["birth_count"] == 0
            and snapshot["current_birth"] is None, "Fresh device did not retain its owner admission fence")
    return {
        "schema": "organism-lineage-provision-v1", "slot": slot, "device_id": node["device_id"],
        "release_id": node["release_id"], "policy_id": policy.policy_id, "book_id": book.book_id,
        "host_capacity": capacity, "storage": storage, "namespace": namespace,
        "initial_control": stopped, "initial_state": snapshot,
        "receiver_started": False, "application_started": False, "qualification_passed": False,
    }


def seed_once(catalog, capsule, *, output):
    """Only the owner seed transfer; the receiver alone starts its application."""
    require(platform.system() == "Linux" and os.geteuid() == 0,
            "Seed delivery requires the admitted native owner")
    node = json.loads(NODE.read_bytes())
    require(node["slot"] == "seed" and node["release_id"] == catalog["release_id"],
            "The owner transfer is restricted to the exact seed slot and release")
    mount_observation = owner_mount_namespace.join_receiver()
    config, public = configuration(CONFIG)
    state, host = native_state(config, public, admit_storage=True)
    backend = native_backend(state, host, config)
    namespace = inspect_namespace("/run/netns/" + lineage_network.BOOTSTRAP_NAMESPACE)
    verify_process_namespace(namespace, os.getpid())
    manifest = base64.b64decode(catalog["manifest_base64"], validate=True)
    keys = {name: base64.b64decode(value, validate=True) for name, value in catalog["release_keys"].items()}
    verify_capsule_directory(capsule, manifest, keys, require_trained=True)
    require(state.device_id == node["device_id"] and state.policy.policy_id == node["policy_id"],
            "Seed receiver differs from its assigned owner identity")
    before = state.snapshot()
    require(before["owner_mode"] == "enabled" and before["birth_count"] == 0
            and before["current_birth"] is None, "Seed availability was not explicitly admitted")
    client = PeerClient(state.policy, node["device_ids"]["bootstrap"], BOOTSTRAP,
                        network=LabNetwork([lineage_network.LAB]))
    endpoint = (node["device_id"], node["bind_address"], 5443)
    operation = uuid.uuid4().hex
    output = Path(output)
    report = {
        "schema": "organism-lineage-owner-seed-v1", "slot": "seed",
        "device_id": node["device_id"], "release_id": node["release_id"],
        "parent_device_id": node["device_ids"]["bootstrap"], "operation_id": operation,
        "started_at": datetime.now(timezone.utc).isoformat(), "deadline_seconds": 1200,
        "bootstrap_namespace": namespace, "seed_provisioned_by_owner": True,
        "owner_mount_namespace": mount_observation,
        "descendant_launch_performed": False, "ready": False, "failure": None,
        "qualification_passed": False,
    }

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)

    require(not output.exists(), "Do not replace a prior seed transfer")
    deadline = time.monotonic() + 1200
    save()
    try:
        observation = wait_listener(client, endpoint, host)
        report["initial_peer_observation"] = observation
        report["transfer"] = client.transfer(endpoint, operation, manifest, capsule, observation=observation)
        save()
        while time.monotonic() < deadline:
            status = client.request(endpoint, "status", operation_id=operation)
            snapshot = state.snapshot()
            birth = snapshot["current_birth"]
            if status["status"] == "ready":
                observed = backend.inspect(birth)
                receiver = owner_mount_namespace.receiver_identity(host)
                owner_mount_namespace.validate(
                    mount_observation,
                    {key: receiver[key] for key in ("pid", "start_ticks")},
                    storage=observed["device_storage"])
                checked = backend.records.get(birth, "check_admission")
                require(checked is not None and observed["state"] == "running"
                        and observed["native_observation"] is True
                        and snapshot["owner_mode"] == "enabled" and birth["status"] == "ready"
                        and status["device_id"] == birth["device_id"] == node["device_id"]
                        and status["release_id"] == birth["release_id"] == node["release_id"]
                        and status["operation_id"] == birth["operation_id"] == operation
                        and birth["parent_device_id"] == node["device_ids"]["bootstrap"]
                        and birth["ready_evidence"] == status["ready_evidence"] == checked["digest"],
                        "Seed readiness differs from its actual receiver, process or checker")
                report.update(ready=True, birth=birth, ready_receipt=status,
                              application_observation=observed,
                              ready_observed_clock=state.clock(),
                              ready_observed_at=datetime.now(timezone.utc).isoformat())
                break
            require(status["status"] not in ("aborted", "expired", "retired", "failed")
                    and not status["preparation_allowance_exhausted"],
                    "The owner seed could not complete its original birth allowance")
            time.sleep(2)
        else:
            raise TimeoutError("The owner seed did not become ready within twenty minutes")
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    save()
    return report
