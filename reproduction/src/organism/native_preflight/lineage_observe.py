"""Observe a peer-established individual without launching or repairing it."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

from ..network_namespace import verify_process_namespace
from ..receiver_service import configuration, native_backend, native_state
from ..supervision_records import KINDS
from ..systemd_supervision import process_start_ticks
from ..training_inputs import require
from . import device_capacity, first_lineage_tasks, owner_mount_namespace, profile_census
from .first_lineage_profile import CASES_SHA256
from .lineage_provision import CONFIG, NODE
from .profile_seed import receiver_observation
from .trained_inputs import validate_catalog


PARENTS = {"seed": "bootstrap", "child": "seed", "grandchild": "child"}


def check_birth(snapshot, node, catalog):
    """Validate the declared lineage; actual transport proof is audited separately."""
    birth = snapshot["current_birth"]
    require(snapshot["owner_mode"] == "enabled" and snapshot["birth_count"] == 1
            and birth is not None and birth["status"] == "ready"
            and birth["device_id"] == node["device_id"]
            and birth["release_id"] == node["release_id"] == catalog["release_id"]
            and birth["parent_device_id"] == node["device_ids"][PARENTS[node["slot"]]]
            and birth["starts"] == 1 and birth["process_token"] is not None,
            "Observed birth changed its owner, parent, release or single-start allowance")
    return birth


def collect_ready(catalog, output, *, maximum_wait=1200):
    require(platform.system() == "Linux" and platform.machine() == "x86_64" and os.geteuid() == 0,
            "Lineage observation requires an independent native Linux owner")
    require(type(maximum_wait) is int and 1 <= maximum_wait <= 1200,
            "Lineage observation cannot enlarge the original birth allowance")
    validate_catalog(catalog)
    node = json.loads(NODE.read_bytes())
    require(node["slot"] in PARENTS and node["release_id"] == catalog["release_id"],
            "Observer node differs from the admitted release")
    mount_observation = owner_mount_namespace.join_receiver()
    config, public = configuration(CONFIG)
    state, host = native_state(config, public, admit_storage=True)
    backend = native_backend(state, host, config)
    require(state.device_id == node["device_id"] and state.policy.policy_id == node["policy_id"]
            and state.policy.body["birth_seconds"] == 1200
            and state.policy.body["max_births_per_device"] == 1
            and state.policy.body["max_process_starts"] == 1,
            "Lineage receiver admission changed")
    before = device_capacity.observe()
    deadline = time.monotonic() + maximum_wait
    while True:
        require(time.monotonic() < deadline, "No ready peer-established individual within the owner wait")
        receiver = receiver_observation(host)
        snapshot = state.snapshot()
        require(snapshot["owner_mode"] == "enabled", "Owner disabled the device during birth observation")
        birth = snapshot["current_birth"]
        if birth is not None:
            require(birth["parent_device_id"] == node["device_ids"][PARENTS[node["slot"]]],
                    "A different peer established this device")
            if birth["status"] == "ready":
                birth = check_birth(snapshot, node, catalog)
                observed = backend.inspect(birth)
                checked = backend.records.get(birth, "check_admission")
                require(observed["state"] == "running" and observed["native_observation"] is True
                        and checked is not None and checked["digest"] == birth["ready_evidence"],
                        "Receiver readiness lacks its live process or carried checker")
                break
            require(birth["status"] not in ("aborted", "expired", "retired", "failed"),
                    "Peer birth failed or expired")
            if birth["process_token"]:
                check = backend.records.get(birth, "check_result")
                require(check is None or check["record"]["observation"]["passed"] is not False,
                        "The actual carried checker rejected this birth")
        time.sleep(2)
    work = Path(config["work_root"]) / birth["individual_id"]
    binding = {name: birth[name] for name in ("device_id", "individual_id", "release_id")}
    binding["start_token"] = birth["process_token"]
    application_pid, uid = observed["process"]["pid"], work.stat().st_uid
    health = first_lineage_tasks.request(
        work / "agent.sock", {"operation": "health"},
        application_pid=application_pid, uid=uid, deadline=time.monotonic() + 10)
    first_lineage_tasks.check_health(health, binding["start_token"])
    application = json.loads(first_lineage_tasks.bounded_file(
        work / ("application-" + binding["start_token"] + ".json"),
        uid=uid, maximum=first_lineage_tasks.MAX_RPC_BYTES))
    require(all(application.get(name) == value for name, value in binding.items())
            and application.get("native") is True and application.get("trained_for_organism") is True,
            "Application receipt belongs to a different individual")
    intent = backend.records.get(birth, "intent")["record"]["observation"]
    inference_namespace = verify_process_namespace(intent["network_namespace"], application["inference"]["pid"])
    pids = {"receiver": receiver["resources"]["receiver"]["pid"],
            "application": application_pid, "inference": application["inference"]["pid"]}
    roles = {name: {"pid": pid, "start_ticks": int(process_start_ticks(
        (host.proc_root / str(pid) / "stat").read_text(), pid))} for name, pid in pids.items()}
    owner_mount_namespace.validate(
        mount_observation, roles["receiver"], storage=observed["device_storage"])
    ready = {
        "schema": "organism-native-lineage-ready-v1",
        "slot": node["slot"], "device_ids": node["device_ids"], "policy_id": node["policy_id"],
        "binding": binding, "birth": birth, "receiver": receiver, "application": application,
        "application_observation": observed, "health": health, "roles": roles,
        "host_capacity_before": before,
        "host_capacity_ready": device_capacity.observe(expected_roles=before["roles"]),
        "namespace": intent["network_namespace"], "inference_namespace": inference_namespace,
        "readiness_records": {kind: backend.records.get(birth, kind) for kind in KINDS},
        "owner_mount_namespace": mount_observation,
        "ready_observed_at": datetime.now(timezone.utc).isoformat(), "ready_observed_clock": state.clock(),
        "descendant_launch_performed": False, "lineage_transport_verified": False,
        "native_execution_verified": False, "qualification_passed": False,
    }
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    first_lineage_tasks.write(output / "ready.json", ready)
    ready["owner_census"] = profile_census.request_ready(
        output, release_id=catalog["release_id"], request_id=birth["process_token"], roles=roles)
    first_lineage_tasks.write(output / "ready-with-census.json", ready)
    return ready, config


def observe(catalog, output, cases):
    ready, config = collect_ready(catalog, output)
    work = Path(config["work_root"]) / ready["binding"]["individual_id"]
    tasks = first_lineage_tasks.run(
        work, Path(output) / "temperature-tasks", cases, CASES_SHA256, binding=ready["binding"],
        application_pid=ready["roles"]["application"]["pid"], uid=work.stat().st_uid)
    result = {"schema": "organism-native-lineage-observation-v1",
              "ready": ready, "temperature_tasks": tasks,
              "tasks_passed": tasks["tasks_passed"],
              "application_left_running": True, "descendant_launch_performed": False,
              "requires_transport_resource_and_cloud_audit": True,
              "native_execution_verified": False, "qualification_passed": False}
    first_lineage_tasks.write(Path(output) / "observation.json", result)
    return result
