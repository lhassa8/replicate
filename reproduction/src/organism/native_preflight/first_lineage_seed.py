"""Owner cold-start observer; only the enrolled receiver launches the seed."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from ..audit_equipment_diagnostics import require
from ..network_namespace import verify_process_namespace
from ..supervision_records import KINDS
from ..systemd_supervision import process_start_ticks
from . import device_capacity, first_lineage_profile, first_lineage_tasks, profile_census
from .startup_seed import health, private_json


def check_incarnations(records):
    require(len(records) == first_lineage_profile.COLD_STARTS, "Three cold starts are required")
    first = records[0]["birth"]
    seen_tokens, seen_applications, seen_models = set(), set(), set()
    for ordinal, record in enumerate(records, 1):
        birth = record["birth"]
        require(all(birth[key] == first[key] for key in (
            "device_id", "individual_id", "release_id", "operation_id", "deadline"))
            and birth["starts"] == ordinal and birth["status"] == "ready"
            and birth["process_token"] not in seen_tokens,
            "Cold start changed its individual, original birth deadline, or start count")
        seen_tokens.add(birth["process_token"])
        for role, seen in (("application", seen_applications), ("inference", seen_models)):
            identity = tuple(record["roles"][role][key] for key in ("pid", "start_ticks"))
            require(identity not in seen, "A cold start reused a prior application or model process")
            seen.add(identity)
        require(record["owner_census"]["passed"] is True
                and record["application_observation"]["native_observation"] is True
                and record["application_observation"]["state"] == "running",
                "Cold start lacks native process and owner observations")
        first_lineage_tasks.check_health(record["application_health"], birth["process_token"])
        if ordinal < first_lineage_profile.COLD_STARTS:
            stopped = record["stopped"]
            require(stopped["state"] == "absent" and stopped["native_observation"] is True
                    and stopped["cgroup_populated"] is False,
                    "The prior application and model process group did not stop")
    return True


def snapshot_record(report, birth, roles):
    return deepcopy({
        "birth": birth, "roles": roles,
        **{key: report[key] for key in (
            "ready_receipt", "application_observation", "receiver_ready", "ready_observed_at",
            "ready_observed_clock", "namespace",
            "application_health", "inference_namespace", "readiness_records",
            "host_capacity_ready", "owner_census")},
    })


def wait_restarted(*, old, ordinal, report, state, host, backend, client, endpoint,
                   operation, catalog, config, output):
    from .profile_seed import receiver_observation, require_start_possible
    started = time.monotonic()
    deadline = started + first_lineage_profile.START_SECONDS
    while time.monotonic() < deadline:
        receiver = receiver_observation(host)
        receipt = client.request(endpoint, "status", operation_id=operation)
        snapshot = state.snapshot()
        birth = snapshot["current_birth"]
        require(snapshot["owner_mode"] == "enabled" and birth is not None
                and all(birth[key] == old[key] for key in (
                    "device_id", "individual_id", "release_id", "operation_id", "deadline")),
                "Receiver lost the original individual during the cold start")
        require(birth["starts"] <= ordinal, "An unrequested extra process start occurred")
        if (birth["process_token"] != old["process_token"]
                and birth["starts"] == ordinal and birth["status"] == "ready"
                and receipt["status"] == "ready" and receipt["ready_evidence"]):
            observed = backend.inspect(birth)
            admitted = backend.records.get(birth, "check_admission")
            require(admitted is not None and birth["ready_evidence"] == receipt["ready_evidence"]
                    == admitted["digest"] and observed["state"] == "running"
                    and observed["native_observation"] is True
                    and receipt.get("operation_id") == operation
                    and receipt.get("device_id") == birth["device_id"]
                    and receipt.get("release_id") == catalog["release_id"],
                    "Restarted native readiness differs from local ownership")
            break
        require(receipt["status"] not in ("aborted", "expired", "retired", "failed")
                and not receipt["preparation_allowance_exhausted"], "Native cold start failed")
        require_start_possible(birth, state.policy.body["max_process_starts"])
        if birth["process_token"] != old["process_token"] and birth["process_token"]:
            checked = backend.records.get(birth, "check_result")
            require(not checked or checked["record"]["observation"]["passed"] is not False,
                    "Restarted carried readiness checker failed")
        time.sleep(2)
    else:
        raise TimeoutError("Native cold start exceeded its 20-minute observation allowance")
    report.update(ready_receipt=receipt, application_observation=observed, receiver_ready=receiver,
                  ready_observed_at=datetime.now(timezone.utc).isoformat(),
                  ready_observed_clock=state.clock())
    work = Path(config["work_root"]) / birth["individual_id"]
    report["application_health"] = health(work / "agent.sock")
    first_lineage_tasks.check_health(report["application_health"], birth["process_token"])
    app = json.loads(first_lineage_tasks.bounded_file(
        work / ("application-" + birth["process_token"] + ".json"),
        uid=work.stat().st_uid, maximum=first_lineage_tasks.MAX_RPC_BYTES))
    binding = {key: birth[key] for key in ("device_id", "individual_id", "release_id")}
    binding["start_token"] = birth["process_token"]
    require(all(app.get(key) == value for key, value in binding.items()),
            "Restarted application receipt changed its identity")
    intent = backend.records.get(birth, "intent")["record"]["observation"]
    report["namespace"] = intent["network_namespace"]
    report["inference_namespace"] = verify_process_namespace(intent["network_namespace"], app["inference"]["pid"])
    report["readiness_records"] = {kind: backend.records.get(birth, kind) for kind in KINDS}
    report["host_capacity_ready"] = device_capacity.observe(
        expected_roles=report["host_capacity_before"]["roles"])
    pids = {"receiver": receiver["resources"]["receiver"]["pid"],
            "application": observed["process"]["pid"], "inference": app["inference"]["pid"]}
    roles = {role: {"pid": pid, "start_ticks": int(process_start_ticks(
        (host.proc_root / str(pid) / "stat").read_text(), pid))} for role, pid in pids.items()}
    census_directory = output / ("cold-start-" + str(ordinal).zfill(2))
    census_directory.mkdir(mode=0o700)
    report["owner_census"] = profile_census.request_ready(
        census_directory, release_id=catalog["release_id"],
        request_id=birth["process_token"], roles=roles)
    require(time.monotonic() < deadline, "Cold-start evidence exceeded its observation allowance")
    record = snapshot_record(report, birth, roles)
    record["observation_seconds"] = time.monotonic() - started
    return birth, record


def complete(report, birth, *, config, state, host, backend, client, endpoint,
             operation, catalog, output, cases, first_roles):
    output = Path(output)
    require(state.policy.body["max_process_starts"] == first_lineage_profile.COLD_STARTS
            and state.policy.body["max_births_per_device"] == 1
            and state.policy.body["birth_seconds"] == first_lineage_profile.START_SECONDS
            and birth["starts"] == 1, "Cold-start enrollment or first incarnation differs")
    report["first_lineage"] = first_lineage_profile.descriptor()
    report["cold_starts_verified"] = False
    records = report["cold_starts"] = [snapshot_record(report, birth, first_roles)]
    for ordinal in (2, 3):
        stopped = backend.stop(birth)
        records[-1]["stopped"] = stopped
        private_json(output / ("cold-start-" + str(ordinal - 1).zfill(2) + "-receipt.json"), records[-1])
        require(stopped["state"] == "absent" and stopped["native_observation"] is True
                and stopped["cgroup_populated"] is False,
                "Cold start requires the previous entire process group to be absent")
        # The separate receiver observes the exit and starts the same admitted
        # individual. This owner neither claims a start nor spawns an application.
        birth, record = wait_restarted(
            old=birth, ordinal=ordinal, report=report, state=state, host=host, backend=backend,
            client=client, endpoint=endpoint, operation=operation, catalog=catalog,
            config=config, output=output)
        records.append(record)
    private_json(output / "cold-start-03-receipt.json", records[-1])
    report["cold_starts_verified"] = check_incarnations(records)
    binding = {key: birth[key] for key in ("device_id", "individual_id", "release_id")}
    binding["start_token"] = birth["process_token"]
    work = Path(config["work_root"]) / birth["individual_id"]
    report["temperature_tasks"] = first_lineage_tasks.run(
        work, output / "temperature-tasks", cases, first_lineage_profile.CASES_SHA256,
        binding=binding, application_pid=report["application_observation"]["process"]["pid"],
        uid=work.stat().st_uid)
    require(report["temperature_tasks"]["tasks_passed"] is True, "Native frozen temperature tasks failed")
    return birth
