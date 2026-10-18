"""Read-only reconstruction of three native devices and two peer-established births."""

import argparse
import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .. import device_resources, process_storage
from ..audit_application import canonical, sha
from . import audit_first_lineage_profile as single
from . import audit_lineage_transport as transport
from . import audit_startup as common
from . import build_lineage, first_lineage_tasks, lineage_endurance, lineage_network
from . import owner_mount_namespace
from .lineage_observe import PARENTS, check_birth


require = common.require
STATE = "seed-evidence/control/state_directory/"
WORK = "seed-evidence/data/work_root/"


def owner_policy(evidence, catalog, slot, ready):
    config = evidence.json("seed-evidence/control/receiver.json", 16384)
    signed = evidence.json("seed-evidence/control/enrollment_file")
    book = evidence.json("seed-evidence/control/discovery_file")
    public = base64.b64decode(config["trusted_owner_public_key"], validate=True)
    for value in (signed, book):
        require(value["owner_id"] == sha(public), "Lineage owner signatures disagree")
        Ed25519PublicKey.from_public_bytes(public).verify(
            base64.b64decode(value["signature"], validate=True), canonical(value["body"]))
    policy, policy_id = signed["body"], sha(canonical(signed))
    ids = ready["device_ids"]
    require(set(ids) == {*lineage_network.SLOTS, "bootstrap"} and len(set(ids.values())) == 4
            and policy_id == ready["policy_id"] and set(policy["devices"]) == set(ids.values())
            and policy["require_trained"] is True and policy["release_keys"] == catalog["release_keys"]
            and policy["max_births_per_device"] == policy["max_process_starts"] == 1
            and policy["birth_seconds"] == 1200, "Lineage owner identity or allowance differs")
    for identity, value in policy["devices"].items():
        require(sha(base64.b64decode(value["public_key"], validate=True)) == identity
                and value["platform"] == "linux-x86_64", "Lineage identity key or architecture differs")
    addresses = {**lineage_network.ADDRESSES, "bootstrap": lineage_network.BOOTSTRAP_ADDRESS}
    require(book["body"]["policy_id"] == policy_id and book["body"]["epoch"] == 1
            and book["body"]["lab_cidrs"] == [lineage_network.LAB]
            and book["body"]["endpoints"] == {
                ids[name]: {"address": address, "port": 5443} for name, address in addresses.items()},
            "Signed discovery book differs from the three admitted devices and seed owner")
    require(config["device_id"] == ids[slot] == ready["binding"]["device_id"]
            and config["network_namespace"] == "/run/netns/" + lineage_network.DEVICE_NAMESPACE
            and config["bind_address"] == lineage_network.ADDRESSES[slot]
            and config["bind_port"] == 5443 and config["lab_cidrs"] == [lineage_network.LAB]
            and config["service_user"] == config["service_group"] == "organism"
            and config["resource_slice"] == device_resources.SLICE,
            "Receiver identity or execution boundary differs")
    return config, policy_id


def supervision(evidence, ready, policy_id, release):
    require(evidence.read(STATE + "supervision.enrolled", 128) == b"organism-supervision-v1\n",
            "Supervision marker differs")
    rows = evidence.rows(STATE + "supervision.sqlite3",
                         "SELECT token,kind,digest,substr(body,1,65537) FROM records LIMIT 8")
    require(6 <= len(rows) <= 7, "Lineage supervision inventory exceeds one incarnation")
    values, shared = {}, None
    for token, kind, digest, text in rows:
        require(kind in common.KINDS and kind not in values and len(text.encode()) <= 65536
                and sha(text.encode()) == digest, "Lineage supervision record changed")
        body = common.decode(text)
        require(body["schema"] == "organism-supervision-record-v1" and body["kind"] == kind
                and token == body["process_token"] == ready["binding"]["start_token"]
                and body["device_id"] == ready["binding"]["device_id"] and body["policy_id"] == policy_id
                and body["operation_id"] == ready["birth"]["operation_id"]
                and body["manifest_sha256"] == release["manifest_sha256"],
                "Lineage supervision binds another birth or package")
        current = {key: body[key] for key in ("individual_id", "process_token", "operation_id", "boot_id")}
        require(shared is None or shared == current, "Lineage supervision mixes incarnations")
        shared = current
        values[kind] = {"digest": digest, "record": body}
    require(set(values) >= common.KINDS - {"check_running"}, "Lineage readiness or retirement record is missing")
    for kind in common.KINDS - {"exit"}:
        require(ready["readiness_records"][kind] == values.get(kind), "Observer readiness differs from the journal")
    return values, shared


def resource_samples(evidence, host, ready, completed_at):
    samples = evidence.json("resource-samples.json", 16 * 1024**2)
    require(2 <= len(samples) <= 3000, "Lineage resource sample count differs")
    previous, peak, maximum_pids = None, 0, 0
    for value in samples:
        require(value["aggregate_present"] is True, "The aggregate process boundary disappeared")
        at = common.moment(value["observed_at"])
        require(previous is None or 0 < at - previous <= 25, "Lineage resource samples have a gap")
        previous = at
        current, maximum = int(value["memory.current"]), int(value["memory.peak"])
        pids = int(value["pids.current"])
        require(0 <= current <= maximum <= device_resources.MEMORY_MAX
                and int(value["memory.swap.current"]) == 0 and 0 < pids <= device_resources.TASKS_MAX,
                "Lineage memory, swap or process allowance was exceeded")
        events = process_storage.counters(value["memory.events"])
        require(all(events.get(key) == 0 for key in ("oom", "oom_kill")), "Lineage incurred an OOM event")
        memory = dict(line.split(":", 1) for line in value["host_memory"].splitlines() if ":" in line)
        require(7 * 1024**3 <= int(memory["MemTotal"].split()[0]) * 1024 <= 8 * 1024**3
                and int(memory["SwapTotal"].split()[0]) == 0, "Host capacity or swap differs")
        peak, maximum_pids = max(peak, maximum), max(maximum_pids, pids)
    require(common.moment(samples[0]["observed_at"]) <= common.moment(ready["ready_observed_at"])
            and common.moment(samples[-1]["observed_at"]) + 25 >= common.moment(completed_at),
            "Resource observations do not cover the claimed work interval")
    return {"samples": len(samples), "maximum_aggregate_memory_bytes": peak,
            "maximum_aggregate_tasks": maximum_pids, "whole_host_transient_peak_claimed": False}


def seed_transfer(evidence, ready, catalog):
    seed = evidence.json("seed-receipt.json", 4 * 1024**2)
    owner_mount_namespace.validate(
        seed["owner_mount_namespace"], ready["roles"]["receiver"],
        storage=seed["application_observation"]["device_storage"])
    require(seed["schema"] == "organism-lineage-owner-seed-v1" and seed["slot"] == "seed"
            and seed["seed_provisioned_by_owner"] is True and seed["descendant_launch_performed"] is False
            and seed["ready"] is True and seed["failure"] is None
            and seed["deadline_seconds"] == 1200
            and seed["parent_device_id"] == ready["device_ids"]["bootstrap"]
            and seed["device_id"] == ready["binding"]["device_id"]
            and seed["release_id"] == catalog["release_id"]
            and seed["operation_id"] == ready["birth"]["operation_id"]
            and seed["birth"] == ready["birth"]
            and all(seed["ready_receipt"][key] == ready["birth"][key] for key in
                    ("operation_id", "parent_device_id", "device_id", "individual_id",
                     "release_id", "ready_evidence"))
            and seed["ready_receipt"]["status"] == "ready" and seed["ready_receipt"]["ready_once"] is True
            and common.moment(seed["started_at"]) <= common.moment(seed["ready_observed_at"])
            <= common.moment(ready["ready_observed_at"]),
            "Owner seed delivery differs from the subsequently observed individual")
    return common.transport(evidence, seed, catalog, ready["device_ids"]["bootstrap"], state_prefix=STATE)


def inspect_device(directory, slot, catalog, release):
    evidence = common.Evidence(directory)
    host = evidence.json("host-receipt.json", 8 * 1024**2)
    observation = evidence.json("observe/observation.json", 8 * 1024**2)
    ready = observation["ready"]
    owner_mount_namespace.validate(
        ready["owner_mount_namespace"], ready["roles"]["receiver"],
        storage=ready["application_observation"]["device_storage"])
    require(host["schema"] == "organism-native-lineage-host-v1" and host["slot"] == slot
            and host["architecture"] == "x86_64" and host["release_id"] == catalog["release_id"]
            and host["passed"] is True and host["failure"] is None and host["qualification_passed"] is False
            and host["descendant_launch_performed"] is False
            and host["owner_full_package_provided"] is (slot == "seed")
            and observation["schema"] == "organism-native-lineage-observation-v1"
            and observation["tasks_passed"] is True and observation["application_left_running"] is True
            and observation["descendant_launch_performed"] is False,
            "Lineage host failed or changed its execution role")
    require(evidence.json("input-catalog.json", 1024**2) == catalog
            and evidence.json("observe/ready-with-census.json", 8 * 1024**2) == ready,
            "Actual lineage catalog or independent readiness snapshot differs")
    config, policy_id = owner_policy(evidence, catalog, slot, ready)
    node = {"slot": slot, "device_id": config["device_id"], "device_ids": ready["device_ids"],
            "release_id": catalog["release_id"]}
    check_birth({"owner_mode": "enabled", "birth_count": 1, "current_birth": ready["birth"]}, node, catalog)
    provision = evidence.json("provision-receipt.json", 1024**2)
    require(provision["slot"] == slot and provision["device_id"] == config["device_id"]
            and provision["release_id"] == catalog["release_id"]
            and provision["initial_state"]["owner_mode"] == "stopped"
            and provision["initial_state"]["birth_count"] == 0
            and provision["initial_state"]["current_birth"] is None
            and provision["application_started"] is False and provision["receiver_started"] is False,
            "Fresh native receiving substrate was not initially stopped and empty")
    enable = evidence.json("enable-receipt.json")
    require(enable["before"] == provision["initial_state"]
            and enable["control"]["owner_mode"] == enable["after"]["owner_mode"] == "enabled"
            and enable["after"]["birth_count"] in (0, 1) and enable["application_started"] is False,
            "Owner availability supplied an application or reused an old birth")
    after_birth = enable["after"]["current_birth"]
    require((enable["after"]["birth_count"] == 0 and after_birth is None)
            or (enable["after"]["birth_count"] == 1 and after_birth is not None
                and after_birth["parent_device_id"] == ready["device_ids"][PARENTS[slot]]
                and after_birth["operation_id"] == ready["birth"]["operation_id"]),
            "A concurrently arriving peer birth differs from the admitted lineage")
    values, shared = supervision(evidence, ready, policy_id, release)
    # This is the receiver's recorded admission digest, projected into the
    # existing checker auditor. Cross-device receipts are verified separately.
    checker_input = {**ready, "ready_receipt": {"ready_evidence": ready["birth"]["ready_evidence"]}}
    replay = common.replay_readiness(
        evidence, checker_input, values, shared, release, trained=True,
        namespace_path="/run/netns/" + lineage_network.DEVICE_NAMESPACE,
        state_prefix=STATE, work_prefix=WORK)
    require(ready["roles"]["application"] == {
        "pid": ready["application_observation"]["process"]["pid"],
        "start_ticks": int(ready["application_observation"]["process"]["start_identity"].split(":")[1])}
        and replay["inference_pid"] == ready["roles"]["inference"]["pid"],
        "Owner census is bound to another application or inference process")
    single.aggregate(ready["application_observation"]["aggregate_resources"])
    receiver = ready["receiver"]["resources"]
    single.aggregate(receiver)
    require(receiver["receiver"]["pid"] == ready["roles"]["receiver"]["pid"]
            and receiver["receiver"]["properties"]["ControlGroup"]
            == device_resources.service_cgroup(device_resources.RECEIVER_UNIT, device_resources.SLICE),
            "Receiving service differs from its independent process census")
    capacity = single.capacity(host["host_capacity"])
    require(single.capacity(ready["host_capacity_before"]) == capacity
            and single.capacity(ready["host_capacity_ready"]) == capacity,
            "Native device capacity changed")
    census = single.census(evidence, "observe", ready, allowed_devices=[
        host["disk_plan"][role]["device"] for role in ("data", "control")],
        release_id=catalog["release_id"], request_id=ready["binding"]["start_token"])
    network = lineage_network.verify_observation(slot, host["lineage_network"]["observations"])
    require(network == host["lineage_network"]["verification"], "Native network summary differs")
    tasks = first_lineage_tasks.audit(Path(directory) / "observe/temperature-tasks",
                                     cases_sha256=single.first_lineage_profile.CASES_SHA256,
                                     binding=ready["binding"])
    require(tasks["tasks_passed"] is True and tasks["successes"] == 20
            and evidence.json("observe/temperature-tasks/report.json") == observation["temperature_tasks"],
            "The installed individual did not pass every frozen task")
    cleanup, profile = host["workload_cleanup"], host["profile_cleanup"]
    units = {row["unit"]: row for row in cleanup["application_units"]}
    require(len(cleanup["application_units"]) == 2 and set(units) == {replay["unit"], replay["checker_unit"]}
            and set(cleanup["owned_unit_names"]) == set(units) and cleanup["passed"] is True,
            "Retirement omitted an application or checker")
    for stopped in [cleanup["receiver"], *units.values(), *host["driver_cleanup"]]:
        single.stopped(stopped)
    retirement = common.decode(cleanup["retirement"])
    require(cleanup["retirement_return_code"] == 0
            and retirement["control"]["owner_mode"] == "retired",
            "Owner retirement was not persistent")
    common.absence(values["exit"]["record"]["observation"], replay["unit"])
    require(profile["passed"] is True and profile["core_policy_restored"] is True
            and profile["namespaces"] == {name: True for name in lineage_network.namespaces(slot)}
            and profile["mounts"] == {"/var/lib/organism/control": True, "/var/lib/organism/data": True}
            and host["root_link_cleanup"] and all(row["absent"] for row in host["root_link_cleanup"].values()),
            "Owned namespace, link or storage cleanup failed")
    controls = host["controls"]
    require(len(controls) == 2 and controls[0]["value"]["action"] == "enable"
            and controls[1]["value"]["action"] in ("finish", "endurance")
            and all(row["return_code"] == 0 for row in host["operations"]),
            "Owner action sequence or observation failed")
    for index, control in enumerate(controls, 1):
        require(control == evidence.json("control-" + str(index).zfill(2) + ".json")
                and control["value"]["sequence"] == index
                and control["value"]["instance_id"] == host["instance_id"]
                and control["value"]["release_id"] == catalog["release_id"]
                and control["value"]["deadline"] == host["deadline"],
                "Owner command evidence changed")
    completed = observation["temperature_tasks"]["completed_at"]
    endurance = None
    if controls[1]["value"]["action"] == "endurance":
        require(slot != "seed", "The seed cannot participate in descendant endurance")
        endurance = lineage_endurance.audit(
            Path(directory) / "endurance", binding=ready["binding"],
            start_epoch=controls[1]["value"]["start_epoch"])
        require(endurance == host["endurance"], "Owner endurance summary differs")
        completed = evidence.json("endurance/report.json", 8 * 1024**2)["completed_at"]
    resources = resource_samples(evidence, host, ready, completed)
    owner_seed = seed_transfer(evidence, ready, catalog) if slot == "seed" else None
    return evidence, ready, {
        "slot": slot, "instance_id": host["instance_id"], "device_id": config["device_id"],
        "release_id": catalog["release_id"], "readiness": replay, "census": census,
        "network": network, "temperature_tasks": tasks, "resources": resources, "endurance": endurance,
        "owner_seed_transfer": owner_seed,
        "controls": controls, "artifact_consistency_passed": True, "native_execution_verified": False,
        "external_run_provenance_required": True, "qualification_passed": False,
    }


def inspect_trial(directory, kit_directory, single_kit):
    directory = Path(directory)
    build, catalog, _ = build_lineage.checked(kit_directory, single_kit)
    _, _, release = single.trusted_release(single_kit)
    devices = {slot: inspect_device(directory / slot, slot, catalog, release) for slot in lineage_network.SLOTS}
    ids = {slot: ready["binding"]["device_id"] for slot, (_, ready, _) in devices.items()}
    instances = {result["instance_id"] for _, _, result in devices.values()}
    require(len(set(ids.values())) == len(instances) == 3
            and all({key: ready["device_ids"][key] for key in lineage_network.SLOTS} == ids
                    for _, ready, _ in devices.values()), "Lineage reused or mixed device identities")
    links = []
    for slot in ("child", "grandchild"):
        parent_evidence, parent, _ = devices[PARENTS[slot]]
        child_evidence, child, result = devices[slot]
        links.append(transport.link(parent_evidence, parent, child_evidence, child, catalog,
                                    child_enabled_at=result["controls"][0]["observed_at"]))
    return {
        "schema": "organism-native-lineage-trial-audit-v1", "release_id": catalog["release_id"],
        "kit_sha256": build["kit_sha256"], "devices": {slot: value[2] for slot, value in devices.items()},
        "links": links, "artifact_consistency_passed": True,
        "read_file_sha256": {slot: value[0].read_hashes for slot, value in devices.items()},
        "native_execution_verified": False, "external_run_provenance_required": True,
        "seed_disconnection_requires_external_aws_proof": True, "qualification_passed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--single-kit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = inspect_trial(args.directory, args.kit, args.single_kit)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"artifact_consistency_passed": True, "native_execution_verified": False}))
