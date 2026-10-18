"""Read-only reconstruction of the single-device first-lineage evidence.

AWS instance admission, archive identity and cloud cleanup are separate inputs.
This auditor never treats synthetic fixtures or a host pass boolean as native
execution and never claims lineage reproduction from an owner-provisioned seed.
"""

import argparse
import base64
import json
from pathlib import Path
import re
import tarfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .. import device_resources, device_storage, process_storage
from ..audit_application import canonical, sha
from . import audit_startup as common
from . import device_capacity, first_lineage_profile, first_lineage_seed, first_lineage_tasks, profile_census
from . import owner_mount_namespace
from .profile_operations import checked_kit
from .profile_host import DRIVER
from .profile_seed import BOOTSTRAP_ADDRESS, LAB, RECEIVER_ADDRESS, RECEIVER_NAMESPACE
from .trained_inputs import validate_catalog


require = common.require
STATE = "seed-evidence/control/state_directory/"
WORK = "seed-evidence/data/work_root/"


def archive_member_hash(path, wanted):
    found = []
    with tarfile.open(path, "r:gz") as archive:
        for index, member in enumerate(archive):
            require(index < 256 and not Path(member.name).is_absolute() and ".." not in Path(member.name).parts,
                    "Native component archive is invalid")
            if member.name == wanted:
                require(member.isfile() and 0 < member.size <= 64 * 1024**2, "Invalid native runtime member")
                with archive.extractfile(member) as stream:
                    found.append(sha(stream.read(member.size + 1)))
    require(len(found) == 1, "Native runtime member is absent or duplicated")
    return found[0]


def trusted_release(kit_directory):
    build, inventory = checked_kit(kit_directory)
    require(build["schema"] == "organism-native-profile-kit-v3"
            and build["first_lineage"] == first_lineage_profile.descriptor(),
            "Expected the separately admitted first-lineage profile")
    kit = common.Evidence(kit_directory)
    catalog = validate_catalog(kit.json("catalog.json", 1024**2))
    manifest_raw = base64.b64decode(catalog["manifest_base64"], validate=True)
    body = common.decode(manifest_raw)["body"]
    native = body["platforms"]["linux-x86_64"]
    release = {
        "release_id": catalog["release_id"], "manifest_sha256": sha(manifest_raw),
        "checker_sha256": archive_member_hash(
            inventory["inputs/capsule/" + body["self_test"]]["path"], "readiness.py"),
        "runtime_sha256": archive_member_hash(
            inventory["inputs/capsule/" + native["inference"]]["path"], "llama-b9850/llama-server"),
        "model_sha256": body["files"][body["model"]["path"]]["sha256"],
        "model_bytes": body["files"][body["model"]["path"]]["bytes"],
    }
    return kit, catalog, release


def owner_policy(evidence, catalog):
    config = evidence.json("seed-evidence/control/receiver.json", 16384)
    signed = evidence.json("seed-evidence/control/enrollment_file")
    book = evidence.json("seed-evidence/control/discovery_file")
    public = base64.b64decode(config["trusted_owner_public_key"], validate=True)
    for value in (signed, book):
        require(value["owner_id"] == sha(public), "Signed owner inputs disagree")
        Ed25519PublicKey.from_public_bytes(public).verify(
            base64.b64decode(value["signature"], validate=True), canonical(value["body"]))
    policy, device, policy_id = signed["body"], common.identity(config["device_id"]), sha(canonical(signed))
    require(policy["require_trained"] is True and policy["release_keys"] == catalog["release_keys"]
            and policy["max_births_per_device"] == 1 and policy["max_process_starts"] == 3
            and policy["birth_seconds"] == 1200 and len(policy["devices"]) == 2
            and device in policy["devices"], "Signed cold-start allowance differs")
    bootstrap = next(key for key in policy["devices"] if key != device)
    for key, value in policy["devices"].items():
        require(sha(base64.b64decode(value["public_key"], validate=True)) == key
                and value["platform"] == "linux-x86_64", "Signed device identity or architecture differs")
    require(book["body"]["policy_id"] == policy_id and book["body"]["lab_cidrs"] == [LAB]
            and book["body"]["endpoints"] == {
                device: {"address": RECEIVER_ADDRESS, "port": 5443},
                bootstrap: {"address": BOOTSTRAP_ADDRESS, "port": 5443}},
            "Signed seed transport endpoints differ")
    require(config["network_namespace"] == RECEIVER_NAMESPACE
            and config["bind_address"] == RECEIVER_ADDRESS and config["bind_port"] == 5443
            and config["lab_cidrs"] == [LAB] and config["service_user"] == config["service_group"] == "organism"
            and config["resource_slice"] == device_resources.SLICE
            and config["storage_profile"] == device_storage.PROFILE,
            "Receiving service profile differs")
    return device, policy_id, bootstrap


def supervision(evidence, *, device, policy_id, operation, release, records):
    require(evidence.read(STATE + "supervision.enrolled", 128) == b"organism-supervision-v1\n",
            "Native supervision marker differs")
    rows = evidence.rows(STATE + "supervision.sqlite3",
                         "SELECT token,kind,digest,substr(body,1,65537) FROM records LIMIT 22")
    tokens = {record["birth"]["process_token"] for record in records}
    require(18 <= len(rows) <= 21 and len(tokens) == 3, "Native supervision has another incarnation inventory")
    by_token = {token: {} for token in tokens}
    for token, kind, digest, text in rows:
        require(token in tokens and kind in common.KINDS and kind not in by_token[token]
                and len(text.encode()) <= 65536 and sha(text.encode()) == digest,
                "Native supervision record is corrupt, duplicated or outside the three starts")
        body = common.decode(text)
        require(body["schema"] == "organism-supervision-record-v1" and body["kind"] == kind
                and body["process_token"] == token and body["device_id"] == device
                and body["policy_id"] == policy_id and body["operation_id"] == operation
                and body["manifest_sha256"] == release["manifest_sha256"],
                "Native supervision binding differs")
        by_token[token][kind] = {"digest": digest, "record": body}
    for record in records:
        token = record["birth"]["process_token"]
        values = by_token[token]
        require(set(values) >= common.KINDS - {"check_running"}, "An incarnation lacks readiness or exit evidence")
        for kind in common.KINDS - {"exit"}:
            require(record["readiness_records"][kind] == values.get(kind), "Cold-start report differs from journal")
        identity = values["intent"]["record"]
        require(all(value["record"]["individual_id"] == record["birth"]["individual_id"]
                    and value["record"]["boot_id"] == identity["boot_id"] for value in values.values()),
                "Supervision journal mixes individuals or host boots")
    return by_token


def capacity(value):
    memory = re.findall(r"^MemTotal:\s+([0-9]+) kB$", value["meminfo"], re.MULTILINE)
    require(len(memory) == 1 and value["native_observation"] is True, "Native memory capacity evidence is absent")
    reconstructed = device_capacity.validate(
        value["block_devices"], device_storage.parse_mountinfo(value["mountinfo"]),
        memory_bytes=int(memory[0]) * 1024, swaps=value["swaps"])
    require(all(value[key] == item for key, item in reconstructed.items()), "Native capacity summary differs")
    return reconstructed


def census(evidence, label, record, *, allowed_devices, release_id, request_id):
    prefix = label + "/" if label else ""
    request = evidence.json(prefix + profile_census.REQUEST)
    response = evidence.json(prefix + profile_census.RESPONSE)
    raw = evidence.read(prefix + profile_census.REPORT, 32 * 1024**2)
    report = common.decode(raw)
    require(profile_census.validate_request(request) == request
            and request["roles"] == record["roles"] and request["request_id"] == request_id
            and request["release_id"] == release_id and report["request"] == request
            and response == record["owner_census"] and response["report_sha256"] == sha(raw)
            and response["passed"] is True and report["passed"] is True
            and report["guard_exit_verified"] is True and report["thaw_verified"] is True
            and report["guard_exit_code"] == 0 and not report.get("failure")
            and not report.get("guard_cleanup_failure"), "Native census binding or thaw evidence differs")
    frames = report["guard_frames"]
    require([frame["phase"] for frame in frames] == ["armed", "frozen", "thawed"]
            and frames[0]["lease_seconds"] == 10 and frames[-1]["reason"] == "thaw"
            and 0 <= report["freeze_interval_seconds"] <= 10,
            "Census independent freeze lease differs")
    observation = report["census"]
    require(observation["allowed_devices"] == allowed_devices
            and observation["cgroup"] == device_resources.CGROUP, "Census observes another device")
    profile_census.check_roles(observation, record["roles"])
    for task in observation["tasks"]:
        require(process_storage.parse_maps(task["maps_raw"]) == task["mappings"],
                "Captured memory map interpretation differs")
    reconstructed = process_storage.classify(observation["tasks"], allowed_devices, null_rdev=259)
    require(all(observation[key] == value for key, value in reconstructed.items())
            and reconstructed["backing_device_violations"] == [],
            "Captured direct file backing differs")
    return {"request_id": request_id, "report_sha256": sha(raw),
            "roles_present": True, "direct_file_backing_reconstructed": True,
            "all_process_writes_contained": False}


def stopped(value):
    properties = value["properties"]
    require(value["absence_verified"] is True and properties["MainPID"] == properties["ControlPID"] == "0"
            and properties["ActiveState"] in ("inactive", "failed")
            and (value["cgroup_events"] is None
                 or process_storage.counters(value["cgroup_events"]).get("populated") == 0),
            "Owned service retains a process")


def aggregate(value):
    require(value["native_observation"] is True and value["slice"] == device_resources.SLICE
            and value["cgroup"] == device_resources.CGROUP
            and value["unit_sha256"] == sha(device_resources.render_slice().encode()),
            "Aggregate observation is not the reviewed native slice")
    props, limits = value["properties"], value["kernel_limits"]
    expected = {"Id": device_resources.SLICE, "LoadState": "loaded", "ActiveState": "active",
                "SubState": "active", "ControlGroup": device_resources.CGROUP,
                "DropInPaths": "", "NeedDaemonReload": "no", "MemoryAccounting": "yes",
                "CPUAccounting": "yes", "TasksAccounting": "yes",
                "MemoryMax": str(device_resources.MEMORY_MAX), "MemorySwapMax": "0",
                "TasksMax": str(device_resources.TASKS_MAX)}
    require(all(props.get(key) == item for key, item in expected.items())
            and props["CPUQuotaPerSecUSec"] in ("4s", "4000000")
            and limits["memory.max"] == str(device_resources.MEMORY_MAX)
            and limits["memory.swap.max"] == "0" and limits["pids.max"] == str(device_resources.TASKS_MAX),
            "Native aggregate effective limits differ")
    cpu = limits["cpu.max"].split()
    require(len(cpu) == 2 and all(item.isdigit() for item in cpu)
            and int(cpu[1]) > 0 and int(cpu[0]) == 4 * int(cpu[1]),
            "Native aggregate CPU quota differs")
    counters = value["counters"]
    require(0 <= counters["memory.current"] <= counters["memory.peak"] <= device_resources.MEMORY_MAX
            and counters["memory.swap.current"] == 0 and 0 < counters["pids.current"] <= device_resources.TASKS_MAX
            and all(value["memory_events"][key] == 0 for key in ("oom", "oom_kill")),
            "Native aggregate exceeded a resource bound")
    return True


def resources_and_cleanup(evidence, host, drive, incarnations):
    cleanup, profile = host["workload_cleanup"], host["profile_cleanup"]
    expected = {value[key] for value in incarnations for key in ("unit", "checker_unit")}
    units = {item["unit"]: item for item in cleanup["application_units"]}
    require(len(units) == len(cleanup["application_units"]) == 6 and set(units) == expected
            and set(cleanup["owned_unit_names"]) == expected and cleanup["passed"] is True,
            "Cleanup omitted a cold-start application or checker")
    for value in [cleanup["driver"], cleanup["receiver"], *units.values()]:
        stopped(value)
    require(cleanup["driver"]["unit"] == DRIVER
            and cleanup["receiver"]["unit"] == device_resources.RECEIVER_UNIT
            and drive["final_owner_mode"] == "retired" and drive["cleanup_verified"] is True
            and profile["passed"] is True and profile["core_policy_restored"] is True
            and profile["namespaces"] == {"organism-device": True, "organism-bootstrap": True}
            and profile["mounts"] == {"/var/lib/organism/control": True, "/var/lib/organism/data": True},
            "Owner retirement or native namespace/storage cleanup differs")
    samples = evidence.json("resource-samples.json", 8 * 1024**2)
    require(2 <= len(samples) <= first_lineage_profile.MAXIMUM_SAMPLES, "Resource sample inventory differs")
    previous, peaks, pids = None, [], []
    for value in samples:
        require(value["aggregate_present"] is True, "Aggregate process group was not observed")
        at = common.moment(value["observed_at"])
        require(previous is None or 0 < at - previous <= 20, "Resource observation gap exceeds its allowance")
        previous = at
        peak, current = int(value["memory.peak"]), int(value["memory.current"])
        require(0 <= current <= peak <= device_resources.MEMORY_MAX
                and int(value["memory.swap.current"]) == 0
                and 0 < int(value["pids.current"]) <= device_resources.TASKS_MAX,
                "Native aggregate exceeded memory, swap or process limits")
        events = process_storage.counters(value["memory.events"])
        require(all(events.get(key) == 0 for key in ("oom", "oom_kill")), "Native aggregate had an OOM event")
        memory = dict(line.split(":", 1) for line in value["host_memory"].splitlines() if ":" in line)
        require(7 * 1024**3 <= int(memory["MemTotal"].split()[0]) * 1024 <= 8 * 1024**3
                and int(memory["SwapTotal"].split()[0]) == 0, "Host memory capacity or swap differs")
        peaks.append(peak)
        pids.append(int(value["pids.current"]))
    first, last = [common.moment(samples[index]["observed_at"]) for index in (0, -1)]
    require(first <= common.moment(drive["cold_starts"][0]["ready_observed_at"])
            and last + 20 >= common.moment(drive["temperature_tasks"]["completed_at"]),
            "Resource samples do not cover cold starts and useful work")
    return {"samples": len(samples), "maximum_aggregate_memory_bytes": max(peaks),
            "maximum_aggregate_tasks": max(pids), "whole_host_transient_peak_claimed": False}


def inspect(directory, kit_directory):
    evidence = common.Evidence(directory)
    kit, catalog, release = trusted_release(kit_directory)
    require(evidence.read("input-catalog.json") == kit.read("catalog.json"), "Actual native input catalog differs")
    host, drive = evidence.json("host-receipt.json", 8 * 1024**2), evidence.json("drive-receipt.json", 16 * 1024**2)
    require(host["schema"] == "organism-native-profile-host-v1" and host["architecture"] == "x86_64"
            and drive["schema"] == "organism-native-combined-seed-v1"
            and all(value["passed"] is True and value["qualification_passed"] is False
                    and value["release_id"] == release["release_id"] and not value.get("failure")
                    for value in (host, drive))
            and host["driver_return_code"] == 0 and drive["seed_provisioned_by_owner"] is True
            and drive["autonomous_reproduction_verified"] is False
            and host["first_lineage"] == drive["first_lineage"] == first_lineage_profile.descriptor(),
            "Native host/driver failed or changed its declared scope")
    device, policy_id, bootstrap = owner_policy(evidence, catalog)
    records = drive["cold_starts"]
    first_lineage_seed.check_incarnations(records)
    values = supervision(evidence, device=device, policy_id=policy_id, operation=drive["operation_id"],
                         release=release, records=records)
    capacity_before = capacity(host["host_capacity"])
    require(capacity(drive["host_capacity_before"]) == capacity_before, "Host profile changed before birth")
    allowed = [host["disk_plan"][role]["device"] for role in ("data", "control")]
    replays, censuses = [], []
    for ordinal, record in enumerate(records, 1):
        owner_mount_namespace.validate(
            drive["owner_mount_namespace"], record["roles"]["receiver"],
            storage=record["application_observation"]["device_storage"])
        token = record["birth"]["process_token"]
        current = values[token]
        require(record["birth"]["device_id"] == device
                and capacity(record["host_capacity_ready"]) == capacity_before,
                "Cold start moved devices or changed capacity")
        require(evidence.json("cold-start-" + str(ordinal).zfill(2) + "-receipt.json", 4 * 1024**2) == record,
                "Separate cold-start receipt differs")
        shared = {key: current["intent"]["record"][key] for key in
                  ("individual_id", "process_token", "operation_id", "boot_id")}
        replay = common.replay_readiness(
            evidence, record, current, shared, release, trained=True, namespace_path=RECEIVER_NAMESPACE,
            state_prefix=STATE, work_prefix=WORK)
        require(record["roles"]["application"] == {
            "pid": record["application_observation"]["process"]["pid"],
            "start_ticks": int(record["application_observation"]["process"]["start_identity"].split(":")[1])}
            and replay["inference_pid"] == record["roles"]["inference"]["pid"],
            "Cold-start model process differs from its owner census")
        aggregate(record["application_observation"]["aggregate_resources"])
        receiver = record["receiver_ready"]["resources"]
        aggregate(receiver)
        require(receiver["receiver"]["pid"] == record["roles"]["receiver"]["pid"]
                and receiver["receiver"]["properties"]["ControlGroup"]
                == device_resources.service_cgroup(device_resources.RECEIVER_UNIT, device_resources.SLICE),
                "Receiver process is outside its observed device boundary")
        common.absence(current["exit"]["record"]["observation"], replay["unit"])
        replays.append(replay)
        label = "" if ordinal == 1 else "cold-start-" + str(ordinal).zfill(2)
        censuses.append(census(evidence, label, record, allowed_devices=allowed,
                               release_id=release["release_id"],
                               request_id=drive["operation_id"] if ordinal == 1 else token))
    require(drive["exit_record"] == values[records[-1]["birth"]["process_token"]]["exit"],
            "Final retirement exit differs from supervision")
    peer = common.transport(evidence, drive, catalog, bootstrap, state_prefix=STATE)
    binding = {key: records[-1]["birth"][key] for key in ("device_id", "individual_id", "release_id")}
    binding["start_token"] = records[-1]["birth"]["process_token"]
    tasks = first_lineage_tasks.audit(
        Path(directory) / "temperature-tasks", cases_sha256=first_lineage_profile.CASES_SHA256, binding=binding)
    require(tasks["tasks_passed"] is True and tasks["successes"] == 20
            and evidence.json("temperature-tasks/report.json") == drive["temperature_tasks"],
            "Native twenty-case temperature gate failed")
    resources = resources_and_cleanup(evidence, host, drive, replays)
    return {
        "schema": "organism-first-lineage-native-audit-v1", "artifact_consistency_passed": True,
        "device_id": device, "release_id": release["release_id"], "cold_starts": replays,
        "censuses": censuses, "temperature_tasks": tasks, "transport": peer, "resources": resources,
        "owner_retirement_and_host_cleanup_reconstructed": True,
        "read_file_sha256": evidence.read_hashes, "auditor_sha256": sha(Path(__file__).read_bytes()),
        "native_execution_verified": False, "external_run_provenance_required": True,
        "autonomous_reproduction_verified": False, "qualification_passed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.directory, args.kit)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in
                      ("artifact_consistency_passed", "native_execution_verified", "external_run_provenance_required")}))
