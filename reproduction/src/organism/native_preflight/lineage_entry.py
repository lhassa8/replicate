"""Portable owner entry points; descendant launch is not an available operation."""

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, "/var/lib/organism/data/bootstrap/site-packages")


def main():
    from organism.native_preflight import lineage_observe, lineage_provision
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("provision", "enable", "seed", "observe", "endurance"),
                        required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--slot", choices=("seed", "child", "grandchild"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--capsule", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--ready", type=Path)
    parser.add_argument("--start-epoch", type=float)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_bytes())
    if args.operation == "provision":
        if args.bundle is None or args.capsule is not None:
            raise ValueError("Provisioning requires only the private receiving-substrate bundle")
        result = lineage_provision.provision(
            catalog, args.bundle, args.slot, guard_python=Path("/usr/bin/python3").resolve(),
            guard_script=Path(__file__).resolve().parents[1] / "start_guard.py")
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return 0
    if args.operation == "seed":
        if args.slot != "seed" or args.capsule is None:
            raise ValueError("Only the seed slot can receive an owner-supplied full package")
        result = lineage_provision.seed_once(catalog, args.capsule, output=args.output)
        return 0 if result["ready"] and result["failure"] is None else 1
    if args.capsule is not None or args.bundle is not None:
        raise ValueError("Observation receives only the fixed cases and admitted catalog")
    node = json.loads(lineage_provision.NODE.read_bytes())
    if node["slot"] != args.slot:
        raise ValueError("Observation requested a different device slot")
    if args.operation == "enable":
        from organism.receiver_service import configuration, native_state
        config, public = configuration(lineage_provision.CONFIG)
        state, _ = native_state(config, public, admit_storage=True)
        before = state.snapshot()
        if (before["owner_mode"] != "stopped" or before["birth_count"] != 0
                or before["current_birth"] is not None):
            raise ValueError("Availability requires a fresh, stopped receiving device")
        result = {"before": before, "control": state.owner_control(
            "resume", "Owner introduces this fresh enrolled lineage device"), "after": state.snapshot(),
                  "application_started": False, "descendant_launch_performed": False}
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        return 0
    if args.cases is None:
        raise ValueError("Observation requires the frozen cases")
    if args.operation == "endurance":
        from organism.native_preflight import lineage_endurance
        from organism.receiver_service import configuration
        if args.slot == "seed" or args.ready is None or args.start_epoch is None:
            raise ValueError("Endurance requires a previously observed descendant")
        ready = json.loads((args.ready / "ready-with-census.json").read_bytes())
        if ready["slot"] != args.slot or ready["binding"]["device_id"] != node["device_id"]:
            raise ValueError("Endurance readiness belongs to another device")
        config, _ = configuration(lineage_provision.CONFIG)
        work = Path(config["work_root"]) / ready["binding"]["individual_id"]
        result = lineage_endurance.run(
            work, args.output, args.cases, binding=ready["binding"],
            application_pid=ready["roles"]["application"]["pid"], uid=work.stat().st_uid,
            start_epoch=args.start_epoch)
        return 0 if result["passed"] else 1
    result = lineage_observe.observe(catalog, args.output, args.cases)
    return 0 if result["tasks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
