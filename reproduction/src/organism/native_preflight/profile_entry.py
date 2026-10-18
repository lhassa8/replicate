"""Portable owner driver; no application or model implementation is imported."""

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, "/var/lib/organism/data/bootstrap/site-packages")


def main():
    from organism.native_preflight import profile_seed
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("provision", "drive"), required=True)
    parser.add_argument("--architecture", choices=("arm64", "x86_64"), required=True)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--capsule", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--cases-sha256")
    args = parser.parse_args()
    from organism.native_preflight import first_lineage_profile
    first_lineage = first_lineage_profile.validate_selection(args.cases_sha256, args.architecture)
    if (args.cases is not None) != first_lineage:
        raise ValueError("Native observer needs both the frozen case file and its checksum")
    if first_lineage:
        first_lineage_profile.validate_cases(args.cases)
    catalog = json.loads(args.catalog.read_bytes())
    if args.operation == "provision":
        config = profile_seed.provision(
            catalog, args.capsule, args.architecture,
            guard_python=Path("/usr/bin/python3").resolve(),
            guard_script=Path(__file__).resolve().parents[1] / "start_guard.py",
            cases_sha256=args.cases_sha256)
        result = {"provisioned": True, "device_id": config["device_id"],
                  "seed_provisioned_by_owner": True, "qualification_passed": False}
    else:
        result = profile_seed.drive(catalog, args.capsule, cases=args.cases, cases_sha256=args.cases_sha256)
        result["architecture"] = args.architecture
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("provisioned", "passed", "failure", "qualification_passed")
                      if key in result}), flush=True)
    return 0 if result.get("provisioned") or result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
