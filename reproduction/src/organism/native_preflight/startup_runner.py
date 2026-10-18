"""Portable-Python entry point for the privileged generic receiving substrate."""

import argparse
import json
from pathlib import Path
import sys


KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT))
sys.path.insert(0, "/opt/organism-bootstrap/site-packages")


def main():
    from organism.native_preflight.startup_seed import run_native, retire_native
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("x86_64", "arm64"), required=True)
    parser.add_argument("--operation", choices=("run", "retire"), default="run")
    args = parser.parse_args()
    if args.operation == "retire":
        print(json.dumps(retire_native()), flush=True)
        return 0
    result = run_native(args.architecture)
    print(json.dumps({"passed": result["passed"], "failure": result.get("failure"),
                      "qualification_passed": False}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
