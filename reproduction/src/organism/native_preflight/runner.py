"""Entry point inside a credential-free, network-isolated systemd service."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import resource
import sys

# Invoked with the portable interpreter's -I option. Only the pinned kit and
# dependency directory are explicitly admitted to its import path.
KIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIT))
sys.path.insert(0, "/opt/organism-native/site-packages")

from organism.native_preflight.common import write_json


def isolation_receipt(expected_host_ns, proc=Path("/proc"), interfaces=Path("/sys/class/net")):
    own_ns = os.readlink(proc / "self/ns/net")
    init_ns = expected_host_ns
    names = sorted(p.name for p in interfaces.iterdir())
    if own_ns == init_ns or set(names) != {"lo"}:
        raise RuntimeError("Private network namespace was not established")
    if os.getuid() == 0:
        raise RuntimeError("Model service must run without root")
    return {"own_namespace": own_ns, "host_namespace": init_ns, "interfaces": names,
            "uid": os.getuid(), "private_network_verified": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("x86_64", "arm64"), required=True)
    args = parser.parse_args()
    output = Path("/var/lib/organism-native/output")
    receipt = {
        "started_at": datetime.now(timezone.utc).isoformat(), "requested_architecture": args.architecture,
        "machine": platform.machine(), "system": platform.system(), "python": sys.version,
        "qualification_passed": False, "native_execution_passed": False,
        "scope": "Native 8-GiB EC2 stock-model compatibility; public development fixtures",
    }
    try:
        if platform.system() != "Linux" or platform.machine() != {"arm64": "aarch64", "x86_64": "x86_64"}[args.architecture]:
            raise RuntimeError("Native machine differs from the requested architecture")
        import json
        preparation = json.loads(Path("/opt/organism-native/preparation.json").read_text())
        receipt["isolation"] = isolation_receipt(preparation["host_network_namespace"])
        receipt["os_release"] = Path("/etc/os-release").read_text()
        receipt["meminfo"] = Path("/proc/meminfo").read_text()
        receipt["swaps"] = Path("/proc/swaps").read_text()
        if len(receipt["swaps"].splitlines()) != 1:
            raise RuntimeError("Swap would invalidate the declared memory check")
        memory_kib = int(receipt["meminfo"].split("MemTotal:")[1].split()[0])
        if not 7 * 1024**2 <= memory_kib <= 8 * 1024**2:
            raise RuntimeError("Observed memory is outside the native 8-GiB profile")
        receipt["cpuinfo"] = Path("/proc/cpuinfo").read_text()
        from organism.probe_local import run
        from organism.probe_setup import probe_cases, SELECTION
        from organism.audit_setup import audit
        result = run(
            Path("/opt/organism-native"), output / "probe", case_runner=probe_cases,
            case_count=len(SELECTION), scope=receipt["scope"],
        )
        receipt["stock_successes"] = result["passed_development_cases"]
        receipt["stock_cases"] = result["total_development_cases"]
        receipt["owned_server_stopped"] = result.get("owned_server_stopped", False)
        if result.get("failure"):
            raise RuntimeError(result["failure"])
        receipt["trace_reconstruction"] = audit(output / "probe")["passed"]
        # Wrong stock answers remain wrong. Compatibility only establishes
        # completion of real bounded calls with reconstructable observations.
        receipt["native_execution_passed"] = receipt["owned_server_stopped"] and receipt["trace_reconstruction"]
    except Exception as exc:
        receipt["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        receipt["agent_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                group = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
                for name in ("memory.peak", "memory.max", "memory.swap.max", "memory.events"):
                    path = group / name
                    if path.is_file():
                        receipt[name] = path.read_text().strip()
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "native-receipt.json", receipt)
    return 0 if receipt["native_execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
