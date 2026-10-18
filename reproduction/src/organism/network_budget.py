"""Read-only kernel inspection and owner commands for a shared traffic budget.

The same Linux police action is bound to both directions of one device edge.
This file is also a standalone stdlib observer for execution inside that
owner-created network namespace. It never installs or changes network state.
"""

import argparse
import base64
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import socket
import struct
import subprocess
import time


RATE = 8 * 1024**2
BURST = 16 * 1024**2
OVERHEAD = 38
MTU = 65535
ACTION_INDEX = 5198407
COOKIE = b"organism-net-v1\0"
INTERFACE = "org0"
MAX_REPLY = 65536
PROFILE = "shared-v1"
ATTACHMENT_COMMANDS = {
    "links": ["ip", "-j", "-details", "link", "show"],
    "ingress": ["tc", "-j", "filter", "show", "dev", INTERFACE, "ingress"],
    "egress": ["tc", "-j", "filter", "show", "dev", INTERFACE, "egress"],
    "offloads": ["ethtool", "-k", INTERFACE],
}


class NetworkBudgetError(ValueError):
    def __init__(self, message, *, observation=None):
        super().__init__(message)
        self.observation = observation


def validate_profile(value):
    if value is not None and value != PROFILE:
        raise NetworkBudgetError("Only the fixed shared network budget is supported")
    return value


def validate_attachments(readings):
    """Check the only external edge and both of its shared-action attachments."""
    if not isinstance(readings, dict) or set(readings) != set(ATTACHMENT_COMMANDS):
        raise NetworkBudgetError("Network attachment observations are incomplete")
    for name, command in ATTACHMENT_COMMANDS.items():
        value = readings[name]
        if (not isinstance(value, dict) or value.get("argv") != command or value.get("return_code") != 0
                or not isinstance(value.get("stdout"), str) or len(value["stdout"].encode()) > 65536):
            raise NetworkBudgetError("Network attachment command failed or differs")
    try:
        links = json.loads(readings["links"]["stdout"])
        if (not isinstance(links, list) or len(links) != 2
                or {link["ifname"] for link in links} != {"lo", INTERFACE}):
            raise NetworkBudgetError("Unexpected network edge can bypass the shared budget")
        edge = next(link for link in links if link["ifname"] == INTERFACE)
        loopback = next(link for link in links if link["ifname"] == "lo")
        if ("LOOPBACK" not in loopback["flags"] or "UP" not in edge["flags"]
                or edge["linkinfo"]["info_kind"] != "veth"
                or type(edge["ifindex"]) is not int or edge["ifindex"] <= 0
                or type(edge["mtu"]) is not int or not 576 <= edge["mtu"] <= MTU):
            raise NetworkBudgetError("Owner network interface differs from the fixed edge")
        for direction in ("ingress", "egress"):
            rows = json.loads(readings[direction]["stdout"])
            if (not isinstance(rows, list) or len(rows) != 2 or not all(
                    row["protocol"] == "all" and row["pref"] == 10
                    and row["kind"] == "matchall" and row["chain"] == 0 for row in rows)):
                raise NetworkBudgetError("Network filters differ from the fixed edge")
            rules = [row["options"] for row in rows if "options" in row]
            if len(rules) != 1:
                raise NetworkBudgetError("Expected exactly one network rule per direction")
            rule = rules[0]
            if (rule["handle"] != 1 or rule["skip_hw"] is not True or rule["not_in_hw"] is not True
                    or len(rule["actions"]) != 1):
                raise NetworkBudgetError("Network rule is not one software-only shared action")
            action = rule["actions"][0]
            if (action["kind"] != "police" or action["index"] != ACTION_INDEX
                    or action["cookie"] != COOKIE.hex() or action["bind"] != 2
                    or action["control_action"] != {"type": "drop"} or action["overhead"] != OVERHEAD):
                raise NetworkBudgetError("Network rule does not bind the fixed shared action")
        features = {}
        for line in readings["offloads"]["stdout"].splitlines():
            if ": " in line:
                name, value = line.strip().split(": ", 1)
                if name in features:
                    raise NetworkBudgetError("Duplicate interface feature observation")
                features[name] = value
        for name in ("tcp-segmentation-offload", "generic-segmentation-offload",
                     "generic-receive-offload", "large-receive-offload", "hw-tc-offload"):
            if features[name].split()[0] != "off":
                raise NetworkBudgetError("Interface offload can bypass the software traffic budget")
        return {"interface": INTERFACE, "ifindex": edge["ifindex"], "mtu": edge["mtu"],
                "directions": ["ingress", "egress"], "action_index": ACTION_INDEX,
                "only_loopback_and_budget_edge": True}
    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
        raise NetworkBudgetError("Malformed network attachment observation") from exc


def installation_commands(namespace):
    """For a new empty owner namespace only; callers own admission and cleanup."""
    if not isinstance(namespace, str) or not re.fullmatch(r"organism-[a-z0-9-]{1,48}", namespace):
        raise NetworkBudgetError("Expected an explicit owner organism namespace")
    prefix = ["ip", "netns", "exec", namespace]
    commands = [
        prefix + ["ethtool", "-K", INTERFACE, "tso", "off", "gso", "off", "gro", "off", "lro", "off"],
        prefix + ["tc", "actions", "add", "action", "police", "rate", str(RATE * 8) + "bit",
                  "burst", str(BURST) + "b", "mtu", str(MTU) + "b",
                  "overhead", str(OVERHEAD), "conform-exceed", "drop/ok",
                  "index", str(ACTION_INDEX), "cookie", COOKIE.hex(), "skip_hw"],
        prefix + ["tc", "qdisc", "add", "dev", INTERFACE, "clsact"],
    ]
    for direction in ("ingress", "egress"):
        commands.append(prefix + ["tc", "filter", "add", "dev", INTERFACE, direction,
                                  "protocol", "all", "pref", "10", "handle", "1",
                                  "matchall", "skip_hw", "action", "police", "index", str(ACTION_INDEX)])
    return commands


def _attribute(kind, value):
    size = len(value) + 4
    return struct.pack("=HH", size, kind) + value + b"\0" * (-size % 4)


def _attributes(raw, *, pads=()):
    values, offset = {}, 0
    while offset < len(raw):
        if len(raw) - offset < 4:
            raise NetworkBudgetError("Truncated kernel attribute")
        size, tagged = struct.unpack_from("=HH", raw, offset)
        kind = tagged & 0x3FFF
        aligned = (size + 3) & ~3
        if size < 4 or offset + aligned > len(raw):
            raise NetworkBudgetError("Invalid kernel attribute length")
        if kind not in pads:
            if kind in values:
                raise NetworkBudgetError("Duplicate kernel attribute")
            values[kind] = raw[offset + 4:offset + size]
        offset += aligned
    return values


def request(sequence):
    """RTM_GETACTION for exactly one police action, without mutation flags."""
    if type(sequence) is not int or not 0 < sequence < 2**32:
        raise NetworkBudgetError("Invalid kernel request sequence")
    action = _attribute(1, b"police\0") + _attribute(3, struct.pack("=I", ACTION_INDEX))
    payload = b"\0" * 4 + _attribute(1, _attribute(1, action))
    return struct.pack("=IHHII", 16 + len(payload), 50, 1, sequence, 0) + payload


def _integer(raw, kind):
    value = raw.get(kind)
    if value is None or len(value) != 4:
        raise NetworkBudgetError("Required kernel integer is missing")
    return struct.unpack("=I", value)[0]


def scheduler_ticks_per_second(text):
    values = text.split()
    if len(values) != 4 or any(not re.fullmatch("[0-9a-fA-F]{8}", value) for value in values):
        raise NetworkBudgetError("Kernel scheduler clock is unavailable")
    t2us, us2t, resolution, _ = map(lambda value: int(value, 16), values)
    if min(t2us, us2t, resolution) <= 0:
        raise NetworkBudgetError("Invalid kernel scheduler clock")
    if resolution == 1_000_000_000:
        t2us = us2t
    return Fraction(t2us * resolution, us2t)


def decode_reply(raw, sequence, psched):
    """Decode Linux UAPI values; iproute2 6.1 JSON omits rate/burst/MTU."""
    if not isinstance(raw, bytes) or not 20 <= len(raw) <= MAX_REPLY:
        raise NetworkBudgetError("Kernel reply exceeds bounds")
    size, kind, flags, observed_sequence, _ = struct.unpack_from("=IHHII", raw)
    if size != len(raw) or observed_sequence != sequence:
        raise NetworkBudgetError("Kernel reply identity or size differs")
    if kind == 2:
        error = struct.unpack_from("=i", raw, 16)[0]
        raise NetworkBudgetError("Kernel action request failed: " + str(error))
    if kind not in (48, 50) or flags & 2:
        raise NetworkBudgetError("Unexpected or multipart kernel action reply")
    root = _attributes(raw[20:])
    table = _attributes(root.get(1, b""))
    if set(table) != {1}:
        raise NetworkBudgetError("Expected exactly one kernel action")
    action = _attributes(table[1], pads=(5,))
    if action.get(1) != b"police\0":
        raise NetworkBudgetError("Kernel action kind differs")
    options = _attributes(action.get(2, b""), pads=(7,))
    tbf = options.get(1, b"")
    if len(tbf) != 56:
        raise NetworkBudgetError("Unsupported kernel police structure")
    index, exceed, limit, ticks, mtu = struct.unpack_from("=IiIII", tbf)
    cell_log, linklayer, overhead, cell_align, mpu, rate = struct.unpack_from("=BBHhHI", tbf, 20)
    peak = struct.unpack_from("=BBHhHI", tbf, 32)
    references, bindings, capability = struct.unpack_from("=iiI", tbf, 44)
    if 8 in options:
        if len(options[8]) != 8:
            raise NetworkBudgetError("Malformed extended kernel rate")
        rate = struct.unpack("=Q", options[8])[0]
    burst = Fraction(ticks * rate, 1) / scheduler_ticks_per_second(psched)
    conform = _integer(options, 5) if 5 in options else 0
    action_flags = action.get(7, b"\0" * 8)
    if len(action_flags) != 8:
        raise NetworkBudgetError("Malformed kernel action flags")
    flag_value, flag_selector = struct.unpack("=II", action_flags)
    if (index != ACTION_INDEX or exceed != 2 or conform != 0 or rate != RATE or burst != BURST
            or mtu != MTU or overhead != OVERHEAD or linklayer & 0x0F != 1
            or mpu or peak[-1] or limit or capability
            or set(options) - {1, 5, 6, 8}
            or action.get(6) != COOKIE or flag_value & 4 or not flag_value & 2
            or bindings != 2 or references < bindings):
        raise NetworkBudgetError("Kernel shared police action differs from the fixed profile")
    stats = _attributes(action.get(4, b""), pads=(6,))
    basic, queue = stats.get(1, b""), stats.get(3, b"")
    if len(basic) not in (12, 16) or len(queue) != 20:
        raise NetworkBudgetError("Kernel action counters are incomplete")
    byte_count, packets = struct.unpack_from("=QI", basic)
    if 8 in stats:
        if len(stats[8]) != 8:
            raise NetworkBudgetError("Malformed extended packet counter")
        packets = struct.unpack("=Q", stats[8])[0]
    qlen, backlog, drops, requeues, overlimits = struct.unpack("=IIIII", queue)
    return {
        "index": index, "rate_bytes_per_second": rate, "burst_bytes": int(burst),
        "burst_scheduler_ticks": ticks, "mtu": mtu, "overhead_bytes": overhead,
        "linklayer": linklayer, "cell_log": cell_log, "cell_align": cell_align,
        "references": references, "bindings": bindings, "cookie": COOKIE.hex(),
        "flags": {"value": flag_value, "selector": flag_selector},
        "counters": {"attempted_bytes": byte_count, "attempted_packets": packets,
                     "drops": drops, "overlimits": overlimits, "qlen": qlen,
                     "backlog": backlog, "requeues": requeues},
        "counter_scope": "Attempted traffic, including drops; not admitted throughput",
        "qualification_passed": False,
    }


def collect(*, include_attachments=False):
    observation = {
        "schema": "organism-native-network-budget-v1", "native_kernel_observation": False,
        "profile_verified": False, "filter_and_interface_membership_verified": False,
        "started_monotonic_seconds": time.monotonic(),
        "qualification_passed": False,
    }
    try:
        if platform.system() != "Linux" or os.geteuid() != 0:
            raise NetworkBudgetError("Kernel action collection requires the owner Linux observer")
        sequence = time.monotonic_ns() % (2**32 - 1) + 1
        query = request(sequence)
        observation.update(sequence=sequence, request_base64=base64.b64encode(query).decode())
        with socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, socket.NETLINK_ROUTE) as connection:
            connection.settimeout(5)
            connection.bind((0, 0))
            connection.sendto(query, (0, 0))
            raw, _, flags, address = connection.recvmsg(MAX_REPLY, 0)
        # Retain the bounded capture before validating either its origin or its
        # contents. A rejected observation must remain available for diagnosis.
        observation.update(reply_base64=base64.b64encode(raw).decode(),
                           reply_sha256=hashlib.sha256(raw).hexdigest(),
                           reply_sender=list(address), reply_flags=flags)
        if address[0] != 0 or flags & socket.MSG_TRUNC:
            raise NetworkBudgetError("Kernel reply sender or capture differs")
        observation["native_kernel_observation"] = True
        psched = Path("/proc/net/psched").read_text()
        observation["psched"] = psched
        namespace = Path("/proc/self/ns/net").stat()
        observation.update(boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                           kernel=platform.release(), architecture=platform.machine(),
                           namespace={"device": namespace.st_dev, "inode": namespace.st_ino})
        observation["action"] = decode_reply(raw, sequence, psched)
        observation["profile_verified"] = True
        if include_attachments:
            readings = observation["attachment_readings"] = {}
            deadline = time.monotonic() + 5
            for name, argv in ATTACHMENT_COMMANDS.items():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise NetworkBudgetError("Network attachment observation exceeded its deadline")
                result = subprocess.run(argv, capture_output=True, text=True, timeout=min(2, remaining),
                                        env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"})
                readings[name] = {"argv": argv, "return_code": result.returncode,
                                  "stdout": result.stdout[:65537], "stderr": result.stderr[:4096]}
                if len(result.stdout.encode()) > 65536:
                    raise NetworkBudgetError("Network attachment output exceeds its bound")
            observation["attachments"] = validate_attachments(readings)
            observation["filter_and_interface_membership_verified"] = True
    except (NetworkBudgetError, OSError, subprocess.TimeoutExpired) as exc:
        observation["profile_verified"] = False
        observation["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise NetworkBudgetError(str(exc), observation=observation) from exc
    finally:
        observation["completed_monotonic_seconds"] = time.monotonic()
    return observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", required=True)
    parser.add_argument("--include-attachments", action="store_true")
    args = parser.parse_args()
    try:
        observation = collect(include_attachments=args.include_attachments)
    except NetworkBudgetError as exc:
        print(json.dumps(exc.observation, sort_keys=True))
        return 1
    print(json.dumps(observation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
