"""Owner-only, three-host laboratory network with one metered device edge.

The bridge has no host IP address, and each device namespace has only its
connected laboratory route. The fixed unicast tunnel peers are the three
private EC2 addresses. AWS security-group admission is checked separately.
Kernel reference: https://docs.kernel.org/networking/vxlan.html
"""

import argparse
import base64
import hashlib
import json
import os
import platform
import re
import socket
import struct
import time

from .. import network_budget
from ..network_namespace import inspect_namespace
from ..training_inputs import require


SLOTS = ("seed", "child", "grandchild")
UNDERLAY = {slot: "10.198.0." + str(10 + index) for index, slot in enumerate(SLOTS)}
ADDRESSES = {slot: "10.74.0." + str(index + 1) for index, slot in enumerate(SLOTS)}
LAB = "10.74.0.0/29"
BOOTSTRAP_ADDRESS = "10.74.0.4"
DEVICE_NAMESPACE = "organism-device"
BOOTSTRAP_NAMESPACE = "organism-bootstrap"
BRIDGE = "orglinebr"
TUNNEL = "orglinevx"
VNI = 74016
PORT = 4789
MTU = 1450


def link_request(index, sequence):
    """One read-only RTM_GETLINK request, using the stable Linux UAPI."""
    require(type(index) is int and 0 < index < 2**31
            and type(sequence) is int and 0 < sequence < 2**32,
            "Invalid kernel link request identity")
    payload = struct.pack("=BBHiII", 0, 0, 0, index, 0, 0)
    return struct.pack("=IHHII", 32, 18, 1, sequence, 0) + payload


def decode_link(observation):
    """Reconstruct required fields from the retained, bounded kernel reply.

    Ubuntu's iproute2 VXLAN detail formatter can crash on newer attributes.
    Unknown attributes remain in the raw evidence but are not interpreted.
    """
    require(observation["schema"] == "organism-lineage-link-readback-v1"
            and observation["ifname"] in (BRIDGE, TUNNEL)
            and observation["reply_sender"] == [0, 0]
            and observation["reply_flags"] == 0,
            "Kernel link reply origin or capture differs")
    sequence, index = observation["sequence"], observation["ifindex"]
    query = base64.b64decode(observation["request_base64"], validate=True)
    raw = base64.b64decode(observation["reply_base64"], validate=True)
    require(query == link_request(index, sequence)
            and 32 <= len(raw) <= 16384
            and hashlib.sha256(raw).hexdigest() == observation["reply_sha256"],
            "Kernel link request or reply identity differs")
    size, kind, flags, actual_sequence, _ = struct.unpack_from("=IHHII", raw)
    require(size == len(raw) and kind == 16 and flags == 0
            and actual_sequence == sequence, "Kernel link request failed or is incomplete")
    _, _, _, actual_index, _, _ = struct.unpack_from("=BBHiII", raw, 16)
    require(actual_index == index, "Kernel link index differs")
    attributes = network_budget._attributes(raw[32:], pads=(42,))  # IFLA_PAD
    name = attributes.get(3, b"")
    require(name == observation["ifname"].encode("ascii") + b"\0",
            "Kernel link name differs")
    info = network_budget._attributes(attributes.get(18, b""))
    expected_kind = "bridge" if observation["ifname"] == BRIDGE else "vxlan"
    require(info.get(1) == expected_kind.encode() + b"\0", "Kernel link kind differs")
    result = {"ifindex": index, "ifname": observation["ifname"],
              "mtu": network_budget._integer(attributes, 4),
              "linkinfo": {"info_kind": expected_kind}}
    if 10 in attributes:
        result["master_index"] = network_budget._integer(attributes, 10)
    if expected_kind == "vxlan":
        details = network_budget._attributes(info.get(2, b""))
        require(len(details.get(4, b"")) == 4 and len(details.get(15, b"")) == 2
                and details.get(7) in (b"\0", b"\1"), "Kernel VXLAN fields are incomplete")
        result["linkinfo"]["info_data"] = {
            "id": network_budget._integer(details, 1),
            "local": socket.inet_ntop(socket.AF_INET, details[4]),
            "port": struct.unpack("!H", details[15])[0],
            "learning": details[7] == b"\1",
        }
        require(info.get(4) == b"bridge\0", "Tunnel is not an observed bridge port")
        port = network_budget._attributes(info.get(5, b""))
        require(port.get(8) in (b"\0", b"\1") and port.get(9) in (b"\0", b"\1"),
                "Kernel bridge-port learning or flooding fields are incomplete")
        result["bridge_port"] = {
            "learning": port[8] == b"\1", "unicast_flood": port[9] == b"\1",
        }
    return result


def collect_link(name):
    require(platform.system() == "Linux" and os.geteuid() == 0
            and name in (BRIDGE, TUNNEL), "Only the Linux owner may inspect the two lab links")
    index = socket.if_nametoindex(name)
    sequence = time.monotonic_ns() % (2**32 - 1) + 1
    query = link_request(index, sequence)
    with socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, socket.NETLINK_ROUTE) as connection:
        connection.settimeout(5)
        connection.bind((0, 0))
        connection.sendto(query, (0, 0))
        raw, _, flags, sender = connection.recvmsg(16384, 0)
    observation = {
        "schema": "organism-lineage-link-readback-v1", "ifname": name, "ifindex": index,
        "sequence": sequence, "request_base64": base64.b64encode(query).decode(),
        "reply_base64": base64.b64encode(raw).decode(),
        "reply_sha256": hashlib.sha256(raw).hexdigest(),
        "reply_sender": list(sender), "reply_flags": flags,
    }
    # Preserve a rejected kernel response in the command receipt too.
    print(json.dumps(observation, sort_keys=True), flush=True)
    decode_link(observation)


def slot_value(slot):
    require(slot in SLOTS, "Unknown first-lineage device slot")
    return slot


def namespaces(slot):
    slot_value(slot)
    return (DEVICE_NAMESPACE, BOOTSTRAP_NAMESPACE) if slot == "seed" else (DEVICE_NAMESPACE,)


def address(slot, namespace):
    require(namespace in namespaces(slot), "Namespace is outside this device's owner plan")
    return BOOTSTRAP_ADDRESS if namespace == BOOTSTRAP_NAMESPACE else ADDRESSES[slot]


def link_address(slot, namespace, *, parent=False):
    """Distinct locally administered unicast addresses for this isolated lab."""
    endpoint = int(address(slot, namespace).rsplit(".", 1)[1])
    return "02:74:00:00:" + ("01" if parent else "00") + ":" + format(endpoint, "02x")


def plans(slot, underlay_interface):
    slot_value(slot)
    require(isinstance(underlay_interface, str)
            and re.fullmatch("[A-Za-z][A-Za-z0-9_.-]{0,14}", underlay_interface),
            "Invalid observed underlay interface")
    commands = [
        ["ip", "link", "add", BRIDGE, "type", "bridge", "stp_state", "0"],
        ["sysctl", "-w", "net.ipv6.conf." + BRIDGE + ".disable_ipv6=1"],
        ["ip", "link", "set", BRIDGE, "mtu", str(MTU)],
        ["ip", "link", "set", BRIDGE, "up"],
        ["ip", "link", "add", TUNNEL, "type", "vxlan", "id", str(VNI),
         "local", UNDERLAY[slot], "dev", underlay_interface, "dstport", str(PORT), "nolearning"],
        ["sysctl", "-w", "net.ipv6.conf." + TUNNEL + ".disable_ipv6=1"],
        ["ip", "link", "set", TUNNEL, "mtu", str(MTU)],
        ["ip", "link", "set", TUNNEL, "master", BRIDGE],
        ["bridge", "link", "set", "dev", TUNNEL, "learning", "off"],
        ["ip", "link", "set", TUNNEL, "up"],
    ]
    for peer in SLOTS:
        if peer != slot:
            commands.append(["bridge", "fdb", "append", "00:00:00:00:00:00",
                             "dev", TUNNEL, "dst", UNDERLAY[peer], "self", "permanent"])
    for namespace in namespaces(slot):
        parent = "orglineboot" if namespace == BOOTSTRAP_NAMESPACE else "orglinedev"
        commands += [
            ["ip", "netns", "add", namespace],
            ["ip", "netns", "exec", namespace, "sysctl", "-w", "net.ipv6.conf.all.disable_ipv6=1"],
            ["ip", "netns", "exec", namespace, "sysctl", "-w", "net.ipv6.conf.default.disable_ipv6=1"],
            ["ip", "-n", namespace, "link", "set", "lo", "up"],
            ["ip", "link", "add", parent, "address", link_address(slot, namespace, parent=True),
             "type", "veth", "peer", "name", "orglinepeer",
             "address", link_address(slot, namespace)],
            ["ip", "link", "set", parent, "mtu", str(MTU)],
            ["ip", "link", "set", parent, "master", BRIDGE],
            ["ip", "link", "set", parent, "up"],
            ["ip", "link", "set", "orglinepeer", "netns", namespace],
            ["ip", "-n", namespace, "link", "set", "orglinepeer", "name", "org0"],
            ["ip", "-n", namespace, "link", "set", "org0", "mtu", str(MTU)],
            ["ip", "-n", namespace, "addr", "add", address(slot, namespace) + "/29", "dev", "org0"],
            ["ip", "-n", namespace, "link", "set", "org0", "up"],
        ]
    commands.extend(network_budget.installation_commands(DEVICE_NAMESPACE))
    return commands


def inspect_commands(slot):
    slot_value(slot)
    result = {
        "root_links": ["ip", "-j", "link", "show"],
        "bridge_details": ["python3", "-m", "organism.native_preflight.lineage_network",
                           "--observe-link", BRIDGE],
        "tunnel_details": ["python3", "-m", "organism.native_preflight.lineage_network",
                           "--observe-link", TUNNEL],
        "bridge_addresses": ["ip", "-j", "address", "show", "dev", BRIDGE],
        "tunnel_addresses": ["ip", "-j", "address", "show", "dev", TUNNEL],
        "forwarding": ["bridge", "-j", "fdb", "show", "dev", TUNNEL],
    }
    for namespace in namespaces(slot):
        prefix = ["ip", "-j", "-n", namespace]
        result[namespace + "/links"] = prefix + ["-details", "link", "show"]
        result[namespace + "/addresses"] = prefix + ["address", "show"]
        result[namespace + "/routes"] = prefix + ["route", "show", "table", "all"]
        result[namespace + "/ipv6_routes"] = prefix + ["-6", "route", "show", "table", "all"]
    return result


def verify_forwarding(slot, rows, tunnel):
    """Check every entry in the device-filtered FDB, including local entries.

    With ``show dev`` iproute2 omits the repeated interface name. Linux also
    reports the tunnel's own permanent bridge entries, with and without VLAN 1.
    Neither local entry describes a remote tunnel destination.
    """
    require(isinstance(rows, list) and len(rows) == 4,
            "Lineage forwarding inventory differs")
    local_mac = tunnel.get("address")
    require(isinstance(local_mac, str)
            and re.fullmatch("[0-9a-f]{2}(:[0-9a-f]{2}){5}", local_mac)
            and local_mac != "00:00:00:00:00:00" and not int(local_mac[:2], 16) & 1,
            "Lineage tunnel lacks its observed unicast MAC")
    peers, local_vlans = [], []
    for row in rows:
        require(isinstance(row, dict), "Unexpected forwarding entry")
        require(all(row[name] == TUNNEL for name in ("dev", "ifname") if name in row),
                "Forwarding entry belongs to another interface")
        entry = {key: value for key, value in row.items() if key not in ("dev", "ifname")}
        if "dst" in entry:
            require(set(entry) == {"mac", "dst", "flags", "state"}
                    and entry["mac"] == "00:00:00:00:00:00"
                    and entry["flags"] == ["self"] and entry["state"] == "permanent",
                    "Lineage remote forwarding entry differs")
            peers.append(entry["dst"])
        else:
            require(set(entry) in ({"mac", "flags", "master", "state"},
                                   {"mac", "flags", "master", "state", "vlan"})
                    and entry["mac"] == local_mac and entry["flags"] == []
                    and entry["master"] == BRIDGE and entry["state"] == "permanent",
                    "Lineage local forwarding entry differs")
            vlan = entry.get("vlan", 0)
            require(type(vlan) is int and vlan in (0, 1),
                    "Unexpected local forwarding VLAN")
            local_vlans.append(vlan)
    require(len(peers) == 2 and set(peers) == {
                UNDERLAY[peer] for peer in SLOTS if peer != slot}
            and sorted(local_vlans) == [0, 1],
            "Lineage forwarding escapes its two declared peers")


def verify_observation(slot, observation):
    """Reconstruct connectivity from kernel command results, without mutation."""
    expected = inspect_commands(slot)
    require(set(observation) == set(expected), "Incomplete lineage network observations")
    decoded = {}
    for name, argv in expected.items():
        item = observation[name]
        require(item["argv"] == argv and item["return_code"] == 0
                and isinstance(item["stdout"], str) and len(item["stdout"].encode()) <= 65536,
                "Lineage network command changed or failed")
        decoded[name] = json.loads(item["stdout"])
    links = {row["ifname"]: row for row in decoded["root_links"]}
    bridge, tunnel = links[BRIDGE], links[TUNNEL]
    bridge_details = decode_link(decoded["bridge_details"])
    tunnel_details = decode_link(decoded["tunnel_details"])
    require(bridge_details["ifindex"] == bridge["ifindex"]
            and tunnel_details["ifindex"] == tunnel["ifindex"]
            and bridge_details["mtu"] == tunnel_details["mtu"] == MTU
            and tunnel_details["master_index"] == bridge["ifindex"]
            and bridge["mtu"] == tunnel["mtu"] == MTU
            and tunnel["master"] == BRIDGE
            and bridge_details["linkinfo"]["info_kind"] == "bridge"
            and tunnel_details["linkinfo"]["info_kind"] == "vxlan",
            "Lineage bridge, tunnel or MTU differs")
    details = tunnel_details["linkinfo"]["info_data"]
    require(details["id"] == VNI and details["local"] == UNDERLAY[slot]
            and details["port"] == PORT and details["learning"] is False,
            "Lineage tunnel has unexpected endpoints, port or learning")
    require(tunnel_details["bridge_port"] == {"learning": False, "unicast_flood": True},
            "Lineage bridge-port learning or unicast flooding differs")
    for name in ("bridge_addresses", "tunnel_addresses"):
        require(len(decoded[name]) == 1 and decoded[name][0]["addr_info"] == [],
                "The owner bridge or tunnel exposes a host network address")
    verify_forwarding(slot, decoded["forwarding"], tunnel)
    for namespace in namespaces(slot):
        local = decoded[namespace + "/links"]
        require(len(local) == 2 and {row["ifname"] for row in local} == {"lo", "org0"},
                "An additional device edge bypasses the traffic budget")
        edge = next(row for row in local if row["ifname"] == "org0")
        require(edge["mtu"] == MTU and "UP" in edge["flags"]
                and edge["linkinfo"]["info_kind"] == "veth", "Device edge differs")
        parent = links.get("orglineboot" if namespace == BOOTSTRAP_NAMESPACE else "orglinedev")
        require(edge.get("address") == link_address(slot, namespace)
                and isinstance(parent, dict)
                and parent.get("address") == link_address(slot, namespace, parent=True)
                and parent.get("master") == BRIDGE
                and parent.get("link_index") == edge.get("ifindex")
                and edge.get("link_index") == parent.get("ifindex")
                and type(edge.get("ifindex")) is int and type(parent.get("ifindex")) is int,
                "Laboratory veth addresses or paired bridge port differ")
        addresses = decoded[namespace + "/addresses"]
        require(len(addresses) == 2 and {row["ifname"] for row in addresses} == {"lo", "org0"},
                "Device address inventory differs")
        actual = {row["ifname"]: row["addr_info"] for row in addresses}
        require(len(actual["org0"]) == 1 and actual["org0"][0]["family"] == "inet"
                and actual["org0"][0]["local"] == address(slot, namespace)
                and actual["org0"][0]["prefixlen"] == 29
                and next(row for row in addresses if row["ifname"] == "org0").get("address")
                    == link_address(slot, namespace)
                and all(item["family"] == "inet" and item["local"] == "127.0.0.1"
                        for item in actual["lo"]), "Device addresses differ")
        routes = decoded[namespace + "/routes"]
        main = [row for row in routes if row.get("table", "main") == "main"]
        require(len(main) == 1 and main[0]["dst"] == LAB and main[0]["dev"] == "org0"
                and main[0]["scope"] == "link" and "gateway" not in main[0],
                "Device has a default or non-laboratory route")
        require(all(row.get("table", "main") in ("main", "local") for row in routes)
                and decoded[namespace + "/ipv6_routes"] == [],
                "An additional routing table or IPv6 route escapes the laboratory")
        local_routes = [row for row in routes if row.get("table") == "local"]
        require(all(row.get("type") in ("local", "broadcast")
                    and row["dev"] in ("lo", "org0") and "gateway" not in row
                    and row["dst"] in {"127.0.0.0/8", "127.0.0.1", "127.255.255.255",
                                       "10.74.0.0", "10.74.0.7", address(slot, namespace)}
                    for row in local_routes), "Unexpected local routing-table entry")
    return {"schema": "organism-lineage-network-v1", "slot": slot, "lab": LAB,
            "underlay": dict(UNDERLAY), "device_address": ADDRESSES[slot],
            "no_device_default_route": True, "host_bridge_has_no_address": True,
            "fixed_unicast_peers": True, "distinct_endpoint_link_addresses": True,
            "native_execution_verified": False}


def configure(profile, slot):
    """Called only after the independent instance/disk permit has been checked."""
    slot_value(slot)
    require(os.geteuid() == 0, "Only the admitted Linux owner may construct this lab")
    require(not profile.namespaces and all(not os.path.lexists("/run/netns/" + name)
                                          for name in namespaces(slot)), "Do not reuse owner network state")
    before = json.loads(profile.command(["ip", "-j", "address", "show"]).stdout)
    named = {row["ifname"] for row in before}
    require(not named & {BRIDGE, TUNNEL, "orglinedev", "orglineboot", "orglinepeer"},
            "Do not replace an existing host link")
    interfaces = [row["ifname"] for row in before if any(
        item.get("local") == UNDERLAY[slot] and item.get("family") == "inet"
        for item in row["addr_info"])]
    require(len(interfaces) == 1, "Actual underlay address differs from the fixed instance slot")
    evidence = {"slot": slot, "underlay_interface": interfaces[0], "root_addresses_before": before,
                "created_root_links": []}
    profile.receipt["lineage_network"] = evidence
    profile.save()
    for argv in plans(slot, interfaces[0]):
        profile.command(argv)
        if argv[:3] == ["ip", "link", "add"]:
            evidence["created_root_links"].append(argv[3])
        if argv[:3] == ["ip", "netns", "add"]:
            profile.namespaces.append(argv[3])
        profile.save()
    observations = {}
    for name, argv in inspect_commands(slot).items():
        reply = profile.command(argv)
        observations[name] = {"argv": argv, "return_code": reply.returncode, "stdout": reply.stdout}
    evidence["observations"] = observations
    evidence["verification"] = verify_observation(slot, observations)
    profile.receipt["namespaces"] = {
        name: inspect_namespace("/run/netns/" + name) for name in namespaces(slot)}
    profile.save()
    return evidence


def remove_root_links(profile):
    """After device services and namespaces have been independently stopped."""
    result = {}
    evidence = profile.receipt.get("lineage_network", {})
    for link in reversed(evidence.get("created_root_links", [])):
        require(link in (BRIDGE, TUNNEL, "orglinedev", "orglineboot"), "Unexpected owned root link")
        before = profile.command(["ip", "-j", "link", "show"], check=True)
        if link in {row["ifname"] for row in json.loads(before.stdout)}:
            profile.command(["ip", "link", "delete", link])
        reply = profile.command(["ip", "-j", "link", "show"], check=True)
        result[link] = {"absent": link not in {row["ifname"] for row in json.loads(reply.stdout)},
                        "observation": reply.stdout}
        require(result[link]["absent"], "Owned laboratory link remains")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observe-link", choices=(BRIDGE, TUNNEL), required=True)
    collect_link(parser.parse_args().observe_link)
