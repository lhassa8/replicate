"""Owner-only setup/cleanup for a fresh, dedicated three-disk native seed host.

A controller must validate the instance-bound launch permit before constructing
this adapter. No received model, peer input or application can invoke it.
"""

import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

from .. import device_resources, device_storage, network_budget
from ..network_namespace import inspect_namespace
from . import device_capacity


NAMESPACES = ("organism-device", "organism-bootstrap")
ADDRESSES = ("10.73.0.1/30", "10.73.0.2/30")
LINKS = ("orgseed0", "orgseed1")
SLICE_GROUP = Path("/sys/fs/cgroup") / device_resources.CGROUP.lstrip("/")


def require(value, message):
    if not value:
        raise ValueError(message)


def disk_plan(volumes, blocks, serials, mounts, *, owned_mounts=()):
    """Bind the exact three new EBS disks to the externally checked permit."""
    require(isinstance(volumes, dict) and set(volumes) == {"root", "data", "control"},
            "Launch permit lacks its three disk identities")
    require(len(set(volumes.values())) == 3 and all(
        isinstance(value, str) and re.fullmatch(r"vol-[0-9a-f]{17}", value) for value in volumes.values()),
        "Launch permit disk identities are invalid or repeated")
    disks = {name: item for name, item in blocks.items() if item["kind"] == "disk"}
    require(len(disks) == 3 and set(serials) == set(disks), "Unexpected physical disks on the new host")
    require(len(set(serials.values())) == 3, "Physical EBS serials are ambiguous")
    by_device = {item["device"]: name for name, item in blocks.items()}
    roots = [mount for mount in mounts if mount["target"] == "/"]
    require(len(roots) == 1 and roots[0]["device"] in by_device, "Root disk identity is unavailable")
    root_name = by_device[roots[0]["device"]]
    if blocks[root_name]["kind"] == "partition":
        root_name = blocks[root_name]["parent"]
    result = {}
    for role, volume in volumes.items():
        names = [name for name, serial in serials.items() if serial == volume.replace("-", "")]
        require(len(names) == 1, "An approved EBS disk was not found by its kernel serial")
        name = names[0]
        require(disks[name]["bytes"] == device_capacity.ROLE_LIMITS[role],
                "EBS disk capacity differs from the approved role")
        if role == "root":
            require(name == root_name, "The approved root disk differs from the running root")
        else:
            require(name != root_name, "Formatting the running root is prohibited")
            require(not any(item.get("parent") == name for item in blocks.values()),
                    "A new data/control disk unexpectedly contains partitions")
            matches = [mount for mount in mounts if mount["device"] == disks[name]["device"]]
            target = str(device_storage.ROOT / role)
            if target in owned_mounts:
                require(len(matches) == 1 and matches[0]["target"] == target
                        and matches[0]["root"] == "/" and matches[0]["filesystem"] == "ext4",
                        "An already provisioned disk changed its mounted identity")
            else:
                require(not matches, "A new data/control disk is already mounted")
        result[role] = {"name": name, "path": "/dev/" + name, "volume_id": volume, **disks[name]}
    return result


class HostProfile:
    def __init__(self, volumes, *, receipt, save):
        self.volumes, self.receipt, self.save = volumes, receipt, save
        self.mounted, self.namespaces, self.units = [], [], []
        self.original_core = {}
        self.created_veth = False
        self.started = time.monotonic()
        self.receipt["commands"] = []
        self.receipt["owned"] = {"mounts": self.mounted, "namespaces": self.namespaces, "units": self.units}

    def command(self, argv, *, check=True, timeout=30):
        argv = list(map(str, argv))
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C",
                                     "SYSTEMD_PAGER": "", "SYSTEMD_COLORS": "0"})
        require(len(result.stdout.encode()) + len(result.stderr.encode()) <= 2 * 1024**2,
                "Owner setup command output exceeds its bound")
        self.receipt["commands"].append({"argv": argv, "return_code": result.returncode,
                                         "stdout": result.stdout, "stderr": result.stderr})
        self.save()
        if check and result.returncode:
            raise RuntimeError("Owner setup command failed: " + " ".join(argv))
        return result

    def inspect_disks(self):
        blocks = device_capacity.inventory()
        disks = sorted(name for name, item in blocks.items() if item["kind"] == "disk")
        require(len(disks) == 3, "The new seed host must have exactly three physical disks")
        reply = self.command(["lsblk", "--json", "--nodeps", "--output", "NAME,SERIAL",
                              *("/dev/" + name for name in disks)])
        rows = json.loads(reply.stdout)["blockdevices"]
        require(len(rows) == 3 and all(set(row) == {"name", "serial"}
                                      and isinstance(row["serial"], str) for row in rows)
                and {row["name"] for row in rows} == set(disks),
                "EBS serial observation is incomplete or ambiguous")
        serials = {row["name"]: row["serial"].strip() for row in rows}
        mounts = device_storage.parse_mountinfo(device_capacity.read("/proc/self/mountinfo", 4 * 1024**2))
        plan = disk_plan(self.volumes, blocks, serials, mounts, owned_mounts=self.mounted)
        for item in plan.values():
            info = Path(item["path"]).stat()
            require(stat.S_ISBLK(info.st_mode)
                    and f"{os.major(info.st_rdev)}:{os.minor(info.st_rdev)}" == item["device"],
                    "Named block device differs from the kernel disk identity")
        return plan

    def prepare_storage(self):
        require(not os.path.lexists(device_storage.ROOT), "Do not reuse an existing organism host layout")
        plan = self.inspect_disks()
        # Refuse any existing signature on either disk before changing either.
        for role in ("data", "control"):
            reply = self.command(["blkid", "-p", "-o", "export", plan[role]["path"]], check=False)
            require(reply.returncode == 2 and not reply.stdout.strip() and not reply.stderr.strip(),
                    "New data/control disk is not blank")
        self.receipt["disk_plan"] = plan
        self.save()
        device_storage.ROOT.mkdir(mode=0o755)
        device_storage.ROOT.chmod(0o755)
        for role in ("data", "control"):
            # Repeat the serial/root/mount check immediately before formatting.
            current = self.inspect_disks()
            require(current[role] == plan[role], "Disk identity changed before initialization")
            target = device_storage.ROOT / role
            target.mkdir(mode=0o755 if role == "data" else 0o700)
            self.command(["mkfs.ext4", "-F", "-m", "0", "-E",
                          "lazy_itable_init=0,lazy_journal_init=0", plan[role]["path"]], timeout=120)
            options = "nosuid,nodev" + (",noexec" if role == "control" else "")
            self.command(["mount", "-o", options, plan[role]["path"], target])
            self.mounted.append(str(target))
            self.save()
            target.chmod(0o755 if role == "data" else 0o700)
        for name in ("core_pattern", "core_uses_pid"):
            self.original_core[name] = device_capacity.read(Path("/proc/sys/kernel") / name)
        self.receipt["original_core_policy"] = self.original_core
        self.save()
        Path("/proc/sys/kernel/core_pattern").write_text("\n")
        Path("/proc/sys/kernel/core_uses_pid").write_text("0\n")
        self.receipt["host_capacity"] = device_capacity.observe()
        self.receipt["storage"] = device_storage.observe(self.host())
        self.save()

    @staticmethod
    def host():
        from ..systemd_supervision import NativeLinuxHost
        return NativeLinuxHost()

    def prepare_network(self):
        require(all(not os.path.lexists("/run/netns/" + name) for name in NAMESPACES),
                "Do not reuse existing seed namespaces")
        for name in NAMESPACES:
            self.command(["ip", "netns", "add", name])
            self.namespaces.append(name)
            self.save()
            self.command(["ip", "-n", name, "link", "set", "lo", "up"])
        self.command(["ip", "link", "add", LINKS[0], "type", "veth", "peer", "name", LINKS[1]])
        self.created_veth = True
        for name, link, address in zip(NAMESPACES, LINKS, ADDRESSES):
            self.command(["ip", "link", "set", link, "netns", name])
            self.command(["ip", "-n", name, "link", "set", link, "name", "org0"])
            self.command(["ip", "-n", name, "addr", "add", address, "dev", "org0"])
            self.command(["ip", "-n", name, "link", "set", "org0", "up"])
        for argv in network_budget.installation_commands(NAMESPACES[0]):
            self.command(argv)
        self.receipt["namespaces"] = {name: inspect_namespace("/run/netns/" + name) for name in NAMESPACES}
        for name in NAMESPACES:
            links = json.loads(self.command(["ip", "-j", "-n", name, "link", "show"]).stdout)
            routes = json.loads(self.command(["ip", "-j", "-n", name, "route", "show"]).stdout)
            require({item["ifname"] for item in links} == {"lo", "org0"}
                    and len(routes) == 1 and routes[0]["dst"] == "10.73.0.0/30"
                    and routes[0]["dev"] == "org0", "Seed namespace has unexpected connectivity")
        self.save()

    def install_unit(self, name, text):
        require(name in (device_resources.SLICE, device_resources.RECEIVER_UNIT),
                "Unexpected owner unit name")
        path = self.host().unit_root / name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        self.units.append(name)
        self.save()

    def start_receiver(self, config, *, config_path, python, entrypoint, libraries, runtime_seconds=1500):
        from .receiver_unit import render
        logs = device_storage.ROOT / "data/logs/receiver"
        # The application later shares this owner-only log parent.
        logs.parent.mkdir(mode=0o700)
        logs.mkdir(mode=0o700)
        for name in ("stdout.log", "stderr.log"):
            descriptor = os.open(logs / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
        self.install_unit(device_resources.SLICE, device_resources.render_slice())
        self.install_unit(device_resources.RECEIVER_UNIT, render(
            config, config_path=config_path, python=python, entrypoint=entrypoint,
            libraries=libraries, runtime_seconds=runtime_seconds))
        self.command(["systemctl", "daemon-reload"])
        self.command(["systemctl", "start", device_resources.SLICE])
        self.command(["systemctl", "start", device_resources.RECEIVER_UNIT])

    def stop_unit(self, name, *, shared=True):
        stopped = self.command(["systemctl", "stop", name], check=False, timeout=45)
        observed = self.command(["systemctl", "show", name,
                                "--property=Id,LoadState,ActiveState,SubState,MainPID,ControlPID"],
                               check=False)
        values = dict(line.split("=", 1) for line in observed.stdout.splitlines() if "=" in line)
        group = (SLICE_GROUP if shared else Path("/sys/fs/cgroup/system.slice")) / name
        events = device_capacity.read(group / "cgroup.events") if group.exists() else None
        absent = (values.get("Id") == name and values.get("MainPID") == "0"
                  and values.get("ControlPID") == "0" and values.get("ActiveState") in ("inactive", "failed")
                  and (events is None or "populated 0\n" in events))
        return {"unit": name, "stop_return_code": stopped.returncode, "properties": values,
                "cgroup_events": events, "absence_verified": absent}

    def remove(self):
        """After independent workload cleanup and evidence collection."""
        result = {"units": {}, "mounts": {}, "namespaces": {}, "core_policy_restored": False}
        for name in reversed(self.units):
            if name.endswith(".service"):
                status = self.stop_unit(name)
                result["units"][name] = status
                require(status["absence_verified"], "Owned service still has a process")
            else:
                self.command(["systemctl", "stop", name])
            (self.host().unit_root / name).unlink()
        self.command(["systemctl", "daemon-reload"])
        for name in reversed(self.namespaces):
            pids = self.command(["ip", "netns", "pids", name])
            require(not pids.stdout.strip(), "Owned namespace still has a process")
            self.command(["ip", "netns", "delete", name])
            result["namespaces"][name] = not os.path.lexists("/run/netns/" + name)
        # If moving a veth failed, its surviving endpoint may still be on the
        # host. Only names created successfully by this invocation are eligible.
        if self.created_veth:
            for link in LINKS:
                probe = self.command(["ip", "link", "show", link], check=False)
                if probe.returncode == 0:
                    self.command(["ip", "link", "delete", link])
        for name, value in self.original_core.items():
            (Path("/proc/sys/kernel") / name).write_text(value)
        result["core_policy_restored"] = all(
            device_capacity.read(Path("/proc/sys/kernel") / name) == value
            for name, value in self.original_core.items())
        for target in reversed(self.mounted):
            self.command(["umount", target])
            mounts = device_storage.parse_mountinfo(device_capacity.read("/proc/self/mountinfo", 4 * 1024**2))
            result["mounts"][target] = not any(mount["target"] == target for mount in mounts)
        result["passed"] = (all(result["namespaces"].values()) and all(result["mounts"].values())
                            and result["core_policy_restored"])
        return result
