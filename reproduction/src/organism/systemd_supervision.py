"""Bounded systemd process ownership for the native receiving substrate.

The production host adapter requires root on native Linux with cgroup v2.
Unit tests provide an explicit fixture host; those observations never establish
native execution, effective isolation, readiness, or qualification.
"""

from dataclasses import dataclass
import hashlib
import os
import grp
import pwd
from pathlib import Path
import platform
import re
import stat
import subprocess
import tempfile

from .peer_identity import LabNetwork
from . import device_resources, device_storage, runtime_storage
from . import network_budget as network_limits, network_enforcement
from .network_namespace import inspect_namespace, namespace_path, verify_process_namespace
from .receiver_state import boot_identity
from .runtime_environment import library_environment
from .supervision_records import SupervisionRecords


MEMORY_MAX = 6 * 1024**3
TASKS_MAX = 128
UNIT_PREFIX = "organism-"
SHOW_FIELDS = (
    "Id", "Description", "LoadState", "ActiveState", "SubState", "MainPID", "InvocationID",
    "ControlGroup", "Job", "Result", "FragmentPath", "DropInPaths", "NeedDaemonReload",
    "User", "Group", "Restart", "KillMode", "MemoryMax", "MemorySwapMax", "TasksMax",
    "CPUQuotaPerSecUSec", "NoNewPrivileges", "ProtectSystem", "Delegate", "NetworkNamespacePath", "Slice",
)


class SupervisionError(ValueError):
    pass


class ObservationUncertain(SupervisionError):
    pass


class SubmissionUncertain(SupervisionError):
    pass


def identifier(value, length):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value):
        raise SupervisionError("Malformed supervised identity")
    return value


def unit_name(birth):
    return (UNIT_PREFIX + identifier(birth["device_id"], 64)[:16] + "-"
            + identifier(birth["individual_id"], 32) + "-" + identifier(birth["process_token"], 32) + ".service")


def description(birth):
    return " ".join("organism-" + key + "=" + identifier(birth[key], 64 if key == "device_id" else 32)
                    for key in ("device_id", "individual_id", "operation_id", "process_token"))


def quoted(value, *, specifiers=True):
    """systemd unit quoting; no shell, control characters or env expansion."""
    value = str(value)
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise SupervisionError("Invalid unit argument")
    if specifiers:
        value = value.replace("%", "%%")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def safe_path(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or ":" in str(path):
        raise SupervisionError("An absolute non-ambiguous owner path is required")
    quoted(path)
    return path


def literal_path(path):
    """Encode a scalar path directive, whose parser does not remove quotes."""
    value = str(safe_path(path))
    # Unit-file parsing trims the value and treats a final backslash as a
    # continuation. Reject paths that cannot survive that outer parser.
    if value[-1].isspace() or value.endswith("\\"):
        raise SupervisionError("Scalar unit path would be trimmed or continued")
    return value.replace("%", "%%")


@dataclass(frozen=True)
class ProcessSpec:
    """Supplied only by the trusted verified-package preparer, never a peer."""
    command: tuple[str, ...]
    working_directory: Path
    readonly_paths: tuple[Path, ...]
    user: str = "organism"
    group: str = "organism"
    lab_cidrs: tuple[str, ...] = ("127.0.0.0/24",)
    runtime_seconds: int = 96 * 3600
    library_paths: tuple[Path, ...] = ()
    readiness: dict | None = None

    def validate(self):
        if not isinstance(self.command, tuple) or not 1 <= len(self.command) <= 32:
            raise SupervisionError("Prepared command exceeds bounds")
        safe_path(self.command[0])
        for argument in self.command:
            if not isinstance(argument, str) or len(argument) > 1024:
                raise SupervisionError("Prepared argument exceeds bounds")
            quoted(argument)
        literal_path(self.working_directory)
        if not 1 <= len(self.readonly_paths) <= 8:
            raise SupervisionError("Prepared input paths exceed bounds")
        for path in self.readonly_paths:
            safe_path(path)
        if len(self.library_paths) > 4:
            raise SupervisionError("Prepared library search paths exceed bounds")
        for path in self.library_paths:
            safe_path(path)
            if not any(Path(path).is_relative_to(root) for root in self.readonly_paths):
                raise SupervisionError("Native library paths must be in the verified read-only installation")
        for account in (self.user, self.group):
            if account == "root" or not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", account):
                raise SupervisionError("An unprivileged service account is required")
        LabNetwork(self.lab_cidrs)
        if type(self.runtime_seconds) is not int or not 60 <= self.runtime_seconds <= 96 * 3600:
            raise SupervisionError("Prepared runtime allowance exceeds bounds")
        if self.readiness is not None:
            expected = {"checker_path", "checker_sha256", "model_path", "model_sha256", "model_bytes",
                        "runtime_path", "runtime_sha256"}
            if set(self.readiness) != expected:
                raise SupervisionError("Prepared readiness declaration differs")
            for name in ("checker", "model", "runtime"):
                path = safe_path(self.readiness[name + "_path"])
                if not path.is_relative_to(self.readonly_paths[0]):
                    raise SupervisionError("Readiness input must be inside the verified installation")
                identifier(self.readiness[name + "_sha256"], 64)
            if (type(self.readiness["model_bytes"]) is not int
                    or not 0 < self.readiness["model_bytes"] <= 8 * 1024**3):
                raise SupervisionError("Readiness model byte bound differs")


def render_unit(birth, spec, *, state_directory, policy_id, guard_python, guard_script, network_namespace=None,
                resource_slice=None, storage=None):
    spec.validate()
    device_resources.validate_slice(resource_slice)
    identifier(policy_id, 64)
    unit_name(birth)
    guard = (
        str(safe_path(guard_python)), "-I", str(safe_path(guard_script)),
        "--state", str(safe_path(state_directory)), "--policy-id", policy_id,
        "--device-id", birth["device_id"], "--operation-id", birth["operation_id"],
        "--process-token", birth["process_token"],
    )
    # ':' disables systemd's environment expansion. '+' gives only the
    # owner-installed condition access to the private receiver database.
    lines = [
        "[Unit]", "Description=" + description(birth), "After=local-fs.target",
        "[Service]", "Type=exec", "Restart=no", "Slice=" + (resource_slice or "system.slice"),
        "ExecCondition=+:" + " ".join(quoted(arg) for arg in guard),
        "ExecStart=:" + " ".join(quoted(arg) for arg in
                               ("/usr/bin/env", "-i", "PATH=/usr/bin:/bin",
                                *(key + "=" + value for key, value in runtime_storage.environment(storage).items()),
                                "LC_ALL=C.UTF-8", "AWS_EC2_METADATA_DISABLED=true", "OTEL_SDK_DISABLED=true",
                                "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "PYTHONDONTWRITEBYTECODE=1",
                                "ORGANISM_START_TOKEN=" + birth["process_token"],
                                "ORGANISM_DEVICE_ID=" + birth["device_id"],
                                "ORGANISM_INDIVIDUAL_ID=" + birth["individual_id"],
                                "ORGANISM_SERVICE_CGROUP=" + device_resources.service_cgroup(
                                    unit_name(birth), resource_slice),
                                *(key + "=" + value for key, value in library_environment(spec.library_paths).items()),
                                *spec.command)),
        "User=" + spec.user, "Group=" + spec.group,
        "WorkingDirectory=" + literal_path(spec.working_directory),
        "UMask=0077", "NoNewPrivileges=yes", "CapabilityBoundingSet=",
        "PrivateTmp=yes", "PrivateDevices=yes", "ProtectSystem=strict", "ProtectHome=yes",
        "ProtectControlGroups=yes", "ProtectKernelTunables=yes", "ProtectKernelModules=yes",
        "RestrictSUIDSGID=yes", "RestrictRealtime=yes", "LockPersonality=yes", "Delegate=no",
        "RestrictAddressFamilies=AF_UNIX AF_INET",
        "ReadWritePaths=" + quoted(spec.working_directory),
        "ReadOnlyPaths=" + " ".join(quoted(path) for path in spec.readonly_paths),
        "IPAddressDeny=any", "IPAddressAllow=127.0.0.0/8 " + " ".join(spec.lab_cidrs),
        "MemoryAccounting=yes", "MemoryHigh=" + str(5 * 1024**3),
        "MemoryMax=" + str(MEMORY_MAX), "MemorySwapMax=0", "CPUQuota=400%",
        "TasksMax=" + str(TASKS_MAX), "KillMode=control-group", "SendSIGKILL=yes",
        "TimeoutStartSec=30", "TimeoutStopSec=10", "RuntimeMaxSec=" + str(spec.runtime_seconds),
        "StandardOutput=journal", "StandardError=journal", "",
    ]
    if network_namespace is not None:
        lines.insert(-1, "NetworkNamespacePath=" + literal_path(namespace_path(network_namespace)))
        lines.insert(-1, "RestrictNamespaces=yes")
    if storage is not None:
        lines = [line for line in lines if not line.startswith(("PrivateTmp=", "StandardOutput=", "StandardError="))]
        lines[-1:-1] = [
            *runtime_storage.directives(storage, quoted), "LimitFSIZE=" + str(runtime_storage.FILE_MAX),
            "StandardOutput=append:" + literal_path(storage["outputs"]["stdout"]["path"]),
            "StandardError=append:" + literal_path(storage["outputs"]["stderr"]["path"]),
        ]
    return "\n".join(lines)


class NativeLinuxHost:
    is_fixture = False

    def __init__(self):
        if platform.system() != "Linux" or os.geteuid() != 0:
            raise SupervisionError("Native service supervision requires the root receiving service on Linux")
        self.proc_root = Path("/proc")
        self.sys_root = Path("/sys")
        self.dev_root = Path("/dev")
        self.cgroup_root = Path("/sys/fs/cgroup")
        self.unit_root = Path("/run/systemd/system")
        self.owner_uid = 0
        if not (self.cgroup_root / "cgroup.controllers").is_file() or not Path("/run/systemd/system").is_dir():
            raise SupervisionError("Native systemd and cgroup v2 are required")
        self.boot_id = boot_identity()
        self.machine = platform.machine()

    def run(self, arguments, *, timeout=15):
        return subprocess.run(arguments, capture_output=True, text=True, timeout=timeout, check=False,
                              env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "SYSTEMD_COLORS": "0", "SYSTEMD_PAGER": ""})


def process_start_ticks(contents, expected_pid):
    end = contents.rfind(")")
    begin = contents.find("(")
    if begin < 1 or end <= begin or int(contents[:begin].strip()) != expected_pid:
        raise ObservationUncertain("Invalid process identity record")
    fields = contents[end + 1:].split()
    if len(fields) < 20 or not fields[19].isdigit():
        raise ObservationUncertain("Process start time is unavailable")
    return fields[19]


def parse_show(text):
    if len(text.encode()) > 65536:
        raise ObservationUncertain("Systemd observation exceeds bounds")
    result = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in result:
            raise ObservationUncertain("Malformed systemd observation")
        result[key] = value
    if any(key not in result for key in SHOW_FIELDS):
        raise ObservationUncertain("Systemd observation is incomplete")
    return result


class SystemdBackend:
    def __init__(self, state, *, guard_python, guard_script, host=None, network_namespace=None, resource_slice=None,
                 network_budget=None, storage_profile=None, storage_layout=None):
        self.state = state
        self.host = host or NativeLinuxHost()
        self.guard_python = safe_path(guard_python)
        self.guard_script = safe_path(guard_script)
        # Do not open the namespace here: owner stop must remain possible
        # after its mount disappears. Start and running observations verify it.
        self.network_namespace = namespace_path(network_namespace) if network_namespace is not None else None
        # Only start/running paths inspect the boundary. Owner stop must work
        # even when that boundary has been removed or changed.
        self.resource_slice = device_resources.validate_slice(resource_slice)
        self.network_budget = network_limits.validate_profile(network_budget)
        self.storage_profile = device_storage.validate_profile(storage_profile)
        self.storage_layout = storage_layout
        if self.host.boot_id != state.boot_id:
            raise SupervisionError("Supervisor host boot differs from receiver state")
        expected = {"linux-x86_64": "x86_64", "linux-arm64": "aarch64"}[state.platform]
        if self.host.machine != expected:
            raise SupervisionError("Supervisor host architecture differs from enrollment")
        self.records = SupervisionRecords(state)

    def _file(self, birth):
        return self.host.unit_root / unit_name(birth)

    def _owned_file(self, birth):
        path = self._file(birth)
        entry = path.lstat()
        if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != self.host.owner_uid
                or entry.st_nlink != 1 or stat.S_IMODE(entry.st_mode) & 0o022):
            raise SupervisionError("Unit file is not controlled by the receiving service")
        text = path.read_text()
        binding = self.records.get(birth, "intent")
        if binding is None or binding["record"]["observation"]["unit_text"] != text:
            raise SupervisionError("Unit file differs from its persisted start intent")
        return path, text

    def preflight(self, spec):
        """Validate owner-prepared paths before consuming a process-start claim."""
        spec.validate()
        binding = self._namespace_binding()
        self.resource_observation(self.resource_slice)
        self.network_observation(self.network_budget, binding)
        self.storage_observation(self.storage_profile)
        for path in (self.guard_python, self.guard_script, Path(spec.command[0]), spec.working_directory,
                     *spec.readonly_paths, *spec.library_paths):
            if not Path(path).exists():
                raise SupervisionError("Required prepared runtime path is missing")
        if spec.working_directory.is_symlink() or not spec.working_directory.is_dir():
            raise SupervisionError("Prepared working directory must be a real directory")
        for path in (self.guard_python, self.guard_script, Path(spec.command[0])):
            entry = path.lstat()
            if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != self.host.owner_uid
                    or stat.S_IMODE(entry.st_mode) & 0o022):
                raise SupervisionError("Executable and guard must be owner-controlled regular files")
        if not os.access(self.guard_python, os.X_OK) or not os.access(spec.command[0], os.X_OK):
            raise SupervisionError("Prepared interpreter or application is not executable")

    def _namespace_binding(self):
        if self.network_namespace is None:
            return None
        try:
            return inspect_namespace(self.network_namespace, proc_root=self.host.proc_root, owner_uid=self.host.owner_uid)
        except (OSError, ValueError) as exc:
            raise SupervisionError("Owner network namespace is unavailable") from exc

    def resource_observation(self, resource_slice):
        if resource_slice != self.resource_slice:
            raise ObservationUncertain("Configured device slice differs from the persisted start intent")
        if resource_slice is None:
            return None
        try:
            device_resources.validate_slice(resource_slice)
            return device_resources.inspect_slice(self.host)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            raise ObservationUncertain("Owner aggregate resource boundary is unavailable") from exc

    def verify_receiver_resources(self):
        if self.resource_slice is None:
            if self.network_budget is not None or self.storage_profile is not None:
                raise ObservationUncertain("Device budgets require the aggregate device slice")
            return None
        try:
            result = device_resources.verify_receiver(self.host)
            result["device_storage"] = self.storage_observation(self.storage_profile)
            if self.network_budget is not None:
                binding = self._namespace_binding()
                result["receiver_network_namespace"] = verify_process_namespace(
                    binding, os.getpid(), proc_root=self.host.proc_root, owner_uid=self.host.owner_uid)
                result["network_budget"] = self.network_observation(self.network_budget, binding)
            return result
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            raise ObservationUncertain("Receiving process is outside the verified aggregate boundary") from exc

    def storage_observation(self, profile, binding=None, layout=None):
        if profile != self.storage_profile or layout is not None and layout != self.storage_layout:
            raise ObservationUncertain("Configured storage differs from the persisted start intent")
        if profile is None:
            return None
        try:
            if self.resource_slice != device_resources.SLICE:
                raise ValueError("Bounded storage requires the aggregate device slice")
            return device_storage.admit(self.host, self.storage_layout, expected_binding=binding)
        except (OSError, ValueError) as exc:
            raise ObservationUncertain("Owner storage admission boundary is unavailable") from exc

    def network_observation(self, profile, binding):
        if profile != self.network_budget:
            raise ObservationUncertain("Configured network budget differs from the persisted start intent")
        if profile is None:
            return None
        try:
            network_limits.validate_profile(profile)
            if (self.resource_slice != device_resources.SLICE or binding is None
                    or binding["path"] != self.network_namespace):
                raise ValueError("Shared budget requires the bound private namespace and aggregate slice")
            return network_enforcement.observe(self.host, binding, self.guard_python)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            error = ObservationUncertain("Owner shared network budget is unavailable")
            error.network_observation = getattr(exc, "observation", None)
            raise error from exc

    def _birth_slice(self, birth):
        intent = self.records.get(birth, "intent")
        value = intent["record"]["observation"].get("resource_slice") if intent else self.resource_slice
        return device_resources.validate_slice(value)

    def install_and_start(self, birth, spec):
        """Internal operation; caller holds the per-device supervisor lock."""
        self.preflight(spec)
        binding = self._namespace_binding()
        storage = self.storage_observation(self.storage_profile)
        runtime = None
        if storage:
            user, group = pwd.getpwnam(spec.user), grp.getgrnam(spec.group)
            runtime = runtime_storage.prepare(
                spec.working_directory, storage["binding"], birth["process_token"],
                uid=user.pw_uid, gid=group.gr_gid, owner_uid=self.host.owner_uid, owner_gid=os.getegid())
        text = render_unit(birth, spec, state_directory=self.state.directory, policy_id=self.state.policy.policy_id,
                           guard_python=self.guard_python, guard_script=self.guard_script,
                           network_namespace=self.network_namespace, resource_slice=self.resource_slice, storage=runtime)
        self.records.put(birth, "intent", {
            "unit": unit_name(birth), "unit_text": text, "unit_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "user": spec.user, "group": spec.group,
            "guard_sha256": hashlib.sha256(self.guard_script.read_bytes()).hexdigest(),
            "readiness": spec.readiness, "interpreter": spec.command[0],
            "working_directory": str(spec.working_directory),
            "readonly_paths": [str(path) for path in spec.readonly_paths],
            "library_paths": [str(path) for path in spec.library_paths],
            "network_namespace": binding,
            "resource_slice": self.resource_slice,
            "network_budget": self.network_budget,
            "storage_profile": self.storage_profile,
            "storage_layout": self.storage_layout,
            "storage_binding": storage["binding"] if storage else None,
            "runtime_storage": runtime,
        })
        path = self._file(birth)
        if os.path.lexists(path):
            _, actual = self._owned_file(birth)
            if actual != text:
                raise SupervisionError("Existing unit content differs; do not overwrite it")
        else:
            fd, temporary = tempfile.mkstemp(prefix="." + unit_name(birth) + "-", dir=self.host.unit_root)
            try:
                with os.fdopen(fd, "w") as target:
                    target.write(text)
                    target.flush()
                    os.fchmod(target.fileno(), 0o644)
                    os.fsync(target.fileno())
                # All substrate callers serialize this publication under the
                # device lock; never replace an existing operator-owned unit.
                if os.path.lexists(path):
                    raise SupervisionError("Unit appeared during publication")
                os.rename(temporary, path)
                directory_fd = os.open(self.host.unit_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if os.path.lexists(temporary):
                    os.unlink(temporary)
        try:
            loaded = self.host.run(["systemctl", "daemon-reload"])
            if loaded.returncode:
                raise SubmissionUncertain("Unit reload did not confirm success")
            started = self.host.run(["systemctl", "start", unit_name(birth)], timeout=40)
            if started.returncode:
                raise SubmissionUncertain("Service start did not confirm success")
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SubmissionUncertain("Service start observation was interrupted") from exc
        return {"unit": unit_name(birth), "unit_sha256": hashlib.sha256(text.encode()).hexdigest()}

    def _group_population(self, unit, reported_group, *, resource_slice=None):
        expected = device_resources.service_cgroup(unit, resource_slice)
        if reported_group not in ("", expected):
            raise ObservationUncertain("Service cgroup differs from its owned unit")
        group = self.host.cgroup_root / expected.lstrip("/")
        if not group.exists():
            return False
        if group.is_symlink() or not group.is_dir():
            raise ObservationUncertain("Unexpected cgroup path")
        fields = {}
        for line in (group / "cgroup.events").read_text().splitlines():
            key, value = line.split()
            fields[key] = value
        if fields.get("populated") not in ("0", "1"):
            raise ObservationUncertain("Cgroup population is unavailable")
        return fields["populated"] == "1"

    def _kernel_limits(self, name, *, resource_slice=None):
        group = self.host.cgroup_root / device_resources.service_cgroup(name, resource_slice).lstrip("/")
        observed = {field: (group / field).read_text().strip()
                    for field in ("memory.max", "memory.swap.max", "pids.max", "cpu.max")}
        if (observed["memory.max"] != str(MEMORY_MAX) or observed["memory.swap.max"] != "0"
                or observed["pids.max"] != str(TASKS_MAX)):
            raise ObservationUncertain("Kernel memory, swap or task limits differ from the native profile")
        fields = observed["cpu.max"].split()
        if len(fields) != 2 or not all(value.isdigit() for value in fields):
            raise ObservationUncertain("Kernel CPU quota is unavailable")
        quota, period = map(int, fields)
        if period <= 0 or quota != 4 * period:
            raise ObservationUncertain("Kernel CPU quota differs from the native profile")
        return observed

    def inspect(self, birth):
        name = unit_name(birth)
        try:
            resource_slice = self._birth_slice(birth)
            response = self.host.run(["systemctl", "show", name, "--no-pager", "--property=" + ",".join(SHOW_FIELDS)])
            value = parse_show(response.stdout)
            populated = self._group_population(name, value["ControlGroup"], resource_slice=resource_slice)
            if value["Id"] != name or not value["MainPID"].isdigit():
                raise ObservationUncertain("Observed service identity differs")
            pid = int(value["MainPID"])
            # Covers a receiver crash after claim_start but before publication.
            # Missing files alone never establish absence. Require a complete
            # manager observation, no job/PID, and an empty entire cgroup.
            if (not os.path.lexists(self._file(birth)) and value["LoadState"] == "not-found"
                    and value["ActiveState"] == "inactive" and pid == 0 and not populated
                    and value["Job"] in ("", "0") and not value["FragmentPath"] and not value["DropInPaths"]):
                return {"unit": name, "boot_id": self.host.boot_id, "properties": value,
                        "cgroup_populated": False, "unit_sha256": None, "state": "absent",
                        "native_observation": not self.host.is_fixture}
            if response.returncode:
                raise ObservationUncertain("Systemd observation did not succeed")
            path, text = self._owned_file(birth)
            if (value["Id"] != name or value["Description"] != description(birth)
                    or value["FragmentPath"] != str(path) or value["DropInPaths"]
                    or value["NeedDaemonReload"] != "no" or value["LoadState"] != "loaded"):
                raise ObservationUncertain("Systemd unit identity or loaded configuration differs")
            result = {"unit": name, "boot_id": self.host.boot_id, "properties": value,
                      "cgroup_populated": populated, "unit_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "native_observation": not self.host.is_fixture}
            if value["ActiveState"] in ("inactive", "failed") and pid == 0 and not populated and value["Job"] in ("", "0"):
                return {**result, "state": "absent"}
            if value["ActiveState"] != "active" or value["SubState"] != "running" or pid <= 1 or not populated:
                return {**result, "state": "pending"}
            if not re.fullmatch(r"[0-9a-f]{32}", value["InvocationID"]):
                raise ObservationUncertain("Service invocation identity is missing")
            expected_limits = {"Restart": "no", "KillMode": "control-group", "MemoryMax": str(MEMORY_MAX),
                               "MemorySwapMax": "0", "TasksMax": str(TASKS_MAX), "NoNewPrivileges": "yes",
                               "ProtectSystem": "strict", "Delegate": "no",
                               "Slice": resource_slice or "system.slice"}
            if any(value[key] != expected for key, expected in expected_limits.items()):
                raise ObservationUncertain("Effective service restrictions differ from the native profile")
            if value["CPUQuotaPerSecUSec"] not in ("4s", "4000000"):
                raise ObservationUncertain("Effective CPU quota differs from the native profile")
            intent = self.records.get(birth, "intent")["record"]["observation"]
            if value["User"] != intent["user"] or value["Group"] != intent["group"]:
                raise ObservationUncertain("Native service account differs from its start intent")
            binding = intent.get("network_namespace")
            if value["NetworkNamespacePath"] != (binding["path"] if binding else ""):
                raise ObservationUncertain("Loaded network namespace differs from the start intent")
            if binding is not None:
                result["network_namespace"] = verify_process_namespace(
                    binding, pid, proc_root=self.host.proc_root, owner_uid=self.host.owner_uid)
            aggregate = self.resource_observation(resource_slice)
            network = self.network_observation(intent.get("network_budget"), binding)
            storage = self.storage_observation(intent.get("storage_profile"), intent.get("storage_binding"),
                                               intent.get("storage_layout"))
            limits = self._kernel_limits(name, resource_slice=resource_slice)
            ticks = process_start_ticks((self.host.proc_root / str(pid) / "stat").read_text(), pid)
            runtime = (runtime_storage.observe_process(self.host, pid, intent["runtime_storage"])
                       if intent.get("storage_profile") is not None else None)
            memberships = (self.host.proc_root / str(pid) / "cgroup").read_text().splitlines()
            if "0::" + device_resources.service_cgroup(name, resource_slice) not in memberships:
                raise ObservationUncertain("Observed PID is not in the owned service cgroup")
            return {**result, "state": "running",
                    "kernel_limits": limits,
                    "aggregate_resources": aggregate,
                    "network_budget": network,
                    "device_storage": storage,
                    "runtime_storage": runtime,
                    "process": {"pid": pid, "start_identity": value["InvocationID"] + ":" + ticks,
                                "boot_id": self.host.boot_id}}
        except ObservationUncertain:
            raise
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            raise ObservationUncertain("Native process observation is unavailable") from exc

    def stop(self, birth):
        # Validate the exact locally owned unit before asking systemd to act.
        if not os.path.lexists(self._file(birth)):
            observation = self.inspect(birth)
            if observation["state"] != "absent":
                raise ObservationUncertain("Missing unit file does not establish process absence")
            return observation
        self._owned_file(birth)
        try:
            result = self.host.run(["systemctl", "stop", unit_name(birth)], timeout=25)
            if result.returncode:
                raise SubmissionUncertain("Service stop did not confirm success")
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SubmissionUncertain("Service stop observation was interrupted") from exc
        return self.inspect(birth)
