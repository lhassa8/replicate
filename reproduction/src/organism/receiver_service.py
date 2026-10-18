"""Device-local authenticated loader loop; no model, task knowledge or planner.

The native CLI requires an already enrolled Linux device and owner-controlled
configuration. Injectable existing backends support explicitly labeled Mac
tests; the production entry point never selects a fixture backend.
"""

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import grp
import hashlib
import json
import os
from pathlib import Path
import platform
import pwd
import re
import signal
import stat
import threading

from .capsule import MAX_MANIFEST_BYTES, CapsuleError, canonical
from .device_resources import validate_slice
from . import device_storage
from .enrollment import Enrollment, MAX_POLICY_BYTES, unique_object
from .discovery_book import MAX_BOOK_BYTES
from .node_context import NodeContextError, NodeContextProvisioner
from .network_namespace import namespace_path
from .network_budget import validate_profile as validate_network_budget
from .peer_identity import LabNetwork
from .peer_transport import PeerServer
from .preparation import PackagePreparer, PreparationError
from .receiver import LocalReceiver
from .receiver_state import (
    AdmissionError, MAX_PREPARATION_FAILURES, PREPARATION_RETRY_SECONDS, ReceiverState,
)
from .supervision import Supervisor, SupervisorBusy
from .systemd_supervision import NativeLinuxHost, SupervisionError, SystemdBackend


SCHEMA = "organism-receiver-service-v1"
POLL_SECONDS = 2
MAX_CONFIG_BYTES = 16384
MAX_STATUS_BYTES = 65536
PATH_FIELDS = ("enrollment_file", "state_directory", "capsule_store", "prepared_store",
               "work_root", "identity_directory", "guard_python", "guard_script", "discovery_file", "context_root")


class ReceiverServiceError(ValueError):
    pass


def owned_file(path, maximum, *, owner_uid=None):
    owner_uid = os.geteuid() if owner_uid is None else owner_uid
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != owner_uid or info.st_nlink != 1
                or info.st_mode & 0o022 or info.st_size > maximum):
            raise ReceiverServiceError("Expected a bounded owner-controlled regular file")
        raw = source.read(maximum + 1)
    if len(raw) > maximum:
        raise ReceiverServiceError("Owner-controlled file exceeds its bound")
    return raw


def configuration(path, *, owner_uid=0):
    value = json.loads(owned_file(path, MAX_CONFIG_BYTES, owner_uid=owner_uid), object_pairs_hook=unique_object)
    expected = {"schema", "trusted_owner_public_key", "device_id", "bind_address", "bind_port",
                "lab_cidrs", "service_user", "service_group", *PATH_FIELDS}
    if (not isinstance(value, dict) or not expected <= set(value)
            or set(value) - expected - {"network_namespace", "resource_slice", "network_budget", "incoming_directory",
                                       "storage_profile"}
            or value["schema"] != SCHEMA):
        raise ReceiverServiceError("Unexpected receiving-service configuration fields")
    if "resource_slice" in value:
        if value["resource_slice"] is None:
            raise ReceiverServiceError("An explicit resource slice cannot be null")
        validate_slice(value["resource_slice"])
    if "network_namespace" in value:
        try:
            namespace_path(value["network_namespace"])
        except ValueError as exc:
            raise ReceiverServiceError("Invalid owner network namespace path") from exc
    if "network_budget" in value:
        if (value["network_budget"] is None or "network_namespace" not in value
                or "resource_slice" not in value):
            raise ReceiverServiceError("Shared network budget requires an explicit namespace and device slice")
        validate_network_budget(value["network_budget"])
    if "storage_profile" in value:
        if value["storage_profile"] is None or "resource_slice" not in value:
            raise ReceiverServiceError("Bounded storage requires an explicit profile and device slice")
        device_storage.validate_profile(value["storage_profile"])
        device_storage.validate_layout(value)
    if not isinstance(value["device_id"], str) or not re.fullmatch("[0-9a-f]{64}", value["device_id"]):
        raise ReceiverServiceError("Invalid configured device identity")
    try:
        public = base64.b64decode(value["trusted_owner_public_key"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ReceiverServiceError("Invalid configured owner key") from exc
    if len(public) != 32:
        raise ReceiverServiceError("Invalid configured owner key")
    for field in (*PATH_FIELDS, *(("incoming_directory",) if "incoming_directory" in value else ())):
        path = value[field]
        if (not isinstance(path, str) or not path.startswith("/") or Path(path).as_posix() != path
                or ".." in Path(path).parts or any(ord(character) < 32 for character in path)):
            raise ReceiverServiceError("Configured paths must be explicit absolute paths")
    roots = [Path(value[field]).resolve() for field in ("state_directory", "prepared_store", "work_root", "context_root")]
    if any(left.is_relative_to(right) or right.is_relative_to(left)
           for index, left in enumerate(roots) for right in roots[index + 1:]):
        raise ReceiverServiceError("Private state, readable installations and mutable work must have separate roots")
    if "incoming_directory" in value:
        incoming = Path(value["incoming_directory"]).resolve()
        occupied = [Path(value[field]).resolve() for field in (
            "state_directory", "capsule_store", "prepared_store", "work_root", "context_root", "identity_directory")]
        if any(incoming.is_relative_to(path) or path.is_relative_to(incoming) for path in occupied):
            raise ReceiverServiceError("Explicit transfer storage must be separate from state and installed work")
    for field in ("service_user", "service_group"):
        if value[field] == "root" or not isinstance(value[field], str) or not re.fullmatch("[a-z_][a-z0-9_-]{0,31}", value[field]):
            raise ReceiverServiceError("An unprivileged application account is required")
    if not isinstance(value["lab_cidrs"], list) or not all(isinstance(item, str) for item in value["lab_cidrs"]):
        raise ReceiverServiceError("Explicit lab subnets are required")
    network = LabNetwork(value["lab_cidrs"])
    network.address(value["bind_address"])
    if type(value["bind_port"]) is not int or not 1 <= value["bind_port"] <= 65535:
        raise ReceiverServiceError("A fixed native listening port is required")
    return value, public


class Workspaces:
    """Owner-controlled parent with one private application directory per individual."""

    def __init__(self, root, *, uid, gid, owner_uid, maximum):
        self.root = Path(root).absolute()
        self.uid, self.gid, self.owner_uid, self.maximum = uid, gid, owner_uid, maximum
        info = self.root.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != owner_uid or info.st_mode & 0o022
                or not info.st_mode & 0o001):
            raise ReceiverServiceError("Workspace parent must be owner-controlled and traversable")

    def prepare(self, individual_id):
        if not isinstance(individual_id, str) or not re.fullmatch("[0-9a-f]{32}", individual_id):
            raise ReceiverServiceError("Invalid workspace individual identity")
        path = self.root / individual_id
        if len(os.fsencode(path / "agent.sock")) >= 104:
            raise ReceiverServiceError("Application workspace exceeds the portable socket-path bound")
        entries = list(self.root.iterdir())
        if len(entries) > self.maximum or any(not re.fullmatch("[0-9a-f]{32}", entry.name) for entry in entries):
            raise ReceiverServiceError("Workspace inventory exceeds its declared identities or allowance")
        if not os.path.lexists(path):
            if len(entries) >= self.maximum:
                raise ReceiverServiceError("Workspace count allowance exhausted")
            path.mkdir(mode=0o700)
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                created = os.fstat(fd)
                if (self.uid, self.gid) != (created.st_uid, created.st_gid):
                    os.fchown(fd, self.uid, self.gid)
                os.fchmod(fd, 0o700)
                os.fsync(fd)
            finally:
                os.close(fd)
            parent = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        info = path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != self.uid or info.st_gid != self.gid
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise ReceiverServiceError("Existing individual workspace has different ownership or permissions")
        return path


class ReceiverService:
    def __init__(self, supervisor, preparer, workspaces, peer, *,
                 lab_cidrs=("127.0.0.0/24",), user="organism", group="organism", interval=POLL_SECONDS,
                 node_contexts=None):
        self.supervisor, self.preparer, self.workspaces, self.peer = supervisor, preparer, workspaces, peer
        self.state = supervisor.state
        if peer.receiver is not supervisor.receiver or preparer.policy.policy_id != self.state.policy.policy_id:
            raise ReceiverServiceError("Receiving-service components have different ownership")
        if not 0.01 <= interval <= 10:
            raise ReceiverServiceError("Invalid observation interval")
        self.lab_cidrs, self.user, self.group, self.interval = tuple(lab_cidrs), user, group, interval
        LabNetwork(self.lab_cidrs)
        self.node_contexts = node_contexts
        if node_contexts is not None:
            if (node_contexts.policy.policy_id != self.state.policy.policy_id or node_contexts.device_id != self.state.device_id
                    or tuple(node_contexts.book.body["lab_cidrs"]) != self.lab_cidrs):
                raise ReceiverServiceError("Application discovery context differs from its device or lab")
        elif not supervisor.backend.host.is_fixture:
            raise ReceiverServiceError("Native application activation requires its device-local discovery context")
        self.stop_requested, self.started = threading.Event(), threading.Event()
        self.lock_fd, self.previous_status = None, None
        self.failure = None

    def _acquire(self):
        fd = os.open(self.state.directory / ".receiver-lifecycle.lock",
                     os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or info.st_uid != os.geteuid() or info.st_mode & 0o077):
                raise ReceiverServiceError("Invalid receiving-service ownership lock")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise
        self.lock_fd = fd

    def status(self, result):
        encoded = canonical(result)
        if encoded == self.previous_status:
            return
        record = {"schema": SCHEMA, "policy_id": self.state.policy.policy_id, "device_id": self.state.device_id,
                  "boot_id": self.state.boot_id, "observed_at": datetime.now(timezone.utc).isoformat(),
                  "result": result, "qualification_passed": False}
        raw = canonical(record)
        if len(raw) > MAX_STATUS_BYTES:
            raise ReceiverServiceError("Receiving-service status exceeds its bound")
        pending = self.state.directory / ".receiver-status.pending"
        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        with os.fdopen(fd, "wb") as output:
            info = os.fstat(output.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_nlink != 1 or info.st_mode & 0o077):
                raise ReceiverServiceError("Invalid receiving-service status file")
            os.ftruncate(output.fileno(), 0)
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.replace(pending, self.state.directory / "receiver-status.json")
        fd = os.open(self.state.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.previous_status = encoded

    def tick(self):
        if self.lock_fd is None:
            raise ReceiverServiceError("Receiving-service ownership has not been acquired")
        if self.stop_requested.is_set():
            return {"state": "receiver_stopping"}
        snapshot = self.state.snapshot()
        birth = snapshot["current_birth"]
        if birth and birth["process_token"]:
            # This branch always observes the existing incarnation. A timeout
            # cannot turn into another start or another preparation.
            return self.supervisor.reconcile()
        if snapshot["owner_mode"] != "enabled" or not birth or birth["status"] != "staged":
            return {"state": "waiting", "owner_mode": snapshot["owner_mode"],
                    "birth_status": birth["status"] if birth else None, "useful_readiness_verified": False}
        if birth["starts"] >= self.state.policy.body["max_process_starts"]:
            return {"state": "start_allowance_exhausted", "operation_id": birth["operation_id"]}
        if birth.get("preparation_failures", 0) >= MAX_PREPARATION_FAILURES:
            return {"state": "preparation_allowance_exhausted", "operation_id": birth["operation_id"]}
        previous = birth.get("last_preparation_failure")
        if (previous and previous["boot_id"] == self.state.boot_id
                and self.state.clock() < previous["observed_clock"] + PREPARATION_RETRY_SECONDS):
            return {"state": "preparation_retry_wait", "operation_id": birth["operation_id"]}
        try:
            self.supervisor.backend.storage_observation(self.supervisor.backend.storage_profile)
            manifest = owned_file(self.supervisor.receiver.store / birth["release_id"] / ".manifest.json",
                                  MAX_MANIFEST_BYTES)
            if hashlib.sha256(manifest).hexdigest() != birth["manifest_sha256"]:
                raise ReceiverServiceError("Staged manifest differs from the admitted birth")
            work = self.workspaces.prepare(birth["individual_id"])
            prepared = self.preparer.prepare(self.supervisor.receiver.store / birth["release_id"],
                                             manifest, self.state.platform)
            node_context = (self.node_contexts.prepare(birth["individual_id"], birth["release_id"])
                            if self.node_contexts is not None else None)
            if self.stop_requested.is_set():
                return {"state": "receiver_stopping", "operation_id": birth["operation_id"]}
            spec = prepared.process_spec(work, lab_cidrs=self.lab_cidrs, user=self.user, group=self.group,
                                         node_context=node_context)
            if not self.supervisor.backend.host.is_fixture and spec.readiness is None:
                raise ReceiverServiceError("Native activation requires the carried readiness checker")
            return self.supervisor.start(birth["operation_id"], manifest, spec)
        except SupervisorBusy:
            return {"state": "supervisor_busy", "operation_id": birth["operation_id"]}
        except (AdmissionError, CapsuleError, NodeContextError, PreparationError, ReceiverServiceError, SupervisionError, OSError) as exc:
            latest = self.state.snapshot()
            current = latest["current_birth"]
            reason = (type(exc).__name__ + ": " + str(exc))[:256]
            if (latest["owner_mode"] == "enabled" and current and current["operation_id"] == birth["operation_id"]
                    and current["status"] == "staged" and not current["process_token"]
                    and current["starts"] == birth["starts"]):
                self.state.record_preparation_failure(birth["operation_id"], birth["starts"], reason)
            return {"state": "activation_not_confirmed", "operation_id": birth["operation_id"], "reason": reason}

    def run(self):
        owns_peer = False
        try:
            if self.peer._thread is not None:
                raise ReceiverServiceError("Receiving service requires its own unstarted peer listener")
            owns_peer = True
            self.supervisor.backend.verify_receiver_resources()
            self._acquire()
            self.peer.start()
            self.started.set()
            while not self.stop_requested.is_set():
                if not self.peer._thread.is_alive():
                    raise ReceiverServiceError("Owned authenticated listener stopped unexpectedly")
                if (self.supervisor.backend.network_budget is not None
                        or self.supervisor.backend.storage_profile is not None):
                    # A failed current resource check closes the owned listener
                    # and in-flight transfers in finally; existing claims remain.
                    self.supervisor.backend.verify_receiver_resources()
                try:
                    result = self.tick()
                except SupervisorBusy:
                    result = {"state": "supervisor_busy"}
                except (AdmissionError, SupervisionError, OSError) as exc:
                    result = {"state": "observation_uncertain", "reason": (type(exc).__name__ + ": " + str(exc))[:256]}
                self.status(result)
                self.stop_requested.wait(self.interval)
        except BaseException as exc:
            self.failure = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            try:
                if owns_peer:
                    self.peer.close()
            finally:
                if self.lock_fd is not None:
                    os.close(self.lock_fd)
                    self.lock_fd = None


def native_state(config, public, *, admit_storage=False):
    host = NativeLinuxHost()
    if admit_storage and config.get("storage_profile") is not None:
        device_storage.admit(host, config)
    policy = Enrollment(owned_file(config["enrollment_file"], MAX_POLICY_BYTES, owner_uid=0), public)
    state = ReceiverState(config["state_directory"], policy, config["device_id"])
    return state, host


def native_backend(state, host, config):
    return SystemdBackend(state, guard_python=config["guard_python"], guard_script=config["guard_script"], host=host,
                          network_namespace=config.get("network_namespace"), resource_slice=config.get("resource_slice"),
                          network_budget=config.get("network_budget"), storage_profile=config.get("storage_profile"),
                          storage_layout=(device_storage.validate_layout(config)
                                          if config.get("storage_profile") is not None else None))


def native_components(config, state, host, public):
    policy = state.policy
    user, group = pwd.getpwnam(config["service_user"]), grp.getgrnam(config["service_group"])
    if user.pw_uid == 0 or group.gr_gid == 0 or user.pw_gid != group.gr_gid:
        raise ReceiverServiceError("Configured application user and primary group must be unprivileged")
    receiver = LocalReceiver(state, config["capsule_store"])
    preparer = PackagePreparer(config["prepared_store"], policy)
    workspaces = Workspaces(config["work_root"], uid=user.pw_uid, gid=group.gr_gid, owner_uid=0,
                            maximum=policy.body["max_births_per_device"])
    backend = native_backend(state, host, config)
    contexts = NodeContextProvisioner(config["context_root"], policy, state.device_id, public,
                                      owned_file(config["discovery_file"], MAX_BOOK_BYTES),
                                      config["identity_directory"], group=group.gr_gid)
    return Supervisor(receiver, backend), preparer, workspaces, contexts


def apply_owner_control(state, backend_factory, action, reason):
    # Persist the owner fence before opening supervision records or touching
    # any package/work directory. Failure to observe exit remains explicit.
    control = state.owner_control(action, reason)
    try:
        observed = Supervisor(None, backend_factory()).reconcile()
    except (SupervisionError, OSError, ValueError) as exc:
        observed = {"state": "uncertain", "process_exit_verified": False,
                    "reason": (type(exc).__name__ + ": " + str(exc))[:256]}
    return {"control": control, "supervision": observed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("operation", choices=("serve", "status", "control"))
    parser.add_argument("--action", choices=("stop", "resume", "retire"))
    parser.add_argument("--reason")
    args = parser.parse_args()
    if platform.system() != "Linux" or os.geteuid() != 0:
        raise ReceiverServiceError("Native receiving service requires the owner-installed root Linux environment")
    config, public = configuration(args.config)
    state, host = native_state(config, public, admit_storage=args.operation == "serve")
    if args.operation == "control":
        if args.action is None or args.reason is None:
            parser.error("control requires --action and --reason")
        print(json.dumps(apply_owner_control(state, lambda: native_backend(state, host, config),
                                            args.action, args.reason), sort_keys=True))
    elif args.action is not None or args.reason is not None:
        parser.error("owner control arguments are only valid for control")
    elif args.operation == "status":
        supervisor = Supervisor(None, native_backend(state, host, config))
        observed = supervisor.reconcile()
        print(json.dumps({"snapshot": state.snapshot(), "supervision": observed}, sort_keys=True))
    else:
        supervisor, preparer, workspaces, contexts = native_components(config, state, host, public)
        # TCPServer binds/listens during construction, before its serving thread.
        supervisor.backend.verify_receiver_resources()
        peer = PeerServer(supervisor.receiver, config["identity_directory"],
                          bind=(config["bind_address"], config["bind_port"]), network=LabNetwork(config["lab_cidrs"]),
                          incoming_directory=config.get("incoming_directory"))
        service = ReceiverService(supervisor, preparer, workspaces, peer, lab_cidrs=config["lab_cidrs"],
                                  user=config["service_user"], group=config["service_group"], node_contexts=contexts)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: service.stop_requested.set())
        service.run()


if __name__ == "__main__":
    main()
