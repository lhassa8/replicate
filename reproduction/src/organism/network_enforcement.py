"""Owner-side validation of the shared budget in an existing private namespace."""

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import time

from . import network_budget
from .network_namespace import inspect_namespace, namespace_path


MAX_PERSISTED_OBSERVATION = 24 * 1024


class NetworkEnforcementError(ValueError):
    def __init__(self, message, *, observation=None):
        super().__init__(message)
        self.observation = observation


def validate_observation(value, binding, boot_id, started, completed):
    """Recheck raw observations, not just a child process's success booleans."""
    try:
        if (not isinstance(value, dict) or value.get("schema") != "organism-native-network-budget-v1"
                or value.get("native_kernel_observation") is not True
                or value.get("profile_verified") is not True
                or value.get("filter_and_interface_membership_verified") is not True
                or value.get("qualification_passed") is not False or value.get("failure")
                or value["boot_id"] != boot_id
                or value["namespace"] != {"device": binding["device"], "inode": binding["inode"]}):
            raise ValueError("Incomplete or incorrectly bound network observation")
        first, last = value["started_monotonic_seconds"], value["completed_monotonic_seconds"]
        if (type(first) not in (int, float) or type(last) not in (int, float)
                or not math.isfinite(first) or not math.isfinite(last)
                or not started <= first <= last <= completed or completed - started > 15):
            raise ValueError("Network observation does not belong to this bounded call")
        raw = base64.b64decode(value["reply_base64"], validate=True)
        query = base64.b64decode(value["request_base64"], validate=True)
        if (hashlib.sha256(raw).hexdigest() != value["reply_sha256"]
                or query != network_budget.request(value["sequence"])
                or value["reply_sender"] != [0, 0]
                or type(value["reply_flags"]) is not int or value["reply_flags"] & socket.MSG_TRUNC):
            raise ValueError("Network capture identity differs")
        action = network_budget.decode_reply(raw, value["sequence"], value["psched"])
        attachments = network_budget.validate_attachments(value["attachment_readings"])
        if action != value["action"] or attachments != value["attachments"]:
            raise ValueError("Reported network policy differs from raw observations")
    except (KeyError, TypeError, ValueError) as exc:
        raise NetworkEnforcementError("Shared network budget could not be verified", observation=value) from exc
    return value


def observe(host, binding, interpreter):
    """Run only the owner observer; do not change the receiver thread's namespace."""
    capture = None
    observer = Path(network_budget.__file__).resolve()
    try:
        path = namespace_path(binding["path"])
        if inspect_namespace(path, proc_root=host.proc_root, owner_uid=host.owner_uid) != binding:
            raise ValueError("Owner namespace changed before budget inspection")
        for executable in (Path(interpreter), observer):
            entry = executable.lstat()
            if (not executable.is_absolute() or not stat.S_ISREG(entry.st_mode)
                    or entry.st_uid != host.owner_uid or entry.st_nlink != 1 or entry.st_mode & 0o022):
                raise ValueError("Network observer and interpreter must be owner-controlled regular files")
        if observer.stat().st_size > 128 * 1024 or not os.access(interpreter, os.X_OK):
            raise ValueError("Network observer executable or size differs")
        source_hash = hashlib.sha256(observer.read_bytes()).hexdigest()
        started = time.monotonic()
        response = host.run(
            ["nsenter", "--net=" + path, "--", str(interpreter), "-I", str(observer),
             "--collect", "--include-attachments"], timeout=15)
        completed = time.monotonic()
        if len(response.stdout.encode()) > 384 * 1024:
            raise ValueError("Network observer output exceeds its bound")
        capture = json.loads(response.stdout)
        if response.returncode != 0:
            raise NetworkEnforcementError("Owner network observer rejected the current profile", observation=capture)
        validate_observation(capture, binding, host.boot_id, started, completed)
        # Leave space for process and aggregate evidence in a 64 KiB record.
        if len(json.dumps(capture, sort_keys=True).encode()) > MAX_PERSISTED_OBSERVATION:
            raise ValueError("Network observation exceeds the persisted evidence bound")
        if (hashlib.sha256(observer.read_bytes()).hexdigest() != source_hash
                or inspect_namespace(path, proc_root=host.proc_root, owner_uid=host.owner_uid) != binding):
            raise ValueError("Owner observer or namespace changed during inspection")
        return {"profile": network_budget.PROFILE, "binding": binding, "observation": capture,
                "observer_sha256": source_hash, "native_observation": not host.is_fixture,
                "qualification_passed": False}
    except NetworkEnforcementError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise NetworkEnforcementError("Owner network budget is unavailable", observation=capture) from exc
