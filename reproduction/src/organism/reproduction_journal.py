"""Bounded reproduction intents and immutable transition history."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid

from .application_store import ApplicationStore
from .capsule import canonical
from .peer_audit import PeerAudit


TERMINAL = frozenset(("complete", "deferred", "decision_failed", "aborted", "failed", "stopped"))
PHASES = TERMINAL | {"proposed", "authorized", "reserving", "receiving", "waiting", "cancelling", "uncertain"}


class ReproductionError(ValueError):
    pass


class ReproductionJournal:
    def __init__(self, directory, config, book):
        self.directory = Path(directory).absolute()
        self.directory.mkdir(mode=0o700, exist_ok=True)
        entry = self.directory.lstat()
        if not stat.S_ISDIR(entry.st_mode) or entry.st_uid != os.geteuid() or entry.st_mode & 0o077:
            raise ReproductionError("Reproduction state must be private to the individual")
        self.identity = {"schema": "organism-reproduction-v1", "device_id": config.device_id,
                         "individual_id": config.individual_id, "release_id": config.release_id,
                         "policy_id": book.policy.policy_id, "book_id": book.book_id}
        self.start_token = config.start_token
        self.peers = set(book.body["endpoints"]) - {config.device_id}
        self.maximum = len(self.peers) * book.policy.body["max_births_per_device"]
        marker = self.directory / "identity.json"
        if not os.path.lexists(marker):
            if any(self.directory.iterdir()):
                raise ReproductionError("Reproduction identity is missing; preserve existing state")
            with marker.open("xb") as output:
                output.write(canonical(self.identity)); output.flush(); os.fsync(output.fileno())
            marker.chmod(0o600)
        else:
            info = marker.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1
                    or info.st_mode & 0o077 or info.st_size > 4096 or json.loads(marker.read_bytes()) != self.identity):
                raise ReproductionError("Reproduction journal belongs to another individual or release")
            for name in ("peer-audit.sqlite3", "peer-audit.enrolled", "application.sqlite3", "application.enrolled"):
                if not os.path.lexists(self.directory / name):
                    raise ReproductionError("Reproduction history is incomplete; do not reset it")
        self.audit = PeerAudit(self.directory)
        self.decisions = ApplicationStore(self.directory, config.start_token)
        self.operations = {}
        self.seen = set()
        for item in self.audit.iter_records():
            record = item["record"]
            if record["operation"] != "reproduce" or record["phase"] != "transition":
                raise ReproductionError("Unknown reproduction history record")
            value = record["detail"]
            self.validate(value)
            if value["operation_id"] != record["operation_id"] or value["peer_id"] != record["peer_id"]:
                raise ReproductionError("Reproduction transition identity differs")
            old = self.operations.get(value["operation_id"])
            if old and (old["phase"] in TERMINAL or any(old[key] != value[key] for key in
                        ("peer_id", "peer_revision", "boot_id", "issued_clock", "deadline_clock", "decision_token"))):
                raise ReproductionError("Reproduction history changed an existing intent")
            self.operations[value["operation_id"]] = value
            self.seen.add((value["peer_id"], value["peer_revision"]))
        if len(self.operations) > self.maximum or len(self.active()) > 1:
            raise ReproductionError("Reproduction intent allowance or single-child ownership differs")

    def validate(self, value):
        fields = {"operation_id", "peer_id", "peer_revision", "phase", "boot_id", "issued_clock",
                  "deadline_clock", "platform", "payload_index", "attempts", "receipt", "failure",
                  "decision_hash", "retry_clock"}
        fields.add("decision_token")
        if (not isinstance(value, dict) or set(value) != fields or value["peer_id"] not in self.peers
                or not re.fullmatch("[0-9a-f]{32}", value["operation_id"])
                or value["phase"] not in PHASES or type(value["peer_revision"]) is not int
                or not re.fullmatch("[0-9a-f]{32}", value["decision_token"])
                or value["peer_revision"] < 0 or type(value["payload_index"]) is not int
                or not 0 <= value["payload_index"] <= 11):
            raise ReproductionError("Invalid reproduction intent")
        if (not isinstance(value["attempts"], dict) or len(value["attempts"]) > 14
                or any(not re.fullmatch(r"reserve|commit|cancel|file-(?:[0-9]|10)", key)
                       or type(count) is not int or not 1 <= count <= 3 for key, count in value["attempts"].items())):
            raise ReproductionError("Invalid reproduction action allowance")
        if value["failure"] is not None and (not isinstance(value["failure"], str) or len(value["failure"]) > 192):
            raise ReproductionError("Reproduction failure exceeds its bound")
        if (any(type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] < 0
                for key in ("issued_clock", "deadline_clock", "retry_clock"))
                or value["deadline_clock"] != value["issued_clock"] + 1200
                or not isinstance(value["boot_id"], str) or not 1 <= len(value["boot_id"]) <= 128
                or value["platform"] not in (None, "linux-x86_64", "linux-arm64")
                or value["decision_hash"] is not None and not re.fullmatch("[0-9a-f]{64}", value["decision_hash"])):
            raise ReproductionError("Reproduction lifetime, procedure or decision identity differs")

    def active(self):
        return [deepcopy(value) for value in self.operations.values() if value["phase"] not in TERMINAL]

    def save(self, value):
        self.validate(value)
        old = self.operations.get(value["operation_id"])
        if old and old["phase"] in TERMINAL:
            raise ReproductionError("Completed reproduction evidence cannot be replaced")
        if old and any(old[key] != value[key] for key in
                       ("peer_id", "peer_revision", "boot_id", "issued_clock", "deadline_clock", "decision_token")):
            raise ReproductionError("Existing reproduction intent cannot be reassigned")
        if not old and (len(self.operations) >= self.maximum or self.active()):
            raise ReproductionError("Reproduction allowance or active child slot is occupied")
        self.audit.append(request_id=uuid.uuid4().hex, phase="transition", peer_id=value["peer_id"],
                          operation="reproduce", operation_id=value["operation_id"], detail=value)
        self.operations[value["operation_id"]] = deepcopy(value)
        self.seen.add((value["peer_id"], value["peer_revision"]))
        return deepcopy(value)

    def propose(self, candidate, *, boot_id, clock):
        if (candidate["device_id"], candidate["observation"]["revision"]) in self.seen:
            return None
        value = {"operation_id": uuid.uuid4().hex, "peer_id": candidate["device_id"],
                 "peer_revision": candidate["observation"]["revision"], "phase": "proposed",
                 "boot_id": boot_id, "issued_clock": clock, "deadline_clock": clock + 1200,
                 "platform": None, "payload_index": 0, "attempts": {}, "receipt": None, "failure": None,
                 "decision_hash": None, "retry_clock": clock}
        value["decision_token"] = self.start_token
        return self.save(value)

    def public(self):
        active = self.active()
        recent = list(self.operations.values())[-5:]
        return {"active_operation": active[0]["operation_id"] if active else None,
                "recorded_operations": len(self.operations), "operation_allowance": self.maximum,
                "recent": [{key: value[key] for key in ("operation_id", "peer_id", "phase", "failure")}
                           for value in recent],
                "completed_children": sum(value["phase"] == "complete" for value in self.operations.values()),
                "qualification_passed": False}
