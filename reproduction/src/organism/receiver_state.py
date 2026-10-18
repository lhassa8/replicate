"""Durable device ownership and birth lifecycle for the receiving substrate.

This module does not authenticate sockets, launch processes or prove readiness.
Peer-facing calls take an identity already authenticated by the transport;
supervisor and owner methods must be exposed only through local privileged
interfaces. The copied organism cannot write this store or reset enrollment.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import base64
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import time
import uuid

from .capsule import canonical, verify_manifest
from .enrollment import Enrollment, unique_object


SCHEMA = "organism-receiver-state-v1"
MAX_EVENTS = 8192
EVENT_RESERVE = 8
MAX_EVENT_BYTES = 8192
OBSERVATION_SECONDS = 30
MAX_PREPARATION_FAILURES = 3
PREPARATION_RETRY_SECONDS = 30
ID = re.compile(r"[0-9a-f]{32}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
TERMINAL = {"aborted", "failed", "stopped"}
PENDING = {"receiving", "staged", "starting"}


class AdmissionError(ValueError):
    pass


def boot_identity():
    """Runtime supports Linux; tests can provide an explicit boot identity."""
    path = Path("/proc/sys/kernel/random/boot_id")
    if not path.is_file():
        raise AdmissionError("A real Linux boot identity or explicit development identity is required")
    return path.read_text().strip()


def elapsed_clock():
    clock = getattr(time, "CLOCK_BOOTTIME", None)
    return time.clock_gettime(clock) if clock is not None else time.monotonic()


def _text_identity(value, pattern, label):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise AdmissionError("Invalid " + label)
    return value


def _private_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    observed = os.fstat(fd)
    if observed.st_uid != os.geteuid() or stat.S_IMODE(observed.st_mode) & 0o077:
        os.close(fd)
        raise AdmissionError("Receiver state directory must be private to its service owner")
    return fd


class ReceiverState:
    @classmethod
    def enroll(cls, directory, enrollment, device_id, *, boot_id=None, clock=elapsed_clock):
        """Owner-only first enrollment. Existing directories are never reset."""
        enrollment.platform(device_id)
        directory = Path(directory).absolute()
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        fd = _private_directory(directory)
        try:
            marker = directory / "enrolled.json"
            with marker.open("xb") as target:
                target.write(canonical({"schema": SCHEMA, "policy_id": enrollment.policy_id, "device_id": device_id}))
                target.flush()
                os.fsync(target.fileno())
            os.fsync(fd)
            parent_fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            path = directory / "state.sqlite3"
            created = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.close(created)
            connection = sqlite3.connect(path)
            try:
                connection.executescript("""
                    PRAGMA journal_mode=DELETE;
                    PRAGMA synchronous=FULL;
                    CREATE TABLE state (singleton INTEGER PRIMARY KEY CHECK(singleton=1), body TEXT NOT NULL);
                    CREATE TABLE births (operation_id TEXT PRIMARY KEY, body TEXT NOT NULL);
                    CREATE TABLE events (
                        sequence INTEGER PRIMARY KEY, previous_hash TEXT NOT NULL,
                        event_hash TEXT NOT NULL UNIQUE, body TEXT NOT NULL
                    );
                """)
                with connection:
                    state = {
                        "schema": SCHEMA, "policy_id": enrollment.policy_id, "device_id": device_id,
                        "owner_mode": "enabled", "revision": 0, "current_birth": None,
                        "boot_id": boot_id or boot_identity(), "last_clock": float(clock()), "birth_count": 0,
                        "observation_secret": secrets.token_hex(32),
                    }
                    connection.execute("INSERT INTO state VALUES (1,?)", (canonical(state).decode(),))
                os.fsync(fd)
            finally:
                connection.close()
        finally:
            os.close(fd)
        return cls(directory, enrollment, device_id, boot_id=boot_id, clock=clock)

    def __init__(self, directory, enrollment: Enrollment, device_id, *, boot_id=None, clock=elapsed_clock):
        self.directory = Path(directory).absolute()
        self.policy = enrollment
        self.device_id = device_id
        self.platform = enrollment.platform(device_id)
        self.boot_id = boot_id or boot_identity()
        if not isinstance(self.boot_id, str) or not 1 <= len(self.boot_id) <= 128:
            raise AdmissionError("Invalid boot identity")
        self.clock = clock
        directory_fd = _private_directory(self.directory)
        try:
            for name in ("enrolled.json", "state.sqlite3"):
                file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
                try:
                    entry = os.fstat(file_fd)
                    if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1 or entry.st_uid != os.geteuid():
                        raise AdmissionError("Invalid receiver state file")
                finally:
                    os.close(file_fd)
            marker = json.loads((self.directory / "enrolled.json").read_text())
            if marker != {"schema": SCHEMA, "policy_id": enrollment.policy_id, "device_id": device_id}:
                raise AdmissionError("Enrollment identity changed; explicit owner migration is required")
        finally:
            os.close(directory_fd)
        # URI mode=rw prevents a lost state file from becoming a fresh receiver.
        self.path = self.directory / "state.sqlite3"
        with self._transaction() as (db, state, now):
            self._verify_events(db, state)
            self._reconcile_boot(db, state, now)

    @contextmanager
    def _transaction(self, *, critical=False):
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=10)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA busy_timeout=10000")
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM state WHERE singleton=1").fetchone()
            if not row:
                raise AdmissionError("Receiver enrollment state is missing")
            state = json.loads(row[0])
            if (state["schema"], state["policy_id"], state["device_id"]) != (SCHEMA, self.policy.policy_id, self.device_id):
                raise AdmissionError("Receiver identity or enrollment policy changed")
            now = float(self.clock())
            if not math.isfinite(now) or now < 0:
                raise AdmissionError("Invalid monotonic clock")
            if state["boot_id"] == self.boot_id and now < state["last_clock"] and not critical:
                raise AdmissionError("Monotonic clock moved backwards; admission is closed")
            yield db, state, now
            state["last_clock"] = now
            db.execute("UPDATE state SET body=? WHERE singleton=1", (canonical(state).decode(),))
            db.commit()
        except sqlite3.DatabaseError as exc:
            db.rollback()
            raise AdmissionError("Receiver journal is unavailable; admission is closed") from exc
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _event(self, db, state, kind, detail, *, critical=False):
        last = db.execute("SELECT sequence,event_hash FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
        sequence, previous = (last[0] + 1, last[1]) if last else (1, "0" * 64)
        if sequence > MAX_EVENTS - (0 if critical else EVENT_RESERVE):
            raise AdmissionError("Receiver journal allowance exhausted")
        state["revision"] += 1
        body = canonical({
            "sequence": sequence, "revision": state["revision"], "kind": kind, "detail": detail,
            "boot_id": self.boot_id, "recorded_at": datetime.now(timezone.utc).isoformat(),
        })
        if len(body) > MAX_EVENT_BYTES:
            raise AdmissionError("Receiver event exceeds bounds")
        event_hash = hashlib.sha256(bytes.fromhex(previous) + body).hexdigest()
        db.execute("INSERT INTO events VALUES (?,?,?,?)", (sequence, previous, event_hash, body.decode()))

    @staticmethod
    def _birth(db, operation_id):
        row = db.execute("SELECT body FROM births WHERE operation_id=?", (operation_id,)).fetchone()
        return json.loads(row[0]) if row else None

    @staticmethod
    def _save_birth(db, birth):
        db.execute("INSERT INTO births VALUES (?,?) ON CONFLICT(operation_id) DO UPDATE SET body=excluded.body",
                   (birth["operation_id"], canonical(birth).decode()))

    def _reconcile_boot(self, db, state, now):
        if state["boot_id"] == self.boot_id:
            return
        birth = self._birth(db, state["current_birth"]) if state["current_birth"] else None
        if birth:
            birth["process_token"] = None
            birth["process_observation"] = None
            if birth["ready_once"] and state["owner_mode"] == "enabled":
                birth["status"] = "staged"
                birth["restart_required"] = True
            else:
                birth["status"] = "stopped" if state["owner_mode"] != "enabled" else "aborted"
                state["current_birth"] = None
            self._save_birth(db, birth)
        old_boot = state["boot_id"]
        state["boot_id"] = self.boot_id
        self._event(db, state, "boot_changed", {"previous_boot": old_boot, "current_birth": state["current_birth"]},
                    critical=state["owner_mode"] != "enabled")

    def _expire(self, db, state, now):
        if not state["current_birth"]:
            return
        birth = self._birth(db, state["current_birth"])
        # Once a useful child is acknowledged, its ownership is not a parent lease.
        if birth["ready_once"] or birth["status"] not in PENDING | {"running"} or now < birth["deadline"]:
            return
        if birth["process_token"]:
            birth["status"] = "stop_requested"
        else:
            birth["status"] = "aborted"
            state["current_birth"] = None
        self._save_birth(db, birth)
        self._event(db, state, "birth_expired", {"operation_id": birth["operation_id"], "status": birth["status"]})

    def snapshot(self):
        with self._transaction() as (db, state, now):
            self._reconcile_boot(db, state, now)
            self._expire(db, state, now)
            birth = self._birth(db, state["current_birth"]) if state["current_birth"] else None
            events = db.execute("SELECT count(*) FROM events").fetchone()[0]
            return {
                "device_id": self.device_id, "policy_id": self.policy.policy_id, "platform": self.platform,
                "revision": state["revision"], "owner_mode": state["owner_mode"],
                "available": state["owner_mode"] == "enabled" and birth is None
                             and state["birth_count"] < self.policy.body["max_births_per_device"]
                             and events < MAX_EVENTS - EVENT_RESERVE,
                "current_birth": birth, "birth_count": state["birth_count"], "boot_id": state["boot_id"],
            }

    def reserve_birth(self, authenticated_peer, operation_id, observed_revision, manifest, *, admission_ticket):
        """Reserve from a fresh observation; retries never create a second birth."""
        self.policy.peer_key(authenticated_peer)
        if authenticated_peer == self.device_id:
            raise AdmissionError("A device cannot parent itself")
        _text_identity(operation_id, ID, "operation identity")
        if type(observed_revision) is not int or observed_revision < 0:
            raise AdmissionError("Invalid observed revision")
        body = self.policy.body
        envelope = verify_manifest(manifest, self.policy.release_keys, require_trained=body["require_trained"],
                                   max_payload_bytes=body["max_payload_bytes"])
        if envelope["body"]["experiment_id"] != body["experiment_id"]:
            raise AdmissionError("Capsule belongs to another experiment")
        intent = {"parent_device_id": authenticated_peer, "release_id": envelope["release_id"],
                  "manifest_sha256": hashlib.sha256(manifest).hexdigest()}
        with self._transaction() as (db, state, now):
            self._verify_events(db, state)
            self._reconcile_boot(db, state, now)
            self._expire(db, state, now)
            existing = self._birth(db, operation_id)
            if existing:
                if any(existing[key] != value for key, value in intent.items()):
                    raise AdmissionError("Operation identity was reused with a different intent")
                return existing
            if state["owner_mode"] != "enabled":
                raise AdmissionError("Owner has disabled admission")
            self._verify_ticket(state, now, admission_ticket, observed_revision)
            if state["revision"] != observed_revision:
                raise AdmissionError("Device observation is stale")
            if state["current_birth"] is not None:
                raise AdmissionError("Device already has an owner or birth reservation")
            if state["birth_count"] >= body["max_births_per_device"]:
                raise AdmissionError("Device birth allowance exhausted")
            birth = {
                "operation_id": operation_id, **intent, "individual_id": uuid.uuid4().hex,
                "device_id": self.device_id, "status": "receiving", "deadline": now + body["birth_seconds"],
                "payload_bytes": sum(entry["bytes"] for entry in envelope["body"]["files"].values()),
                "process_token": None, "process_observation": None,
                "starts": 0, "ready_once": False, "restart_required": False, "ready_evidence": None,
                "last_exit": None, "cancelled": False,
                "preparation_failures": 0, "last_preparation_failure": None,
            }
            state["birth_count"] += 1
            state["current_birth"] = operation_id
            self._save_birth(db, birth)
            self._event(db, state, "birth_reserved", {**intent, "operation_id": operation_id,
                        "individual_id": birth["individual_id"], "deadline": birth["deadline"]})
            return birth

    def abort_birth(self, authenticated_peer, operation_id, reason):
        """The parent can cancel an unfinished birth, never a useful child."""
        self.policy.peer_key(authenticated_peer)
        if not isinstance(reason, str) or not 1 <= len(reason) <= 256:
            raise AdmissionError("Invalid cancellation reason")
        with self._transaction() as (db, state, now):
            birth = self._birth(db, operation_id)
            if not birth or birth["parent_device_id"] != authenticated_peer:
                raise AdmissionError("Cancellation is for another parent's birth")
            if birth["ready_once"]:
                raise AdmissionError("A useful child's ownership is independent of its parent")
            if birth["status"] in TERMINAL:
                return birth
            if birth["cancelled"]:
                return birth
            self._current(db, state, operation_id)
            birth["cancelled"] = True
            if birth["process_token"]:
                birth["status"] = "stop_requested"
            else:
                birth["status"] = "aborted"
                state["current_birth"] = None
            self._save_birth(db, birth)
            self._event(db, state, "parent_cancelled_unfinished_birth",
                        {"operation_id": operation_id, "reason": reason, "status": birth["status"]})
            return birth

    def advertisement(self):
        """Public admission facts plus an expiring, receiver-authenticated ticket."""
        # First reconcile any expired lease. Ticket issuance then uses one
        # transaction's revision and clock so concurrent admission cannot mix them.
        self.snapshot()
        with self._transaction() as (db, state, now):
            ticket_body = {
                "device_id": self.device_id, "policy_id": self.policy.policy_id, "revision": state["revision"],
                "boot_id": self.boot_id, "issued": now, "expires": now + OBSERVATION_SECONDS,
                "nonce": uuid.uuid4().hex,
            }
            ticket = {"body": ticket_body, "mac": hmac.new(
                bytes.fromhex(state["observation_secret"]), canonical(ticket_body), hashlib.sha256,
            ).hexdigest()}
            events = db.execute("SELECT count(*) FROM events").fetchone()[0]
            return {
                "device_id": self.device_id, "policy_id": self.policy.policy_id, "platform": self.platform,
                "revision": state["revision"], "owner_mode": state["owner_mode"],
                "available": state["owner_mode"] == "enabled" and state["current_birth"] is None
                             and state["birth_count"] < self.policy.body["max_births_per_device"]
                             and events < MAX_EVENTS - EVENT_RESERVE,
                "admission_ticket": base64.b64encode(canonical(ticket)).decode(),
                "observation_valid_seconds": OBSERVATION_SECONDS,
            }

    def _verify_ticket(self, state, now, encoded, observed_revision):
        try:
            if not isinstance(encoded, str) or not 1 <= len(encoded) <= 2048:
                raise ValueError()
            ticket = json.loads(base64.b64decode(encoded, validate=True), object_pairs_hook=unique_object)
            if set(ticket) != {"body", "mac"}:
                raise ValueError()
            expected = hmac.new(bytes.fromhex(state["observation_secret"]), canonical(ticket["body"]), hashlib.sha256).hexdigest()
            if not isinstance(ticket["mac"], str) or not hmac.compare_digest(expected, ticket["mac"]):
                raise ValueError()
            body = ticket["body"]
            if (body["device_id"], body["policy_id"], body["boot_id"]) != (self.device_id, self.policy.policy_id, self.boot_id):
                raise ValueError()
            # Reproduce the issuer's sum: subtracting fractional boot clocks
            # can round a valid duration away from 30. Do not add a tolerance.
            if not body["issued"] <= now < body["expires"] or body["expires"] != body["issued"] + OBSERVATION_SECONDS:
                raise AdmissionError("Device observation has expired")
            if body["revision"] != observed_revision or observed_revision != state["revision"]:
                raise AdmissionError("Device observation is stale")
        except AdmissionError:
            raise
        except (KeyError, TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise AdmissionError("Device observation ticket is invalid") from exc

    def peer_receipt(self, authenticated_peer, operation_id):
        self.policy.peer_key(authenticated_peer)
        with self._transaction() as (db, state, now):
            self._expire(db, state, now)
            birth = self._birth(db, operation_id)
            if not birth or birth["parent_device_id"] != authenticated_peer:
                raise AdmissionError("Birth receipt is not addressed to this peer")
            receipt = {name: birth[name] for name in (
                "operation_id", "parent_device_id", "device_id", "individual_id",
                "release_id", "status", "ready_once", "ready_evidence",
            )}
            receipt["preparation_failures"] = birth.get("preparation_failures", 0)
            receipt["preparation_allowance_exhausted"] = receipt["preparation_failures"] >= MAX_PREPARATION_FAILURES
            return receipt

    def self_receipt(self, authenticated_device):
        """Read-only local admission facts, available only to this device key."""
        if authenticated_device != self.device_id:
            raise AdmissionError("Local admission receipt is addressed only to this device")
        snapshot = self.snapshot()
        birth = snapshot["current_birth"]
        token = birth["process_token"] if birth else None
        ready = bool(snapshot["owner_mode"] == "enabled" and birth and birth["status"] == "ready"
                     and token and birth["ready_evidence"])
        return {"device_id": self.device_id, "policy_id": self.policy.policy_id,
                "owner_mode": snapshot["owner_mode"], "status": birth["status"] if birth else "empty",
                "individual_id": birth["individual_id"] if birth else None,
                "release_id": birth["release_id"] if birth else None,
                "incarnation_sha256": hashlib.sha256(token.encode()).hexdigest() if token else None,
                "ready": ready, "ready_evidence": birth["ready_evidence"] if ready else None}

    def _current(self, db, state, operation_id):
        if state["current_birth"] != operation_id:
            raise AdmissionError("Birth no longer owns this device")
        birth = self._birth(db, operation_id)
        if birth is None:
            raise AdmissionError("Current birth record is missing")
        return birth

    def record_staged(self, operation_id, manifest_sha256):
        """Local receiver only, after complete capsule verification and fsync."""
        with self._transaction() as (db, state, now):
            self._expire(db, state, now)
            birth = self._current(db, state, operation_id)
            if state["owner_mode"] != "enabled" or manifest_sha256 != birth["manifest_sha256"]:
                raise AdmissionError("Staging is no longer authorized")
            if birth["status"] == "staged":
                return birth
            if birth["status"] != "receiving":
                raise AdmissionError("Birth is not receiving")
            birth["status"] = "staged"
            self._save_birth(db, birth)
            self._event(db, state, "capsule_staged", {"operation_id": operation_id, "release_id": birth["release_id"]})
            return birth

    def claim_start(self, operation_id):
        """Local supervisor persists a unique intent BEFORE spawning a process."""
        with self._transaction() as (db, state, now):
            self._verify_events(db, state)
            self._expire(db, state, now)
            birth = self._current(db, state, operation_id)
            if state["owner_mode"] != "enabled" or birth["status"] != "staged" or birth["process_token"]:
                raise AdmissionError("Process start is not authorized")
            if birth["starts"] >= self.policy.body["max_process_starts"]:
                raise AdmissionError("Process restart allowance exhausted")
            if birth.get("preparation_failures", 0) >= MAX_PREPARATION_FAILURES:
                raise AdmissionError("Preparation failure allowance exhausted")
            previous = birth.get("last_preparation_failure")
            if (previous and previous["boot_id"] == self.boot_id
                    and now < previous["observed_clock"] + PREPARATION_RETRY_SECONDS):
                raise AdmissionError("Preparation retry is too early")
            birth.update(status="starting", process_token=uuid.uuid4().hex, process_observation=None,
                         starts=birth["starts"] + 1, restart_required=False)
            self._save_birth(db, birth)
            self._event(db, state, "process_start_claimed", {
                "operation_id": operation_id, "process_token": birth["process_token"], "attempt": birth["starts"]})
            return birth

    def record_preparation_failure(self, operation_id, expected_starts, reason):
        """Local loader records a failed preparation without inventing a parent action."""
        if type(expected_starts) is not int or not isinstance(reason, str) or not 1 <= len(reason) <= 256:
            raise AdmissionError("Invalid preparation failure")
        with self._transaction() as (db, state, now):
            self._expire(db, state, now)
            birth = self._current(db, state, operation_id)
            if (state["owner_mode"] != "enabled" or birth["status"] != "staged"
                    or birth["process_token"] or birth["starts"] != expected_starts):
                raise AdmissionError("Preparation failure belongs to another activation state")
            failures = birth.get("preparation_failures", 0)
            if failures >= MAX_PREPARATION_FAILURES:
                raise AdmissionError("Preparation failure allowance exhausted")
            previous = birth.get("last_preparation_failure")
            if (previous and previous["boot_id"] == self.boot_id
                    and now < previous["observed_clock"] + PREPARATION_RETRY_SECONDS):
                raise AdmissionError("Preparation retry is too early")
            failure = {"boot_id": self.boot_id, "observed_clock": now, "reason": reason}
            birth.update(preparation_failures=failures + 1, last_preparation_failure=failure)
            self._save_birth(db, birth)
            self._event(db, state, "local_preparation_failed", {
                "operation_id": operation_id, "starts": expected_starts, "failure_number": failures + 1, **failure})
            return birth

    def record_running(self, operation_id, process_token, observation):
        """Local supervisor only; observation must identify its actual process."""
        if not isinstance(observation, dict) or set(observation) != {"pid", "start_identity", "boot_id"}:
            raise AdmissionError("Incomplete process observation")
        if type(observation["pid"]) is not int or observation["pid"] <= 1 or observation["boot_id"] != self.boot_id:
            raise AdmissionError("Invalid process observation")
        if not isinstance(observation["start_identity"], str) or not 1 <= len(observation["start_identity"]) <= 128:
            raise AdmissionError("Missing process start identity")
        with self._transaction() as (db, state, now):
            self._expire(db, state, now)
            birth = self._current(db, state, operation_id)
            if state["owner_mode"] != "enabled" or birth["process_token"] != process_token:
                raise AdmissionError("Stale process ownership token")
            if birth["status"] == "running" and birth["process_observation"] == observation:
                return birth
            if birth["status"] != "starting":
                raise AdmissionError("Process was not claimed for startup")
            birth.update(status="running", process_observation=observation)
            self._save_birth(db, birth)
            self._event(db, state, "process_observed", {"operation_id": operation_id, **observation})
            return birth

    def record_ready(self, operation_id, process_token, evidence_hash, *, valid_until=None):
        """Local readiness verifier records verified useful-child evidence."""
        _text_identity(evidence_hash, HASH, "readiness evidence hash")
        if valid_until is not None and (type(valid_until) not in (int, float)
                                        or not math.isfinite(valid_until) or valid_until < 0):
            raise AdmissionError("Invalid readiness evidence deadline")
        with self._transaction() as (db, state, now):
            self._expire(db, state, now)
            birth = self._current(db, state, operation_id)
            if (state["owner_mode"] != "enabled" or birth["process_token"] != process_token
                    or birth["status"] not in ("running", "ready")):
                raise AdmissionError("Child is not eligible for readiness")
            if birth["status"] == "ready":
                if birth["ready_evidence"] != evidence_hash:
                    raise AdmissionError("Readiness acknowledgement differs")
                return birth
            if valid_until is not None and now >= valid_until:
                raise AdmissionError("Readiness evidence expired before acknowledgement")
            birth.update(status="ready", ready_once=True, ready_evidence=evidence_hash, restart_required=False)
            self._save_birth(db, birth)
            self._event(db, state, "useful_child_acknowledged", {
                "operation_id": operation_id, "individual_id": birth["individual_id"], "evidence_hash": evidence_hash})
            return birth

    def record_process_gone(self, operation_id, process_token, evidence_hash):
        """Local supervisor only, after verifying the owned process is absent."""
        _text_identity(evidence_hash, HASH, "process-exit evidence")
        with self._transaction(critical=True) as (db, state, now):
            earlier = self._birth(db, operation_id)
            exit_receipt = {"process_token": process_token, "evidence_hash": evidence_hash}
            if earlier and earlier["process_token"] is None and earlier["last_exit"] == exit_receipt:
                return earlier
            birth = self._current(db, state, operation_id)
            if birth["process_token"] != process_token or process_token is None:
                raise AdmissionError("Exit observation is for another process")
            if state["owner_mode"] != "enabled":
                birth.update(status="stopped", restart_required=False)
                state["current_birth"] = None
            elif birth["cancelled"]:
                birth.update(status="aborted", restart_required=False)
                state["current_birth"] = None
            elif birth["ready_once"] or now < birth["deadline"]:
                birth.update(status="staged", restart_required=True)
            else:
                birth.update(status="failed", restart_required=False)
                state["current_birth"] = None
            birth.update(process_token=None, process_observation=None, last_exit=exit_receipt)
            self._save_birth(db, birth)
            self._event(db, state, "process_exit_verified", {
                "operation_id": operation_id, "process_token": process_token, "evidence_hash": evidence_hash,
                "resulting_status": birth["status"]}, critical=True)
            return birth

    def owner_control(self, action, reason):
        """Owner-only local control. Requesting stop does not prove process exit."""
        if action not in ("stop", "resume", "retire") or not isinstance(reason, str) or not 1 <= len(reason) <= 256:
            raise AdmissionError("Invalid owner control")
        with self._transaction(critical=action != "resume") as (db, state, now):
            mode = {"stop": "stopped", "resume": "enabled", "retire": "retired"}[action]
            if state["owner_mode"] == "retired" and action != "retire":
                raise AdmissionError("Retirement is permanent for this enrolled device identity")
            if state["owner_mode"] == mode:
                return {"owner_mode": mode, "revision": state["revision"],
                        "process_stop_required": mode != "enabled" and bool(state["current_birth"])}
            if action == "resume" and state["current_birth"]:
                raise AdmissionError("Verify the old process has stopped before resuming admission")
            state["owner_mode"] = mode
            if state["current_birth"]:
                birth = self._birth(db, state["current_birth"])
                if birth["process_token"]:
                    birth["status"] = "stop_requested"
                else:
                    birth["status"] = "stopped"
                    state["current_birth"] = None
                self._save_birth(db, birth)
            self._event(db, state, "owner_" + action, {"reason": reason}, critical=action != "resume")
            return {"owner_mode": mode, "revision": state["revision"], "process_stop_required": bool(state["current_birth"])}

    @staticmethod
    def _verify_events(db, state):
        rows = db.execute("SELECT sequence,previous_hash,event_hash,body FROM events ORDER BY sequence").fetchall()
        previous = "0" * 64
        result = []
        if len(rows) != state["revision"]:
            raise AdmissionError("Receiver event history is incomplete")
        for expected, (sequence, stored_previous, event_hash, text) in enumerate(rows, 1):
            body = text.encode()
            if sequence != expected or previous != stored_previous or hashlib.sha256(bytes.fromhex(previous) + body).hexdigest() != event_hash:
                raise AdmissionError("Receiver event chain is corrupt")
            event = json.loads(text)
            if event["sequence"] != expected or event["revision"] != expected:
                raise AdmissionError("Receiver event sequence differs from its journal")
            result.append({"event": event, "hash": event_hash})
            previous = event_hash
        return result

    def events(self):
        with self._transaction() as (db, state, now):
            return self._verify_events(db, state)
