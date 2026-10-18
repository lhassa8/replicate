"""Small immutable supervision records, stored outside the received capsule."""

from contextlib import contextmanager
import hashlib
import json
import os
import re
import sqlite3
import stat

from .capsule import canonical


MAX_RECORD_BYTES = 65536
KINDS = ("intent", "running", "exit", "check_intent", "check_running", "check_result", "check_admission")


class RecordError(ValueError):
    pass


class SupervisionRecords:
    def __init__(self, state):
        self.state = state
        self.maximum = state.policy.body["max_births_per_device"] * state.policy.body["max_process_starts"] * len(KINDS)
        self.path = state.directory / "supervision.sqlite3"
        marker = state.directory / "supervision.enrolled"
        new = not os.path.lexists(self.path)
        if new:
            # Marker first: an interrupted creation or lost database cannot
            # silently reset previously recorded start and exit observations.
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as target:
                target.write(b"organism-supervision-v1\n")
                target.flush()
                os.fsync(target.fileno())
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.close(fd)
            fd = os.open(state.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        for path in (self.path, marker):
            entry = path.lstat()
            if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.geteuid()
                    or entry.st_nlink != 1 or stat.S_IMODE(entry.st_mode) & 0o077):
                raise RecordError("Supervision records must remain private to the receiving service")
        if marker.read_bytes() != b"organism-supervision-v1\n":
            raise RecordError("Supervision enrollment marker differs")
        with self._db() as db:
            if new:
                db.execute("""CREATE TABLE records (
                    token TEXT NOT NULL, kind TEXT NOT NULL, digest TEXT NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY(token,kind))""")
            rows = db.execute("SELECT token,kind,digest,body FROM records").fetchall()
            if len(rows) > self.maximum:
                raise RecordError("Supervision record allowance exceeded")
            for token, kind, digest, body in rows:
                self._decode(token, kind, digest, body)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=10)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _identity(self, birth, kind):
        token = birth["process_token"]
        if not isinstance(token, str) or not re.fullmatch("[0-9a-f]{32}", token) or kind not in KINDS:
            raise RecordError("Invalid supervision record identity")
        return {"schema": "organism-supervision-record-v1", "kind": kind,
                "policy_id": self.state.policy.policy_id, "device_id": self.state.device_id,
                "operation_id": birth["operation_id"], "individual_id": birth["individual_id"],
                "process_token": token, "manifest_sha256": birth["manifest_sha256"],
                "boot_id": self.state.boot_id}

    def _decode(self, token, kind, digest, body):
        data = body.encode()
        if len(data) > MAX_RECORD_BYTES or hashlib.sha256(data).hexdigest() != digest:
            raise RecordError("Supervision record is corrupt")
        value = json.loads(body)
        if (value.get("process_token") != token or value.get("kind") != kind
                or kind not in KINDS or value.get("policy_id") != self.state.policy.policy_id
                or value.get("device_id") != self.state.device_id):
            raise RecordError("Supervision record identity differs")
        return value

    def get(self, birth, kind):
        identity = self._identity(birth, kind)
        with self._db() as db:
            row = db.execute("SELECT digest,body FROM records WHERE token=? AND kind=?",
                             (identity["process_token"], kind)).fetchone()
        if row is None:
            return None
        value = self._decode(identity["process_token"], kind, *row)
        if any(value.get(key) != expected for key, expected in identity.items()):
            raise RecordError("Supervision record belongs to another start intent")
        return {"digest": row[0], "record": value}

    def put(self, birth, kind, observation):
        identity = self._identity(birth, kind)
        data = canonical({**identity, "observation": observation})
        if len(data) > MAX_RECORD_BYTES:
            raise RecordError("Supervision record exceeds its bound")
        digest = hashlib.sha256(data).hexdigest()
        with self._db() as db:
            previous = db.execute("SELECT digest,body FROM records WHERE token=? AND kind=?",
                                  (identity["process_token"], kind)).fetchone()
            if previous is not None:
                if previous != (digest, data.decode()):
                    raise RecordError("Existing supervision evidence cannot be replaced")
                return digest
            if db.execute("SELECT count(*) FROM records").fetchone()[0] >= self.maximum:
                raise RecordError("Supervision record allowance exhausted")
            db.execute("INSERT INTO records VALUES (?,?,?,?)",
                       (identity["process_token"], kind, digest, data.decode()))
        return digest
