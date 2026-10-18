"""Durable bounded diagnostic jobs; retries read the existing request outcome."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sqlite3
import stat

from .capsule import canonical
from .equipment import DEVICE_ID


MAX_JOBS = 1024
MAX_TRACE_BYTES = 512 * 1024
REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")


class JobError(ValueError):
    pass


class ApplicationStore:
    def __init__(self, directory, start_token):
        if not REQUEST_ID.fullmatch(start_token):
            raise JobError("Invalid application incarnation")
        self.start_token = start_token
        self.path = directory / "application.sqlite3"
        marker = directory / "application.enrolled"
        new = not os.path.lexists(self.path)
        if new:
            with marker.open("xb") as handle:
                handle.write(b"organism-application-v1\n")
                handle.flush()
                os.fsync(handle.fileno())
            marker.chmod(0o600)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.close(fd)
        for path in (self.path, marker):
            entry = path.lstat()
            if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.geteuid()
                    or entry.st_nlink != 1 or stat.S_IMODE(entry.st_mode) & 0o077):
                raise JobError("Application journal must remain private")
        if marker.read_bytes() != b"organism-application-v1\n":
            raise JobError("Application enrollment marker differs")
        with self.db() as db:
            if new:
                db.execute("""CREATE TABLE jobs (
                    request_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, token TEXT NOT NULL,
                    status TEXT NOT NULL, digest TEXT NOT NULL, body TEXT NOT NULL)""")
            rows = db.execute("SELECT request_id,token,status,digest,body FROM jobs").fetchall()
            if len(rows) > MAX_JOBS:
                raise JobError("Application journal allowance exceeded")
            for request, token, status, digest, body in rows:
                value = self.decode(digest, body)
                if value["request_id"] != request or value["status"] != status:
                    raise JobError("Application journal identity differs")
                if status == "running" and token != start_token:
                    value["status"], value["failure"] = "interrupted", "Previous application incarnation ended"
                    self.save(db, value, token=token)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @contextmanager
    def db(self):
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

    @staticmethod
    def decode(digest, body):
        if len(body.encode()) > MAX_TRACE_BYTES or hashlib.sha256(body.encode()).hexdigest() != digest:
            raise JobError("Application trace is corrupt")
        return json.loads(body)

    def save(self, db, value, *, token=None):
        data = canonical(value)
        if len(data) > MAX_TRACE_BYTES:
            raise JobError("Application trace allowance exhausted")
        digest = hashlib.sha256(data).hexdigest()
        db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?) ON CONFLICT(request_id) DO UPDATE SET status=excluded.status,digest=excluded.digest,body=excluded.body",
                   (value["request_id"], value["device_id"], token or self.start_token, value["status"], digest, data.decode()))
        return digest

    def get(self, request_id):
        if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id):
            raise JobError("Invalid diagnostic request ID")
        with self.db() as db:
            row = db.execute("SELECT digest,body FROM jobs WHERE request_id=?", (request_id,)).fetchone()
        if row is None:
            return None
        return {"digest": row[0], "record": self.decode(*row)}

    def reserve(self, request_id, device_id, *, task_kind="temperature_v1"):
        existing = self.get(request_id)
        if task_kind not in ("temperature_v1", "equipment_interface_v1"):
            raise JobError("Invalid diagnostic task kind")
        if not isinstance(device_id, str) or not DEVICE_ID.fullmatch(device_id):
            raise JobError("Invalid diagnostic device ID")
        if existing:
            if existing["record"]["device_id"] != device_id:
                raise JobError("Request ID is already bound to another device")
            if existing["record"].get("task_kind", "temperature_v1") != task_kind:
                raise JobError("Request ID is already bound to another diagnostic interface")
            return existing, False
        value = {"request_id": request_id, "device_id": device_id, "status": "running",
                 "created_at": datetime.now(timezone.utc).isoformat(), "trace": {}, "submission": None,
                 "failure": None, "independently_verified": False}
        if task_kind != "temperature_v1":
            value["task_kind"] = task_kind
        with self.db() as db:
            if db.execute("SELECT count(*) FROM jobs").fetchone()[0] >= MAX_JOBS:
                raise JobError("Diagnostic request allowance exhausted")
            if db.execute("SELECT 1 FROM jobs WHERE request_id=?", (request_id,)).fetchone():
                raise JobError("Concurrent diagnostic admission; inspect the existing request")
            digest = self.save(db, value)
        return {"digest": digest, "record": value}, True

    def update(self, request_id, *, trace, status="running", submission=None, failure=None):
        if status not in ("running", "finished", "failed", "interrupted"):
            raise JobError("Invalid diagnostic completion state")
        with self.db() as db:
            row = db.execute("SELECT token,status,digest,body FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            if not row or row[0] != self.start_token or row[1] != "running":
                raise JobError("Diagnostic request is no longer owned by this incarnation")
            value = self.decode(row[2], row[3])
            value.update(trace=trace, status=status, submission=submission, failure=failure)
            digest = self.save(db, value)
        return {"digest": digest, "record": value}

    @staticmethod
    def public(value):
        if value is None:
            return {"status": "unknown_request"}
        record = value["record"]
        return {**{key: record[key] for key in ("request_id", "device_id", "status", "submission", "failure")},
                "trace_sha256": value["digest"], "independently_verified": False}
