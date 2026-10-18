"""Bounded durable transport attempts, without payloads, tickets or private keys."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat

from .capsule import canonical


MAX_RECORDS = 65536
MAX_RECORD_BYTES = 2048


class PeerAuditError(ValueError):
    pass


class PeerAudit:
    def __init__(self, directory):
        self.path = Path(directory).absolute() / "peer-audit.sqlite3"
        marker = self.path.parent / "peer-audit.enrolled"
        new = not os.path.lexists(self.path)
        if new and os.path.lexists(marker):
            raise PeerAuditError("Transport audit is missing; do not reset its history")
        if not new and (marker.is_symlink() or not marker.is_file()):
            raise PeerAuditError("Transport audit enrollment marker is missing")
        if new:
            marker_fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(marker_fd, "wb") as target:
                target.write(b"organism-peer-audit-v1\n")
                target.flush()
                os.fsync(target.fileno())
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            os.close(fd)
            directory_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        elif marker.read_bytes() != b"organism-peer-audit-v1\n":
            raise PeerAuditError("Transport audit enrollment marker changed")
        entry = self.path.lstat()
        if not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.geteuid() or entry.st_nlink != 1:
            raise PeerAuditError("Invalid transport audit file")
        with self._db() as db:
            if new:
                db.execute("CREATE TABLE records (sequence INTEGER PRIMARY KEY, previous TEXT NOT NULL, digest TEXT NOT NULL, body TEXT NOT NULL)")
                db.execute("CREATE TABLE head (singleton INTEGER PRIMARY KEY CHECK(singleton=1), sequence INTEGER NOT NULL, digest TEXT NOT NULL)")
                db.execute("INSERT INTO head VALUES (1,0,?)", ("0" * 64,))
        # Startup verification must not materialize a 72-hour journal in RAM.
        for _ in self.iter_records():
            pass

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

    def append(self, *, request_id, phase, peer_id=None, operation=None, operation_id=None, detail=None):
        value = {"request_id": request_id, "phase": phase, "peer_id": peer_id, "operation": operation,
                 "operation_id": operation_id, "detail": detail or {},
                 "recorded_at": datetime.now(timezone.utc).isoformat()}
        data = canonical(value)
        if len(data) > MAX_RECORD_BYTES:
            raise PeerAuditError("Transport audit record exceeds its bound")
        with self._db() as db:
            head = db.execute("SELECT sequence,digest FROM head WHERE singleton=1").fetchone()
            last = db.execute("SELECT sequence,digest FROM records ORDER BY sequence DESC LIMIT 1").fetchone()
            if head is None or (last or (0, "0" * 64)) != head or db.execute("SELECT count(*) FROM records").fetchone()[0] != head[0]:
                raise PeerAuditError("Transport audit head differs from its records")
            sequence, previous = head[0] + 1, head[1]
            if sequence > MAX_RECORDS:
                raise PeerAuditError("Transport audit allowance exhausted")
            digest = hashlib.sha256(bytes.fromhex(previous) + data).hexdigest()
            db.execute("INSERT INTO records VALUES (?,?,?,?)", (sequence, previous, digest, data.decode()))
            db.execute("UPDATE head SET sequence=?,digest=? WHERE singleton=1", (sequence, digest))
        return {"sequence": sequence, "digest": digest}

    def iter_records(self):
        """Verify and stream records; callers must exhaust the iterator."""
        with self._db() as db:
            rows = db.execute("SELECT sequence,previous,digest,body FROM records ORDER BY sequence")
            head = db.execute("SELECT sequence,digest FROM head WHERE singleton=1").fetchone()
            previous, count = "0" * 64, 0
            for expected, (sequence, ancestor, digest, body) in enumerate(rows, 1):
                count = expected
                if count > MAX_RECORDS or len(body.encode()) > MAX_RECORD_BYTES:
                    raise PeerAuditError("Transport audit exceeds its bound")
                if sequence != expected or ancestor != previous or hashlib.sha256(bytes.fromhex(previous) + body.encode()).hexdigest() != digest:
                    raise PeerAuditError("Transport audit chain is corrupt")
                yield {"sequence": sequence, "digest": digest, "record": json.loads(body)}
                previous = digest
            if head != (count, previous):
                raise PeerAuditError("Transport audit history is incomplete")

    def records(self):
        """Developer/export convenience; the running service uses streaming verification."""
        return list(self.iter_records())
