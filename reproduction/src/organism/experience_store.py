"""Bounded durable procedure copies and signed copy acknowledgments.

No network or model is started here. A transport must authenticate callers and
carry these bounded records. Historical acknowledgments do not establish that
another peer is currently reachable or still retains its copy.
"""

import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import time
import uuid

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .capsule import canonical
from .enrollment import key_id, public_bytes
from .experience import (
    exact, ExperienceError, hash_value, integer, require, token, unique_object, VerifiedExperience,
)


MAX_RECORDS = 128
MAX_RECEIPTS = 256
MAX_RECEIPT_BYTES = 4096
ACK_SCHEMA = "organism-experience-copy-v1"
ACK_PREFIX = b"organism-experience-copy-v1\0"


def read_private(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1
                and not info.st_mode & 0o077 and 0 < info.st_size <= limit, "Invalid private experience file")
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "Experience file exceeds its bound")
    return raw


def verify_copy_receipt(raw, policy, *, release_id, note_id, sender_id, receiver_id, request_id,
                        now, maximum_age_seconds=120):
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_RECEIPT_BYTES, "Copy receipt exceeds its bound")
    require(integer(now) and type(maximum_age_seconds) is int and 1 <= maximum_age_seconds <= 3600,
            "Invalid copy acknowledgment clock or age bound")
    require(all(hash_value(value) for value in (release_id, note_id, sender_id, receiver_id))
            and token(request_id) and sender_id != receiver_id, "Invalid copy acknowledgment context")
    try:
        envelope = json.loads(raw, object_pairs_hook=unique_object)
        exact(envelope, ("body", "receipt_id", "signature"))
        body = envelope["body"]
        exact(body, ("schema", "policy_id", "release_id", "note_id", "sender_id", "receiver_id",
                     "request_id", "store_id", "commit_sequence", "committed_unix_seconds", "durably_retained"))
        require(body["schema"] == ACK_SCHEMA and body["policy_id"] == policy.policy_id
                and body["release_id"] == release_id and body["note_id"] == note_id
                and body["sender_id"] == sender_id and body["receiver_id"] == receiver_id
                and body["request_id"] == request_id, "Copy receipt differs from its requested transfer")
        require(token(body["store_id"]) and type(body["commit_sequence"]) is int
                and 1 <= body["commit_sequence"] <= MAX_RECEIPTS and body["durably_retained"] is True,
                "Copy receipt does not acknowledge a bounded committed copy")
        require(integer(body["committed_unix_seconds"])
                and 0 <= now - body["committed_unix_seconds"] <= maximum_age_seconds,
                "Copy receipt is stale or from a future clock")
        payload = canonical(body)
        require(envelope["receipt_id"] == hashlib.sha256(payload).hexdigest(), "Copy receipt identity differs")
        policy.peer_key(sender_id)
        Ed25519PublicKey.from_public_bytes(policy.peer_key(receiver_id)).verify(
            base64.b64decode(envelope["signature"], validate=True), ACK_PREFIX + payload)
    except ExperienceError:
        raise
    except (InvalidSignature, ValueError, TypeError, KeyError, RecursionError, UnicodeError) as exc:
        raise ExperienceError("Invalid signed copy receipt") from exc
    return body


class ExperienceStore:
    def __init__(self, directory, policy, device_key, *, release_id, allowed_checkers,
                 require_live_source=True, clock=time.time, max_records=MAX_RECORDS, max_receipts=MAX_RECEIPTS):
        require(type(max_records) is int and 1 <= max_records <= MAX_RECORDS
                and type(max_receipts) is int and 1 <= max_receipts <= MAX_RECEIPTS, "Invalid experience store limits")
        require(hash_value(release_id) and isinstance(allowed_checkers, (set, frozenset))
                and 1 <= len(allowed_checkers) <= 4 and all(hash_value(value) for value in allowed_checkers)
                and type(require_live_source) is bool, "Invalid trusted experience configuration")
        self.directory = Path(directory).absolute()
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        info = self.directory.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & 0o077,
                "Experience directory must be private to its service owner")
        self.policy, self.device_key, self.clock = policy, device_key, clock
        self.device_id = key_id(public_bytes(device_key.public_key()))
        policy.peer_key(self.device_id)
        self.release_id, self.allowed_checkers = release_id, frozenset(allowed_checkers)
        self.require_live_source = require_live_source
        self.max_records, self.max_receipts = max_records, max_receipts
        self.path = self.directory / "experience.sqlite3"
        marker = self.directory / "experience.enrolled"
        database_present, marker_present = os.path.lexists(self.path), os.path.lexists(marker)
        require(database_present == marker_present, "Experience history or enrollment marker is missing; do not reset it")
        expected = {"schema": "organism-experience-store-v1", "policy_id": policy.policy_id,
                    "device_id": self.device_id, "release_id": release_id, "allowed_checkers": sorted(allowed_checkers),
                    "require_live_source": require_live_source, "max_records": max_records, "max_receipts": max_receipts}
        if not database_present:
            marker_body = {**expected, "store_id": uuid.uuid4().hex}
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(canonical(marker_body)); stream.flush(); os.fsync(stream.fileno())
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            os.close(fd)
        else:
            marker_body = json.loads(read_private(marker, 4096), object_pairs_hook=unique_object)
            require(set(marker_body) == set(expected) | {"store_id"}
                    and all(marker_body[key] == value for key, value in expected.items())
                    and token(marker_body["store_id"]), "Experience store belongs to another context")
        self.store_id = marker_body["store_id"]
        with self.db() as db:
            if not database_present:
                db.execute("CREATE TABLE notes (note_id TEXT PRIMARY KEY, raw BLOB NOT NULL)")
                db.execute("""CREATE TABLE issued (
                    sender TEXT NOT NULL, request TEXT NOT NULL, note_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL UNIQUE, raw BLOB NOT NULL, PRIMARY KEY(sender,request))""")
                db.execute("""CREATE TABLE confirmed (
                    note_id TEXT NOT NULL, receiver TEXT NOT NULL, request TEXT NOT NULL,
                    raw BLOB NOT NULL, PRIMARY KEY(note_id,receiver,request))""")
                db.execute("""CREATE TABLE head (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    notes INTEGER NOT NULL, issued INTEGER NOT NULL, confirmed INTEGER NOT NULL)""")
                db.execute("INSERT INTO head VALUES (1,0,0,0)")
            self.check_head(db)
        directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @contextmanager
    def db(self):
        info = self.path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1
                and not info.st_mode & 0o077, "Invalid experience database")
        database = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=5)
        try:
            database.execute("PRAGMA synchronous=FULL")
            database.execute("PRAGMA max_page_count=1024")
            database.execute("BEGIN IMMEDIATE")
            yield database
            database.commit()
        except BaseException:
            database.rollback()
            raise
        finally:
            database.close()

    def now(self):
        value = self.clock()
        require(type(value) in (int, float) and 0 <= value <= 4000000000, "Invalid experience store clock")
        return int(value)

    def check_head(self, db):
        head = db.execute("SELECT notes,issued,confirmed FROM head WHERE singleton=1").fetchone()
        counts = tuple(db.execute("SELECT count(*) FROM " + table).fetchone()[0]
                       for table in ("notes", "issued", "confirmed"))
        maximum = db.execute("SELECT max(sequence) FROM issued").fetchone()[0] or 0
        require(head == counts and maximum == counts[1], "Experience history differs from its committed head")
        require(counts[0] <= self.max_records and counts[1] <= self.max_receipts and counts[2] <= self.max_receipts,
                "Experience store allowance exceeded")

    def verified(self, raw):
        return VerifiedExperience(raw, self.policy, release_id=self.release_id, allowed_checkers=self.allowed_checkers,
                                  now=self.now(), require_live_source=self.require_live_source)

    def put(self, db, record):
        existing = db.execute("SELECT raw FROM notes WHERE note_id=?", (record.note_id,)).fetchone()
        if existing:
            require(bytes(existing[0]) == record.raw, "Stored experience bytes differ from their identity")
        else:
            require(db.execute("SELECT count(*) FROM notes").fetchone()[0] < self.max_records,
                    "Experience record allowance exhausted")
            db.execute("INSERT INTO notes VALUES (?,?)", (record.note_id, record.raw))
            db.execute("UPDATE head SET notes=notes+1 WHERE singleton=1")

    def add(self, raw):
        record = self.verified(raw)
        with self.db() as db:
            self.check_head(db)
            self.put(db, record)
        return record.note_id

    def get(self, note_id):
        require(hash_value(note_id), "Invalid experience lookup identity")
        with self.db() as db:
            self.check_head(db)
            row = db.execute("SELECT raw FROM notes WHERE note_id=?", (note_id,)).fetchone()
        require(row is not None, "Experience record is not retained")
        record = self.verified(bytes(row[0]))
        require(record.note_id == note_id, "Stored experience identity changed")
        return record

    def inventory(self):
        """Return currently usable IDs, retaining and verifying expired history."""
        now, result = self.now(), []
        with self.db() as db:
            self.check_head(db)
            rows = db.execute("SELECT note_id,raw FROM notes ORDER BY note_id").fetchall()
        for note_id, raw in rows:
            try:
                created = json.loads(bytes(raw), object_pairs_hook=unique_object)["body"]["created_unix_seconds"]
                record = VerifiedExperience(bytes(raw), self.policy, release_id=self.release_id,
                    allowed_checkers=self.allowed_checkers, now=created, require_live_source=self.require_live_source)
            except (ValueError, KeyError, TypeError, RecursionError) as exc:
                raise ExperienceError("Invalid retained experience history") from exc
            require(record.note_id == note_id, "Stored experience identity changed")
            if record.body["created_unix_seconds"] <= now < record.body["expires_unix_seconds"]:
                result.append(note_id)
        return result

    def receive(self, raw, *, sender_id, request_id):
        """Return a signed acknowledgment only after the record and receipt commit."""
        require(hash_value(sender_id) and sender_id != self.device_id and token(request_id),
                "Invalid experience transfer identity")
        self.policy.peer_key(sender_id)
        record = self.verified(raw)
        with self.db() as db:
            self.check_head(db)
            existing = db.execute("SELECT note_id,raw FROM issued WHERE sender=? AND request=?",
                                  (sender_id, request_id)).fetchone()
            if existing:
                require(existing[0] == record.note_id, "Transfer request is already bound to different experience")
                retained = db.execute("SELECT raw FROM notes WHERE note_id=?", (record.note_id,)).fetchone()
                require(retained is not None and bytes(retained[0]) == record.raw,
                        "Cannot repeat an acknowledgment after losing its retained copy")
                acknowledgement = bytes(existing[1])
                verify_copy_receipt(acknowledgement, self.policy, release_id=self.release_id, note_id=record.note_id,
                                    sender_id=sender_id, receiver_id=self.device_id, request_id=request_id,
                                    now=self.now(), maximum_age_seconds=3600)
            else:
                count = db.execute("SELECT count(*) FROM issued").fetchone()[0]
                maximum = db.execute("SELECT max(sequence) FROM issued").fetchone()[0] or 0
                require(count == maximum and count < self.max_receipts,
                        "Copy acknowledgment history is incomplete or exhausted")
                self.put(db, record)
                body = {"schema": ACK_SCHEMA, "policy_id": self.policy.policy_id, "release_id": self.release_id,
                        "note_id": record.note_id, "sender_id": sender_id, "receiver_id": self.device_id,
                        "request_id": request_id, "store_id": self.store_id, "commit_sequence": count + 1,
                        "committed_unix_seconds": self.now(), "durably_retained": True}
                payload = canonical(body)
                acknowledgement = canonical({"body": body, "receipt_id": hashlib.sha256(payload).hexdigest(),
                    "signature": base64.b64encode(self.device_key.sign(ACK_PREFIX + payload)).decode()})
                require(len(acknowledgement) <= MAX_RECEIPT_BYTES, "Copy receipt exceeds its bound")
                db.execute("INSERT INTO issued VALUES (?,?,?,?,?)",
                           (sender_id, request_id, record.note_id, count + 1, acknowledgement))
                db.execute("UPDATE head SET issued=issued+1 WHERE singleton=1")
        return acknowledgement

    def confirm_copy(self, note_id, *, receiver_id, request_id, receipt):
        self.get(note_id)
        verify_copy_receipt(receipt, self.policy, release_id=self.release_id, note_id=note_id,
                            sender_id=self.device_id, receiver_id=receiver_id, request_id=request_id, now=self.now())
        with self.db() as db:
            self.check_head(db)
            existing = db.execute("SELECT raw FROM confirmed WHERE note_id=? AND receiver=? AND request=?",
                                  (note_id, receiver_id, request_id)).fetchone()
            if existing:
                require(bytes(existing[0]) == receipt, "Confirmed copy acknowledgment changed")
            else:
                require(db.execute("SELECT count(*) FROM confirmed").fetchone()[0] < self.max_receipts,
                        "Confirmed-copy allowance exhausted")
                db.execute("INSERT INTO confirmed VALUES (?,?,?,?)", (note_id, receiver_id, request_id, receipt))
                db.execute("UPDATE head SET confirmed=confirmed+1 WHERE singleton=1")
        return self.copy_state(note_id)

    def copy_state(self, note_id):
        self.get(note_id)
        with self.db() as db:
            self.check_head(db)
            rows = db.execute("SELECT receiver,request,raw FROM confirmed WHERE note_id=? ORDER BY receiver,request",
                              (note_id,)).fetchall()
        receivers = set()
        for receiver, request, raw in rows:
            # Check the original acknowledgment time for historical accounting.
            envelope = json.loads(bytes(raw), object_pairs_hook=unique_object)
            then = envelope["body"]["committed_unix_seconds"]
            verify_copy_receipt(bytes(raw), self.policy, release_id=self.release_id, note_id=note_id,
                                sender_id=self.device_id, receiver_id=receiver, request_id=request, now=then)
            receivers.add(receiver)
        return {"note_id": note_id, "local_record_retained": True,
                "historically_acknowledged_other_devices": sorted(receivers),
                "historical_acknowledged_device_copies": 1 + len(receivers),
                "current_remote_liveness_verified": False, "independent_physical_hosts_verified": False}
