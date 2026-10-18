"""Bounded local source evidence captured by the running diagnostic application.

Exact validated observation bytes are retained in memory during a task and
committed with its terminal job in one transaction. A crash before that commit
leaves an interrupted capture, never a reviewable source. This journal neither
checks diagnostic correctness nor grants an experience endorsement.
"""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import time
import uuid

from .capsule import canonical
from .equipment import DEVICE_ID
from .experience import hash_value, token, unique_object
from .experience_store import read_private
from .source_inventory import MAX_COMMIT_BYTES, MAX_INVENTORY_BYTES


SCHEMA = "organism-source-journal-v2"
MAX_REQUESTS = 32
MAX_OBSERVATIONS = 16
MAX_OBSERVATION_BYTES = 65536
MAX_CAPTURE_BYTES = 512 * 1024
MAX_JOB_BYTES = 512 * 1024
MAX_DATABASE_PAGES = 32768
STATES = ("capturing", "awaiting_review", "ineligible", "interrupted")


class SourceJournalError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise SourceJournalError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def clock_value(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 4000000000


class Capture:
    def __init__(self, journal, request_id, device_id, started):
        self.journal, self.request_id, self.device_id = journal, request_id, device_id
        self.start_token, self.started = journal.start_token, started
        self.observations = {}
        self.bytes, self.reason, self.closed = 0, None, False

    def inventory(self, raw, receipt=b"", context=None):
        self.journal.save_inventory(self, raw, receipt, context)

    def observe(self, device_id, raw, raw_hash):
        require(not self.closed and device_id == self.device_id, "Observation belongs to another source capture")
        require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_OBSERVATION_BYTES
                and raw_hash == digest(raw), "Source observation bytes differ from their identity")
        if self.reason or raw_hash in self.observations:
            return
        if (len(self.observations) >= self.journal.max_observations
                or self.bytes + len(raw) > self.journal.max_capture_bytes):
            self.reason = "observation_capacity"
            return
        self.observations[raw_hash] = raw
        self.bytes += len(raw)


class SourceJournal:
    def __init__(self, directory, config, *, clock=time.time, max_requests=MAX_REQUESTS,
                 max_observations=MAX_OBSERVATIONS, max_capture_bytes=MAX_CAPTURE_BYTES):
        for value, maximum in ((max_requests, MAX_REQUESTS), (max_observations, MAX_OBSERVATIONS),
                               (max_capture_bytes, MAX_CAPTURE_BYTES)):
            require(type(value) is int and 1 <= value <= maximum, "Invalid source journal allowance")
        require(token(config.start_token) and token(config.individual_id)
                and all(hash_value(value) for value in (config.device_id, config.release_id,
                                                       config.model_sha256, config.executable_sha256))
                and type(config.model_bytes) is int and config.model_bytes > 0
                and type(config.native) is bool and type(config.trained_for_organism) is bool,
                "Invalid source runtime context")
        self.directory, self.clock = Path(directory).absolute(), clock
        self.start_token = config.start_token
        self.max_requests, self.max_observations = max_requests, max_observations
        self.max_capture_bytes = max_capture_bytes
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        info = self.directory.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & 0o077,
                "Source journal directory is not private")
        self.path, marker = self.directory / "sources.sqlite3", self.directory / "sources.enrolled"
        expected = {
            "schema": SCHEMA, "device_id": config.device_id, "individual_id": config.individual_id,
            "release_id": config.release_id, "model_sha256": config.model_sha256,
            "model_bytes": config.model_bytes, "executable_sha256": config.executable_sha256,
            "native": config.native, "trained_for_organism": config.trained_for_organism,
            "max_requests": max_requests, "max_observations": max_observations,
            "max_capture_bytes": max_capture_bytes,
        }
        present = os.path.lexists(self.path)
        require(present == os.path.lexists(marker), "Source journal or context is missing; do not reset it")
        if not present:
            context = {**expected, "journal_id": uuid.uuid4().hex}
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(canonical(context))
                stream.flush()
                os.fsync(stream.fileno())
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.close(fd)
        else:
            context = json.loads(read_private(marker, 4096), object_pairs_hook=unique_object)
            require(set(context) == set(expected) | {"journal_id"}
                    and all(context[key] == value for key, value in expected.items())
                    and token(context["journal_id"]), "Source journal runtime context differs")
        self.context = context
        with self.db() as db:
            if not present:
                db.execute("""CREATE TABLE requests (
                    request_id TEXT PRIMARY KEY, equipment_id TEXT NOT NULL, start_token TEXT NOT NULL,
                    started REAL NOT NULL, ended REAL, state TEXT NOT NULL, reason TEXT,
                    job BLOB, job_sha256 TEXT, source_sha256 TEXT)""")
                db.execute("""CREATE TABLE observations (
                    request_id TEXT NOT NULL REFERENCES requests(request_id), sha256 TEXT NOT NULL,
                    raw BLOB NOT NULL, PRIMARY KEY(request_id,sha256))""")
                db.execute("""CREATE TABLE inventories (
                    request_id TEXT PRIMARY KEY REFERENCES requests(request_id),
                    snapshot BLOB NOT NULL, receipt BLOB NOT NULL, context BLOB NOT NULL)""")
                db.execute("""CREATE TABLE head (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    requests INTEGER NOT NULL, observations INTEGER NOT NULL,
                    bytes INTEGER NOT NULL, declined INTEGER NOT NULL)""")
                db.execute("INSERT INTO head VALUES (1,0,0,0,0)")
            self.check_head(db)
            db.execute("""UPDATE requests SET state='interrupted', reason='previous_application_ended', ended=?
                          WHERE state='capturing' AND start_token != ?""", (self.now(), self.start_token))
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @contextmanager
    def db(self):
        info = self.path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1
                and not info.st_mode & 0o077, "Invalid source journal file")
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=3)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA synchronous=FULL")
            require(db.execute("PRAGMA page_size").fetchone()[0] == 4096
                    and db.execute("PRAGMA max_page_count=" + str(MAX_DATABASE_PAGES)).fetchone()[0]
                    == MAX_DATABASE_PAGES, "Source database exceeds its page allowance")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def now(self):
        value = self.clock()
        require(clock_value(value), "Invalid source journal clock")
        return value

    def check_head(self, db):
        head = db.execute("SELECT requests,observations,bytes,declined FROM head WHERE singleton=1").fetchone()
        count = db.execute("SELECT count(*) FROM requests").fetchone()[0]
        observations, raw_bytes = db.execute("SELECT count(*),coalesce(sum(length(raw)),0) FROM observations").fetchone()
        job_bytes = db.execute("SELECT coalesce(sum(length(job)),0) FROM requests").fetchone()[0]
        inventories = db.execute("SELECT request_id,snapshot,receipt,context FROM inventories").fetchall()
        inventory_bytes = sum(len(value) for row in inventories for value in row[1:])
        require(len(inventories) <= count and all(
            0 < len(raw) <= MAX_INVENTORY_BYTES and len(receipt) <= MAX_COMMIT_BYTES and len(context) <= 1024
            for _, raw, receipt, context in inventories), "Source inventory exceeds its allowance")
        require(head is not None and head[:3] == (count, observations, raw_bytes + job_bytes + inventory_bytes)
                and count <= self.max_requests and observations <= count * self.max_observations
                and raw_bytes <= count * self.max_capture_bytes and job_bytes <= count * MAX_JOB_BYTES
                and type(head[3]) is int and 0 <= head[3] <= 2**53,
                "Source history differs from its committed head or allowance")
        rows = db.execute("""SELECT request_id,equipment_id,start_token,started,ended,state,
                             reason,job_sha256,source_sha256,job IS NOT NULL FROM requests""").fetchall()
        require(all(token(request) and isinstance(device, str) and DEVICE_ID.fullmatch(device)
                    and token(start) and clock_value(began) and (ended is None or clock_value(ended))
                    and state in STATES
                    and ((state == "capturing" and ended is None and reason is None and not has_job)
                         or (state != "capturing" and ended is not None
                             and ((state == "awaiting_review" and reason is None and has_job
                                   and hash_value(source_hash))
                                  or (state != "awaiting_review" and isinstance(reason, str)
                                      and 1 <= len(reason) <= 128))))
                    and (hash_value(job_hash) if has_job else job_hash is None)
                    and (source_hash is None or hash_value(source_hash))
                    for request, device, start, began, ended, state, reason, job_hash, source_hash, has_job in rows),
                "Invalid source request history")
        sizes = db.execute("""SELECT r.request_id, count(o.sha256),coalesce(sum(length(o.raw)),0),
                              coalesce(max(length(o.raw)),0),coalesce(length(r.job),0)
                              FROM requests r LEFT JOIN observations o ON r.request_id=o.request_id
                              GROUP BY r.request_id""").fetchall()
        require(all(count <= self.max_observations and total <= self.max_capture_bytes
                    and largest <= MAX_OBSERVATION_BYTES and job <= MAX_JOB_BYTES
                    for _, count, total, largest, job in sizes), "Source request exceeds its individual allowance")

    def save_inventory(self, capture, raw, receipt, context):
        require(isinstance(capture, Capture) and capture.journal is self and not capture.closed
                and isinstance(raw, bytes) and 0 < len(raw) <= MAX_INVENTORY_BYTES
                and isinstance(receipt, bytes) and len(receipt) <= MAX_COMMIT_BYTES,
                "Invalid captured procedure inventory")
        body = json.loads(raw, object_pairs_hook=unique_object)["body"]
        require(body["producer_device_id"] == self.context["device_id"]
                and body["producer_individual_id"] == self.context["individual_id"]
                and body["release_id"] == self.context["release_id"]
                and body["request_id"] == capture.request_id,
                "Captured inventory belongs to another source")
        encoded = canonical(context) if context is not None else b""
        require(len(encoded) <= 1024 and bool(receipt) == bool(encoded),
                "Captured commitment or model context is incomplete")
        if context is not None:
            require(context["inventory_sha256"] == digest(raw) and context["commit_sha256"] == digest(receipt)
                    and clock_value(context["model_started_unix_seconds"])
                    and capture.started <= context["model_started_unix_seconds"] <= self.now(),
                    "Captured inventory context differs from its source")
        with self.db() as db:
            self.check_head(db)
            state = db.execute("SELECT state,start_token FROM requests WHERE request_id=?",
                               (capture.request_id,)).fetchone()
            require(state == ("capturing", capture.start_token), "Source capture is no longer active")
            old = db.execute("SELECT snapshot,receipt,context FROM inventories WHERE request_id=?",
                             (capture.request_id,)).fetchone()
            require(old is None or (bytes(old[0]) == raw and old[1:] == (b"", b"")),
                    "Source inventory or its commitment cannot be replaced")
            old_bytes = sum(map(len, old)) if old else 0
            db.execute("INSERT OR REPLACE INTO inventories VALUES (?,?,?,?)",
                       (capture.request_id, raw, receipt, encoded))
            db.execute("UPDATE head SET bytes=bytes+? WHERE singleton=1",
                       (len(raw) + len(receipt) + len(encoded) - old_bytes,))
            self.check_head(db)

    def inventory_for(self, request_id):
        with self.db() as db:
            self.check_head(db)
            row = db.execute("SELECT snapshot,receipt,context FROM inventories WHERE request_id=?",
                             (request_id,)).fetchone()
        if row is None:
            return None
        raw, receipt, context = map(bytes, row)
        decoded = json.loads(context, object_pairs_hook=unique_object) if context else None
        if decoded is not None:
            require(decoded["inventory_sha256"] == digest(raw) and decoded["commit_sha256"] == digest(receipt),
                    "Retained inventory commitment is corrupt")
        return {"snapshot": raw, "receipt": receipt, "context": decoded}

    def begin(self, request_id, device_id):
        require(token(request_id) and isinstance(device_id, str) and DEVICE_ID.fullmatch(device_id),
                "Invalid source request")
        started = self.now()
        with self.db() as db:
            self.check_head(db)
            row = db.execute("SELECT equipment_id FROM requests WHERE request_id=?", (request_id,)).fetchone()
            if row is not None:
                require(row[0] == device_id, "Source request belongs to another equipment identity")
                return None
            count, declined = db.execute("SELECT requests,declined FROM head WHERE singleton=1").fetchone()
            if count >= self.max_requests:
                require(declined < 2**53, "Source decline counter exhausted")
                db.execute("UPDATE head SET declined=declined+1 WHERE singleton=1")
                return None
            db.execute("INSERT INTO requests VALUES (?,?,?,?,NULL,'capturing',NULL,NULL,NULL,NULL)",
                       (request_id, device_id, self.start_token, started))
            db.execute("UPDATE head SET requests=requests+1 WHERE singleton=1")
        return Capture(self, request_id, device_id, started)

    def finish(self, capture, job):
        require(isinstance(capture, Capture) and capture.journal is self and not capture.closed
                and capture.start_token == self.start_token,
                "Source capture belongs to another application or is closed")
        require(len(capture.observations) <= self.max_observations
                and capture.bytes == sum(len(raw) for raw in capture.observations.values())
                and capture.bytes <= self.max_capture_bytes
                and all(isinstance(raw, bytes) and 0 < len(raw) <= MAX_OBSERVATION_BYTES
                        and hash_value(identity) and digest(raw) == identity
                        for identity, raw in capture.observations.items()), "Captured observation allowance or identity differs")
        require(isinstance(job, dict) and set(job) == {"record", "digest"}, "Source has no diagnostic job")
        record = job["record"]
        require(isinstance(record, dict), "Invalid source job record")
        raw_job = canonical(record)
        require(len(raw_job) <= MAX_JOB_BYTES and digest(raw_job) == job["digest"]
                and record.get("request_id") == capture.request_id and record.get("device_id") == capture.device_id
                and record.get("task_kind") == "equipment_interface_v1", "Source job identity differs")
        ended = self.now()
        trace = record.get("trace", {})
        require(isinstance(trace, dict) and isinstance(trace.get("tool_events", []), list)
                and all(isinstance(event, dict) and isinstance(event.get("result", {}), dict)
                        for event in trace.get("tool_events", [])), "Invalid source tool trace")
        reads = [event["result"] for event in trace.get("tool_events", [])
                 if event.get("tool") == "read_equipment" and event.get("result", {}).get("status") == "observed"]
        hashes = [reading.get("observation_sha256") for reading in reads]
        source_hash = hashes[-1] if hashes and hash_value(hashes[-1]) else None
        inventory = self.inventory_for(capture.request_id)
        committed = inventory is not None and inventory["context"] is not None
        reason = capture.reason
        if reason is None:
            if record.get("status") != "finished":
                reason = "diagnostic_incomplete"
            elif not reads:
                reason = "no_executed_reading"
            elif any(not hash_value(value) or value not in capture.observations for value in hashes):
                reason = "observation_not_captured"
            elif len(set(hashes)) != 1:
                reason = "multiple_reading_versions"
            elif not trace.get("model_calls"):
                reason = "no_model_trace"
            elif (record.get("submission") or {}).get("status") not in ("normal", "low", "high"):
                reason = "inconclusive_diagnosis"
            elif trace.get("source_inventory") != (inventory["context"] if committed else None):
                reason = "source_inventory_not_captured"
            elif (any(event.get("tool") == "prepare_reader_from_experience" for event in trace.get("tool_events", []))
                  and not committed):
                reason = "inherited_source_inventory_required"
            elif not all(clock_value(reading.get("read_unix_seconds"))
                         and capture.started <= reading["read_unix_seconds"] <= ended for reading in reads):
                reason = "source_clock_outside_capture"
        state = "awaiting_review" if reason is None else "ineligible"
        if record.get("status") in ("running", "interrupted"):
            state, reason = "interrupted", "diagnostic_interrupted"
        with self.db() as db:
            self.check_head(db)
            row = db.execute("SELECT state,start_token,equipment_id,started FROM requests WHERE request_id=?",
                             (capture.request_id,)).fetchone()
            require(row == ("capturing", self.start_token, capture.device_id, capture.started),
                    "Source capture is no longer active or its identity differs")
            for identity, raw in capture.observations.items():
                require(digest(raw) == identity, "Captured observation changed before commit")
                db.execute("INSERT INTO observations VALUES (?,?,?)", (capture.request_id, identity, raw))
            db.execute("""UPDATE requests SET ended=?,state=?,reason=?,job=?,job_sha256=?,source_sha256=?
                          WHERE request_id=?""",
                       (ended, state, reason, raw_job, job["digest"], source_hash, capture.request_id))
            db.execute("UPDATE head SET observations=observations+?,bytes=bytes+? WHERE singleton=1",
                       (len(capture.observations), capture.bytes + len(raw_job)))
            self.check_head(db)
        capture.closed = True
        return self.get(capture.request_id)

    def get(self, request_id):
        require(token(request_id), "Invalid source request identity")
        with self.db() as db:
            self.check_head(db)
            row = db.execute("""SELECT equipment_id,start_token,started,ended,state,reason,job,job_sha256,source_sha256
                                FROM requests WHERE request_id=?""", (request_id,)).fetchone()
            if row is None:
                return None
            observations = {identity: bytes(raw) for identity, raw in db.execute(
                "SELECT sha256,raw FROM observations WHERE request_id=?", (request_id,))}
        require(all(hash_value(identity) and digest(raw) == identity for identity, raw in observations.items()),
                "Retained source observation is corrupt")
        raw_job = bytes(row[6]) if row[6] is not None else None
        require(raw_job is None or digest(raw_job) == row[7], "Retained source job is corrupt")
        record = json.loads(raw_job, object_pairs_hook=unique_object) if raw_job is not None else None
        require(record is None or (isinstance(record, dict) and record.get("request_id") == request_id
                and record.get("device_id") == row[0] and record.get("task_kind") == "equipment_interface_v1"),
                "Retained source job identity differs")
        require(row[4] != "awaiting_review" or row[8] in observations,
                "Reviewable source has no retained observation")
        return {"request_id": request_id, "equipment_id": row[0], "start_token": row[1],
                "started": row[2], "ended": row[3], "state": row[4], "reason": row[5],
                "job": {"record": record, "digest": row[7]} if raw_job is not None else None,
                "source_inventory": self.inventory_for(request_id),
                "source_sha256": row[8], "observations": observations,
                "runtime_context": dict(self.context), "independently_verified": False}

    def snapshot(self):
        with self.db() as db:
            self.check_head(db)
            head = db.execute("SELECT requests,observations,bytes,declined FROM head WHERE singleton=1").fetchone()
            rows = db.execute("SELECT request_id,equipment_id,state,reason FROM requests ORDER BY rowid").fetchall()
        return {"journal_id": self.context["journal_id"], "retained_requests": head[0],
                "retained_observations": head[1], "retained_bytes": head[2], "capacity_declines": head[3],
                "requests": [{"request_id": request, "equipment_id": device, "state": state, "reason": reason}
                             for request, device, state, reason in rows],
                "sources_independently_verified": False, "qualification_passed": False}
