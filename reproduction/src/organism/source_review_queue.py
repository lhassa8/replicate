"""Bounded producer queue for automatically reviewing captured source evidence.

Selection and approvals are durable. Network retries use the exchange's existing
attempt ledger. A saved approval is installed idempotently without another peer
request after a crash between the two local database commits.
"""

from contextlib import contextmanager
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import stat
import uuid

from .capsule import canonical
from .experience import hash_value, require, token, unique_object
from .experience_proposal import (
    MAX_APPROVAL_BYTES, MAX_PROPOSAL_BYTES, MAX_SOURCE_AGE_SECONDS,
    accept, check_approval, check_proposal, decode, sha, source_facts,
)
from .experience_store import read_private
from .peer_audit import PeerAuditError


SCHEMA = "organism-source-review-queue-v1"
MAX_REQUESTS = 32
MAX_REVIEWS_PER_ROUND = 1
STATES = ("pending", "approved_pending_install", "installed", "rejected", "exhausted", "expired",
          "approval_expired")
RUNTIME_FIELDS = ("device_id", "individual_id", "release_id", "model_sha256", "model_bytes",
                  "executable_sha256", "native", "trained_for_organism")


class SourceReviewQueue:
    def __init__(self, directory, exchange, journal, *, max_requests=MAX_REQUESTS):
        require(type(max_requests) is int and 1 <= max_requests <= MAX_REQUESTS
                and all(journal.context[field] == getattr(exchange.config, field) for field in RUNTIME_FIELDS),
                "Source-review queue belongs to another runtime or exceeds its allowance")
        self.exchange, self.journal, self.max_requests = exchange, journal, max_requests
        self.directory = Path(directory).absolute()
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        info = self.directory.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & 0o077,
                "Source-review queue must be private")
        self.path, marker = self.directory / "queue.sqlite3", self.directory / "queue.enrolled"
        expected = {"schema": SCHEMA, "journal_id": journal.context["journal_id"],
                    "store_id": exchange.store.store_id, "book_id": exchange.book.book_id,
                    "policy_id": exchange.policy.policy_id, "max_requests": max_requests,
                    **{field: journal.context[field] for field in RUNTIME_FIELDS}}
        present = os.path.lexists(self.path)
        require(present == os.path.lexists(marker), "Source-review queue or enrollment is missing")
        if present:
            context = json.loads(read_private(marker, 4096), object_pairs_hook=unique_object)
            require(set(context) == set(expected) | {"queue_id"}
                    and all(context[key] == value for key, value in expected.items())
                    and token(context["queue_id"]), "Source-review queue context differs")
        else:
            context = {**expected, "queue_id": uuid.uuid4().hex}
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as output:
                output.write(canonical(context))
                output.flush()
                os.fsync(output.fileno())
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.close(fd)
        self.context, self.skipped = context, {}
        with self.db() as db:
            if not present:
                db.execute("""CREATE TABLE requests (
                    request TEXT PRIMARY KEY, peer TEXT NOT NULL, proposal BLOB NOT NULL,
                    state TEXT NOT NULL, approval BLOB NOT NULL, note_id TEXT)""")
                db.execute("""CREATE TABLE head (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1), requests INTEGER NOT NULL,
                    bytes INTEGER NOT NULL)""")
                db.execute("INSERT INTO head VALUES (1,0,0)")
            self.check_head(db)
        self.records()
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @contextmanager
    def db(self):
        info = self.path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
                and info.st_nlink == 1 and not info.st_mode & 0o077, "Invalid source-review queue database")
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=3)
        try:
            db.execute("PRAGMA synchronous=FULL")
            require(db.execute("PRAGMA page_size").fetchone()[0] == 4096
                    and db.execute("PRAGMA max_page_count=4096").fetchone()[0] == 4096,
                    "Source-review queue exceeds its database allowance")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def check_head(self, db):
        head = db.execute("SELECT requests,bytes FROM head WHERE singleton=1").fetchone()
        actual = db.execute("SELECT count(*),coalesce(sum(length(proposal)+length(approval)),0) FROM requests").fetchone()
        require(head == actual and 0 <= head[0] <= self.max_requests
                and 0 <= head[1] <= self.max_requests * (MAX_PROPOSAL_BYTES + MAX_APPROVAL_BYTES),
                "Source-review queue differs from its committed head")

    def validate_row(self, row):
        request, peer, proposal, state, approval, note_id = row
        require(token(request) and peer in self.exchange.identities and state in STATES
                and isinstance(proposal, bytes) and isinstance(approval, bytes),
                "Invalid retained source-review request")
        body = check_proposal(proposal, self.exchange.policy, release_id=self.exchange.config.release_id)
        source = self.journal.get(request)
        committed_peer = self.exchange.capture_inventory_peer(source) if source is not None else None
        require(committed_peer is None or committed_peer == peer, "Review changed its inventory verifier")
        require(source is not None and source["state"] == "awaiting_review"
                and body["request_id"] == request and body["verifier_device_id"] == peer
                and body["producer_device_id"] == self.exchange.device_id
                and body["producer_individual_id"] == self.exchange.config.individual_id
                and body["source_kind"] == "live_model"
                and body["clock_window"] == [source["started"], source["ended"]],
                "Retained review differs from its source capture")
        observation = source["observations"][source["source_sha256"]]
        facts = source_facts(observation, source["job"])
        require(all(body[name] == value for name, value in facts.items()),
                "Retained review source identities changed")
        if state in ("approved_pending_install", "installed", "approval_expired"):
            envelope = decode(approval, MAX_APPROVAL_BYTES)
            created = envelope.get("body", {}).get("created_unix_seconds")
            approved, _ = check_approval(proposal, approval, policy=self.exchange.policy,
                release_id=self.exchange.config.release_id, allowed_checkers=self.exchange.store.allowed_checkers,
                now=created, require_live_source=True)
            require(note_id == sha(canonical(approved)), "Retained approval differs from its procedure")
        else:
            require(approval == b"" and note_id is None, "Unapproved source contains an endorsement")
        return {"request": request, "peer": peer, "proposal": proposal, "state": state,
                "approval": approval, "note_id": note_id, "body": body,
                "observation": observation, "job": source["job"]}

    def records(self):
        with self.db() as db:
            self.check_head(db)
            rows = db.execute("SELECT * FROM requests ORDER BY rowid").fetchall()
        return [self.validate_row(row) for row in rows]

    def enqueue(self, source, peer):
        proposal = self.exchange.capture_proposal(peer, source)
        row = (source["request_id"], peer, proposal, "pending", b"", None)
        self.validate_row(row)
        with self.db() as db:
            self.check_head(db)
            require(db.execute("SELECT requests FROM head WHERE singleton=1").fetchone()[0] < self.max_requests,
                    "Source-review queue capacity exhausted")
            db.execute("INSERT INTO requests VALUES (?,?,?,?,?,?)", row)
            db.execute("UPDATE head SET requests=requests+1,bytes=bytes+? WHERE singleton=1", (len(proposal),))
            self.check_head(db)
        return self.validate_row(row)

    def transition(self, record, state, approval=b"", note_id=None):
        allowed = {"pending": {"approved_pending_install", "rejected", "exhausted", "expired"},
                   "approved_pending_install": {"installed", "approval_expired"}}
        require(state in allowed.get(record["state"], ()), "Invalid source-review state transition")
        row = (record["request"], record["peer"], record["proposal"], state, approval, note_id)
        validated = self.validate_row(row)
        with self.db() as db:
            self.check_head(db)
            prior = db.execute("SELECT * FROM requests WHERE request=?", (record["request"],)).fetchone()
            require(prior == tuple(record[name] for name in ("request", "peer", "proposal", "state", "approval", "note_id")),
                    "Source-review state changed before commit")
            db.execute("UPDATE requests SET state=?,approval=?,note_id=? WHERE request=?",
                       (state, approval, note_id, record["request"]))
            db.execute("UPDATE head SET bytes=bytes+? WHERE singleton=1", (len(approval) - len(record["approval"]),))
            self.check_head(db)
        return validated

    def install(self, record):
        envelope = decode(record["approval"], MAX_APPROVAL_BYTES)
        if self.exchange.store.now() >= envelope["body"]["expires_unix_seconds"]:
            return self.transition(record, "approval_expired", record["approval"], record["note_id"])
        raw = accept(record["proposal"], record["approval"], policy=self.exchange.policy,
                     producer_key=self.exchange.store.device_key, release_id=self.exchange.config.release_id,
                     allowed_checkers=self.exchange.store.allowed_checkers, now=self.exchange.store.now(),
                     require_live_source=True)
        require(self.exchange.store.add(raw) == record["note_id"], "Installed procedure differs from its approval")
        return self.transition(record, "installed", record["approval"], record["note_id"])

    async def round(self, reviewers):
        require(set(reviewers) <= set(self.exchange.identities), "Review queue received an undeclared peer")
        rows = self.records()
        known = {row["request"] for row in rows}
        self.skipped = {}
        for item in self.journal.snapshot()["requests"]:
            request = item["request_id"]
            if item["state"] != "awaiting_review" or request in known or len(rows) >= self.max_requests:
                continue
            if not reviewers:
                continue
            source = self.journal.get(request)
            # A source is assigned once. Rejection does not cause reviewer shopping.
            peer = self.exchange.capture_inventory_peer(source)
            if peer is None:
                peer = sorted(reviewers)[int(sha(request.encode()), 16) % len(reviewers)]
            elif peer not in reviewers:
                continue
            if ((peer, request) not in self.exchange.review_attempts
                    and len(self.exchange.review_attempts) >= self.exchange.review_request_limit):
                self.skipped[request] = "attempt_capacity"
                continue
            try:
                proposal = self.exchange.capture_proposal(peer, source)
                body = check_proposal(proposal, self.exchange.policy, release_id=self.exchange.config.release_id)
            except ValueError:
                self.skipped[request] = "source_not_proposable"
                continue
            if not body["source_clock"] < self.exchange.store.now() + 1 or (
                    self.exchange.store.now() - body["source_clock"] > MAX_SOURCE_AGE_SECONDS):
                self.skipped[request] = "source_outside_review_window"
                continue
            rows.append(self.enqueue(source, peer))
            known.add(request)
        attempts = 0
        for row in rows:
            if self.exchange.closed:
                return
            if row["state"] == "approved_pending_install":
                self.install(row)
                continue
            if row["state"] != "pending":
                continue
            if self.exchange.store.now() - row["body"]["source_clock"] > MAX_SOURCE_AGE_SECONDS:
                self.transition(row, "expired")
                continue
            count = self.exchange.review_attempts.get((row["peer"], row["request"]), (None, 0))[1]
            if count >= self.exchange.review_attempt_limit:
                self.transition(row, "exhausted")
                continue
            if attempts >= MAX_REVIEWS_PER_ROUND or row["peer"] not in reviewers:
                continue
            attempts += 1
            try:
                result = await self.exchange.request_review(row["peer"], row["proposal"], row["observation"], row["job"])
            except PeerAuditError:
                raise
            except (ValueError, OSError, EOFError, asyncio.IncompleteReadError):
                continue
            if result["status"] == "rejected":
                self.transition(row, "rejected")
            else:
                approved, _ = check_approval(row["proposal"], result["approval"], policy=self.exchange.policy,
                    release_id=self.exchange.config.release_id, allowed_checkers=self.exchange.store.allowed_checkers,
                    now=self.exchange.store.now(), require_live_source=True)
                saved = self.transition(row, "approved_pending_install", result["approval"], sha(canonical(approved)))
                self.install(saved)

    def snapshot(self):
        rows = self.records()
        return {"queue_id": self.context["queue_id"], "capacity": self.max_requests,
                "requests": [{"request_id": row["request"], "peer_id": row["peer"], "state": row["state"],
                              "note_id": row["note_id"]} for row in rows],
                "states": {state: sum(row["state"] == state for row in rows) for state in STATES},
                "skipped_current_round": dict(self.skipped), "qualification_passed": False}
