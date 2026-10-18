"""Durable, bounded decisions for authenticated source-review requests.

The caller supplies an independently provisioned local reviewer. No review
procedure or private producer key is accepted over the transport. A committed
decision is immutable and is returned again after a lost response or restart.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import stat
import time
import uuid

from .capsule import canonical
from .equipment_interfaces import bounded_tree, validate_document
from .experience import exact, hash_value, integer, require, token, unique_object
from .experience import ExperienceError
from .experience_proposal import (
    MAX_APPROVAL_BYTES, MAX_JOB_BYTES, MAX_OBSERVATION_BYTES, MAX_PROPOSAL_BYTES,
    check_approval, check_proposal, decode, sha, source_facts,
)
from .experience_store import read_private
from .enrollment import key_id, public_bytes
from .source_inventory import (
    MAX_COMMIT_BYTES, MAX_INVENTORY_BYTES, check_commit, check_snapshot, check_source_inventory, commit_snapshot,
)


SCHEMA = "organism-source-review-inbox-v2"
MAX_REQUESTS = 32
MAX_JOB_TRANSFER_BYTES = MAX_JOB_BYTES + 512
MAX_VALIDATION_BYTES = 65536
MAX_REQUEST_BYTES = MAX_PROPOSAL_BYTES + MAX_OBSERVATION_BYTES + MAX_JOB_TRANSFER_BYTES
MAX_STORED_BYTES = MAX_REQUEST_BYTES + MAX_APPROVAL_BYTES + MAX_VALIDATION_BYTES
MAX_DATABASE_PAGES = 32768


def packet_hash(proposal, observation, job):
    return sha(canonical([{"bytes": len(value), "sha256": sha(value)}
                          for value in (proposal, observation, job)]))


class ReviewInbox:
    def __init__(self, directory, config, policy, *, checker_id, reviewer, require_live_source=True,
                 clock=time.time, max_requests=MAX_REQUESTS, inventory_key=None, inventory_checkers=None):
        require(all(hash_value(value) for value in (config.device_id, config.release_id, checker_id))
                and token(config.individual_id) and callable(reviewer)
                and type(require_live_source) is bool and type(max_requests) is int
                and 1 <= max_requests <= MAX_REQUESTS, "Invalid source-review service configuration")
        policy.peer_key(config.device_id)
        self.directory = Path(directory).absolute()
        self.config, self.policy, self.checker_id, self.reviewer = config, policy, checker_id, reviewer
        self.require_live_source, self.clock, self.max_requests = require_live_source, clock, max_requests
        self.inventory_key = inventory_key
        self.inventory_checkers = frozenset(inventory_checkers if inventory_checkers is not None else {checker_id})
        require(1 <= len(self.inventory_checkers) <= 4 and all(map(hash_value, self.inventory_checkers))
                and (inventory_key is None or key_id(public_bytes(inventory_key.public_key())) == config.device_id),
                "Invalid independent inventory commitment configuration")
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        info = self.directory.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & 0o077,
                "Source-review directory must be private")
        self.path, marker = self.directory / "reviews.sqlite3", self.directory / "reviews.enrolled"
        expected = {"schema": SCHEMA, "policy_id": policy.policy_id, "device_id": config.device_id,
                    "individual_id": config.individual_id, "release_id": config.release_id,
                    "checker_id": checker_id, "require_live_source": require_live_source,
                    "max_requests": max_requests,
                    "inventory_checkers": sorted(self.inventory_checkers)}
        present = os.path.lexists(self.path)
        require(present == os.path.lexists(marker), "Review history or context is missing; do not reset it")
        if present:
            context = json.loads(read_private(marker, 4096), object_pairs_hook=unique_object)
            require(set(context) == set(expected) | {"inbox_id"}
                    and all(context[key] == value for key, value in expected.items())
                    and token(context["inbox_id"]), "Review inbox context differs")
        else:
            context = {**expected, "inbox_id": uuid.uuid4().hex}
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as output:
                output.write(canonical(context))
                output.flush()
                os.fsync(output.fileno())
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.close(fd)
        self.context = context
        with self.db() as db:
            if not present:
                db.execute("""CREATE TABLE decisions (
                    peer TEXT NOT NULL, request TEXT NOT NULL, proposal BLOB NOT NULL,
                    observation BLOB NOT NULL, job BLOB NOT NULL, packet_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL, approval BLOB NOT NULL, validation BLOB NOT NULL,
                    completed INTEGER NOT NULL, PRIMARY KEY(peer,request))""")
                db.execute("""CREATE TABLE head (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1), decisions INTEGER NOT NULL,
                    bytes INTEGER NOT NULL)""")
                db.execute("INSERT INTO head VALUES (1,0,0)")
                db.execute("""CREATE TABLE inventories (
                    peer TEXT NOT NULL, request TEXT NOT NULL, snapshot BLOB NOT NULL,
                    receipt BLOB NOT NULL, PRIMARY KEY(peer,request))""")
            self.check_head(db)
            for peer, request, raw, receipt in db.execute("SELECT * FROM inventories"):
                check_snapshot(bytes(raw), self.policy, release_id=self.config.release_id,
                               allowed_checkers=self.inventory_checkers,
                               require_live_source=self.require_live_source)
                check_commit(bytes(receipt), bytes(raw), self.policy, release_id=self.config.release_id,
                             producer_id=peer, request_id=request, verifier_id=self.config.device_id)
            for row in db.execute("SELECT * FROM decisions"):
                self.validate_row(row, db)
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @contextmanager
    def db(self):
        info = self.path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
                and info.st_nlink == 1 and not info.st_mode & 0o077, "Invalid review database")
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=3)
        try:
            db.execute("PRAGMA synchronous=FULL")
            require(db.execute("PRAGMA page_size").fetchone()[0] == 4096
                    and db.execute("PRAGMA max_page_count=" + str(MAX_DATABASE_PAGES)).fetchone()[0]
                    == MAX_DATABASE_PAGES, "Review database exceeds its page allowance")
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
        require(type(value) in (int, float) and 0 <= value <= 4000000000, "Invalid review clock")
        return int(value)

    def validate_request(self, peer, request_id, proposal):
        require(peer != self.config.device_id and hash_value(peer) and token(request_id),
                "Source review requires another enrolled producer")
        source = check_proposal(proposal, self.policy, release_id=self.config.release_id)
        require(source["producer_device_id"] == peer and source["request_id"] == request_id
                and source["verifier_device_id"] == self.config.device_id,
                "Signed source differs from its authenticated request")
        require(not self.require_live_source or source["source_kind"] == "live_model",
                "Fixture sources are disabled in this review service")
        return source

    def validate_packet(self, peer, request_id, proposal, observation, raw_job):
        source = self.validate_request(peer, request_id, proposal)
        require(isinstance(observation, bytes) and 0 < len(observation) <= MAX_OBSERVATION_BYTES
                and isinstance(raw_job, bytes) and 0 < len(raw_job) <= MAX_JOB_TRANSFER_BYTES,
                "Source-review artifacts exceed their allowances")
        try:
            document = decode(observation, MAX_OBSERVATION_BYTES)
            require(isinstance(document, dict), "Source observation must be an equipment object")
            bounded_tree(document)
            validate_document(document, document.get("device_id"))
            job = decode(raw_job, MAX_JOB_TRANSFER_BYTES)
            exact(job, ("record", "digest"))
            require(isinstance(job["record"], dict) and isinstance(job["record"].get("trace"), dict),
                    "Source job must contain a diagnostic record and trace")
            facts = source_facts(observation, job)
            require(all(source[name] == value for name, value in facts.items()),
                    "Source artifacts differ from their signed proposal")
            require(source["source_kind"] != "live_model" or job["record"].get("request_id") == request_id,
                    "Live source job differs from its review request")
        except (TypeError, KeyError, AttributeError, RecursionError) as exc:
            raise ExperienceError("Invalid source-review artifacts") from exc
        return job

    def validate_decision(self, proposal, approval, validation, job, observation):
        require(isinstance(approval, bytes) and 0 < len(approval) <= MAX_APPROVAL_BYTES
                and isinstance(validation, bytes) and 0 < len(validation) <= MAX_VALIDATION_BYTES,
                "Review decision exceeds its allowance")
        envelope = decode(approval, MAX_APPROVAL_BYTES)
        created = envelope.get("body", {}).get("created_unix_seconds")
        body, _ = check_approval(proposal, approval, policy=self.policy, release_id=self.config.release_id,
                                 allowed_checkers={self.checker_id}, now=created,
                                 require_live_source=self.require_live_source)
        receipt = decode(validation, MAX_VALIDATION_BYTES)
        require(isinstance(receipt, dict)
                and sha(validation) == body["source"]["validation_sha256"]
                and receipt.get("source_proposal_sha256") == sha(proposal)
                and receipt.get("source_job_sha256") == job["digest"]
                and receipt.get("observation_sha256") == sha(observation)
                and receipt.get("checker_sha256") == self.checker_id
                and receipt.get("trace_check", {}).get("semantic_passed") is True,
                "Verifier decision is not bound to its source validation")
        if job["record"]["trace"].get("source_inventory") is not None:
            require(receipt.get("signed_pre_inference_inventory_commit_verified") is True
                    and receipt.get("source_inventory_context") == job["record"]["trace"]["source_inventory"],
                    "Verifier omitted the independently retained inventory")

    def commit_inventory(self, peer, request_id, raw):
        require(self.inventory_key is not None and peer != self.config.device_id and token(request_id),
                "Independent inventory commitments are not configured")
        body, _ = check_snapshot(raw, self.policy, release_id=self.config.release_id,
                                  allowed_checkers=self.inventory_checkers,
                                  require_live_source=self.require_live_source)
        require(body["producer_device_id"] == peer and body["request_id"] == request_id,
                "Inventory differs from its authenticated producer or request")
        with self.db() as db:
            self.check_head(db)
            old = db.execute("SELECT snapshot,receipt FROM inventories WHERE peer=? AND request=?",
                             (peer, request_id)).fetchone()
            if old is not None:
                require(bytes(old[0]) == raw, "Inventory retry substituted procedures")
                check_commit(bytes(old[1]), raw, self.policy, release_id=self.config.release_id,
                             producer_id=peer, request_id=request_id, verifier_id=self.config.device_id)
                return {"receipt": bytes(old[1]), "cached": True}
            require(db.execute("SELECT 1 FROM decisions WHERE peer=? AND request=?",
                               (peer, request_id)).fetchone() is None,
                    "Cannot commit inventory after a source decision")
            require(self.request_count(db) < self.max_requests, "Inventory commitment capacity exhausted")
            receipt = commit_snapshot(raw, self.policy, self.inventory_key, release_id=self.config.release_id,
                                      allowed_checkers=self.inventory_checkers,
                                      require_live_source=self.require_live_source, now=self.now())
            db.execute("INSERT INTO inventories VALUES (?,?,?,?)", (peer, request_id, raw, receipt))
            db.execute("UPDATE head SET bytes=bytes+? WHERE singleton=1", (len(raw) + len(receipt),))
            self.check_head(db)
        return {"receipt": receipt, "cached": False}

    def source_inventory_for(self, peer, request_id, job, db=None):
        context = job["record"]["trace"].get("source_inventory")
        if context is None:
            return None
        if db is None:
            with self.db() as opened:
                self.check_head(opened)
                return self.source_inventory_for(peer, request_id, job, opened)
        row = db.execute("SELECT snapshot,receipt FROM inventories WHERE peer=? AND request=?",
                         (peer, request_id)).fetchone()
        require(row is not None, "Reviewer has no independently committed inventory")
        raw, receipt = map(bytes, row)
        check_source_inventory(raw, receipt, context, self.policy, release_id=self.config.release_id,
                               allowed_checkers=self.inventory_checkers, require_live_source=self.require_live_source,
                               producer_id=peer, request_id=request_id, verifier_id=self.config.device_id)
        return {"snapshot": raw, "receipt": receipt, "context": context}

    def request_count(self, db):
        return db.execute("""SELECT count(*) FROM (
            SELECT peer,request FROM decisions UNION SELECT peer,request FROM inventories)""").fetchone()[0]

    def check_head(self, db):
        head = db.execute("SELECT decisions,bytes FROM head WHERE singleton=1").fetchone()
        observed = db.execute("""SELECT count(*),coalesce(sum(length(proposal)+length(observation)+
                                 length(job)+length(approval)+length(validation)),0) FROM decisions""").fetchone()
        inventories = db.execute("SELECT snapshot,receipt FROM inventories").fetchall()
        size = sum(len(value) for row in inventories for value in row)
        require(len(inventories) <= self.max_requests and all(
            0 < len(raw) <= MAX_INVENTORY_BYTES and 0 < len(receipt) <= MAX_COMMIT_BYTES
            for raw, receipt in inventories), "Retained inventory exceeds its allowance")
        require(head == (observed[0], observed[1] + size) and self.request_count(db) <= self.max_requests
                and 0 <= head[0] <= self.max_requests
                and 0 <= head[1] <= self.max_requests * (MAX_STORED_BYTES + MAX_INVENTORY_BYTES + MAX_COMMIT_BYTES),
                "Review history differs from its committed head or allowance")

    def validate_row(self, row, db=None):
        peer, request, proposal, observation, raw_job, identity, status, approval, validation, completed = row
        require(status in ("approved", "rejected") and integer(completed)
                and all(isinstance(value, bytes) for value in (proposal, observation, raw_job, approval, validation))
                and identity == packet_hash(proposal, observation, raw_job), "Retained review source is corrupt")
        job = self.validate_packet(peer, request, proposal, observation, raw_job)
        if status == "approved":
            self.source_inventory_for(peer, request, job, db)
            self.validate_decision(proposal, approval, validation, job, observation)
        else:
            require(approval == validation == b"", "Rejected source contains a verifier approval")
        return {"peer_id": peer, "request_id": request, "proposal": proposal, "observation": observation,
                "job": job, "status": status, "approval": approval, "validation": validation,
                "completed": completed, "packet_sha256": identity}

    def submit(self, peer, request_id, proposal, observation, raw_job):
        job = self.validate_packet(peer, request_id, proposal, observation, raw_job)
        identity = packet_hash(proposal, observation, raw_job)
        with self.db() as db:
            self.check_head(db)
            row = db.execute("SELECT * FROM decisions WHERE peer=? AND request=?", (peer, request_id)).fetchone()
            if row is not None:
                retained = self.validate_row(row, db)
                require(retained["packet_sha256"] == identity, "Review retry substituted source artifacts")
                return {"status": retained["status"], "approval": retained["approval"], "cached": True}
            known = db.execute("SELECT 1 FROM inventories WHERE peer=? AND request=?",
                               (peer, request_id)).fetchone() is not None
            require(known or self.request_count(db) < self.max_requests,
                    "Review decision capacity exhausted")
            completed = self.now()
            try:
                inventory = self.source_inventory_for(peer, request_id, job, db)
                kwargs = ({"source_inventory": inventory, "inventory_checkers": self.inventory_checkers}
                          if inventory is not None else {})
                approval, receipt = self.reviewer(proposal, observation, job, **kwargs)
            except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
                # A signed source is still untrusted input. Structural failures
                # inside its independent trace checker are terminal rejections;
                # runtime/storage failures continue to stop the service.
                status, approval, validation = "rejected", b"", b""
            else:
                status, validation = "approved", canonical(receipt)
                # Reject a broken local reviewer before committing or replying.
                self.validate_decision(proposal, approval, validation, job, observation)
            db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (peer, request_id, proposal, observation, raw_job, identity,
                        status, approval, validation, completed))
            db.execute("UPDATE head SET decisions=decisions+1,bytes=bytes+? WHERE singleton=1",
                       (sum(map(len, (proposal, observation, raw_job, approval, validation))),))
            self.check_head(db)
        return {"status": status, "approval": approval, "cached": False}

    def get(self, peer, request_id):
        require(hash_value(peer) and token(request_id), "Invalid source-review identity")
        with self.db() as db:
            self.check_head(db)
            row = db.execute("SELECT * FROM decisions WHERE peer=? AND request=?", (peer, request_id)).fetchone()
        return self.validate_row(row) if row is not None else None

    def snapshot(self):
        with self.db() as db:
            self.check_head(db)
            count, size = db.execute("SELECT decisions,bytes FROM head WHERE singleton=1").fetchone()
            approved = db.execute("SELECT count(*) FROM decisions WHERE status='approved'").fetchone()[0]
            committed = db.execute("SELECT count(*) FROM inventories").fetchone()[0]
        return {"inbox_id": self.context["inbox_id"], "decisions": count, "retained_bytes": size,
                "approved": approved, "rejected": count - approved, "capacity": self.max_requests,
                "inventories_committed": committed,
                "require_live_source": self.require_live_source, "qualification_passed": False}
