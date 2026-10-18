"""An individual's bounded, authenticated exchange of endorsed reader records.

The owner declares every endpoint and checker in the signed discovery book.
The receiving substrate has no experience protocol or task knowledge.
An optional local review inbox can approve captured source evidence. Its
reviewer is provisioned separately; remote callers cannot supply procedures
or request model/task execution.
"""

import asyncio
import base64
import fcntl
import json
import os
import stat
import struct
import time
import uuid

from .capsule import canonical
from .experience import ExperienceError, exact, hash_value, require, token, unique_object
from .experience_store import MAX_RECORDS, read_private
from .experience_proposal import (
    MAX_APPROVAL_BYTES, MAX_OBSERVATION_BYTES, MAX_PROPOSAL_BYTES,
    accept as accept_approval, check_approval, check_proposal, propose, sha,
)
from .peer_audit import PeerAudit
from .peer_identity import LabNetwork, authenticate, device_hostname
from .review_inbox import MAX_JOB_TRANSFER_BYTES, MAX_REQUESTS as MAX_REVIEW_REQUESTS
from .source_review_queue import SourceReviewQueue
from .source_inventory import MAX_COMMIT_BYTES, MAX_INVENTORY_BYTES, check_commit, check_source_inventory, freeze


PROTOCOL = "organism-experience-exchange-v2"
MAX_FRAME_BYTES = 16384
REQUEST_SECONDS = 10
POLL_SECONDS = 60
MAX_HANDLERS = 4
MAX_DELIVERIES_PER_ROUND = 2
MAX_DELIVERY_ATTEMPTS = 3
MAX_REVIEW_ATTEMPTS = 3


async def read_blob(reader, maximum):
    count = struct.unpack("!I", await reader.readexactly(4))[0]
    require(0 < count <= maximum, "Source-review blob exceeds its bound")
    return await reader.readexactly(count)


async def write_blob(writer, raw, maximum):
    require(isinstance(raw, bytes) and 0 < len(raw) <= maximum, "Source-review blob exceeds its bound")
    writer.write(struct.pack("!I", len(raw)) + raw)
    await writer.drain()


async def read_frame(reader):
    count = struct.unpack("!I", await reader.readexactly(4))[0]
    require(0 < count <= MAX_FRAME_BYTES, "Experience frame exceeds its bound")
    try:
        return json.loads(await reader.readexactly(count), object_pairs_hook=unique_object)
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise ExperienceError("Invalid experience frame") from exc


async def write_frame(writer, value):
    raw = canonical(value)
    require(len(raw) <= MAX_FRAME_BYTES, "Experience frame exceeds its bound")
    writer.write(struct.pack("!I", len(raw)) + raw)
    await writer.drain()


async def close_writer(writer):
    writer.close()
    try:
        await asyncio.wait_for(writer.wait_closed(), timeout=1)
    except (OSError, TimeoutError):
        writer.transport.abort()


class ExperienceExchange:
    review_attempt_limit = MAX_REVIEW_ATTEMPTS
    review_request_limit = MAX_REVIEW_REQUESTS

    def __init__(self, config, store, book, client_tls, server_tls, *, interval=POLL_SECONDS, reviews=None):
        require(book.experience_checkers and book.policy.policy_id == store.policy.policy_id
                and book.experience_checkers == store.allowed_checkers
                and config.device_id == store.device_id and config.release_id == store.release_id,
                "Experience exchange belongs to another application or owner directory")
        require(type(interval) in (int, float) and 0 < interval <= 3600, "Invalid exchange interval")
        self.config, self.store, self.book = config, store, book
        require(reviews is None or (
            reviews.config == config and reviews.policy.policy_id == store.policy.policy_id
            and reviews.checker_id in store.allowed_checkers
            and reviews.require_live_source == store.require_live_source
            and reviews.inventory_checkers == store.allowed_checkers),
            "Source reviewer belongs to another application or trust context")
        self.reviews = reviews
        self.source_queue = None
        self.device_id, self.policy = store.device_id, store.policy
        self.client_tls, self.server_tls, self.interval = client_tls, server_tls, interval
        self.network = LabNetwork(book.body["lab_cidrs"])
        self.identities = tuple(sorted(set(book.body["endpoints"]) - {self.device_id}))
        self.endpoint = book.experience_endpoint(self.device_id)
        self.listener, self.lock_fd = None, None
        self.handlers = set()
        self.closed, self.started = False, False
        self.failure, self.run_task = None, None
        self.stop_event = asyncio.Event()
        self.round_lock = asyncio.Lock()
        self.peer_states, self.attempts = {}, {}
        self.review_attempts = {}
        self.inventory_attempts = {}
        self.round_number = 0
        expected = canonical({"schema": PROTOCOL, "device_id": self.device_id, "policy_id": self.policy.policy_id,
                              "book_id": book.book_id, "release_id": store.release_id, "store_id": store.store_id})
        marker = store.directory / "exchange.enrolled"
        audit_exists = any(os.path.lexists(store.directory / name)
                           for name in ("peer-audit.sqlite3", "peer-audit.enrolled"))
        if os.path.lexists(marker):
            require(read_private(marker, 4096) == expected, "Experience exchange context changed")
            require(audit_exists, "Experience exchange audit is missing")
        else:
            require(not audit_exists, "Experience exchange enrollment marker is missing")
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as output:
                output.write(expected); output.flush(); os.fsync(output.fileno())
        self.audit = PeerAudit(store.directory)
        # Attempts survive process restarts. Restarting never renews a delivery's
        # allowance. A missing or corrupt audit is not treated as an empty one.
        for item in self.audit.iter_records():
            event = item["record"]
            if event["operation"] == "experience-delivery" and event["phase"] == "attempted":
                peer, note = event["peer_id"], event["detail"].get("note_id")
                require(peer in self.identities and hash_value(note) and token(event["request_id"]),
                        "Invalid persisted experience attempt")
                key = peer, note
                count = self.attempts.get(key, 0) + 1
                require(event["detail"].get("attempt") == count <= MAX_DELIVERY_ATTEMPTS,
                        "Experience attempt history exceeds its allowance")
                self.attempts[key] = count
            elif event["operation"] == "experience-source-review" and event["phase"] == "attempted":
                peer, request = event["peer_id"], event["request_id"]
                identity = event["detail"].get("proposal_sha256")
                require(peer in self.identities and token(request) and hash_value(identity),
                        "Invalid persisted source-review attempt")
                key = peer, request
                prior_identity, prior_count = self.review_attempts.get(key, (identity, 0))
                require(prior_identity == identity
                        and event["detail"].get("attempt") == prior_count + 1 <= MAX_REVIEW_ATTEMPTS,
                        "Source-review history changed identity or exceeds its allowance")
                self.review_attempts[key] = identity, prior_count + 1
                require(len(self.review_attempts) <= MAX_REVIEW_REQUESTS, "Source-review request allowance exceeded")
            elif event["operation"] == "experience-source-inventory" and event["phase"] == "attempted":
                key = event["peer_id"], event["request_id"]
                identity = event["detail"].get("inventory_sha256")
                prior, count = self.inventory_attempts.get(key, (identity, 0))
                require(key[0] in self.identities and token(key[1]) and hash_value(identity)
                        and prior == identity and event["detail"].get("attempt") == count + 1 <= MAX_REVIEW_ATTEMPTS,
                        "Inventory commitment history changed or exceeds its allowance")
                self.inventory_attempts[key] = identity, count + 1
                require(len(self.inventory_attempts) <= MAX_REVIEW_REQUESTS, "Inventory request allowance exceeded")

    def bind_sources(self, journal):
        require(not self.started and self.source_queue is None, "Source capture must bind before experience startup")
        self.source_queue = SourceReviewQueue(self.store.directory / "source-review-queue", self, journal)

    def envelope(self, request_id, **fields):
        return {"protocol": PROTOCOL, "policy_id": self.policy.policy_id, "book_id": self.book.book_id,
                "release_id": self.store.release_id, "request_id": request_id, **fields}

    def check_envelope(self, value, fields):
        exact(value, ("protocol", "policy_id", "book_id", "release_id", "request_id") + tuple(fields))
        require(value["protocol"] == PROTOCOL and value["policy_id"] == self.policy.policy_id
                and value["book_id"] == self.book.book_id and value["release_id"] == self.store.release_id
                and token(value["request_id"]), "Experience exchange context differs")

    async def start(self):
        require(not self.started and not self.closed, "Experience exchange cannot be started again")
        lock = os.open(self.store.directory / ".exchange.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(lock)
            require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1
                    and not info.st_mode & 0o077, "Invalid experience service ownership lock")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_fd = lock
            self.listener = await asyncio.start_server(
                self.accept, self.endpoint[1], self.endpoint[2], ssl=self.server_tls,
                ssl_handshake_timeout=3, ssl_shutdown_timeout=1, limit=MAX_FRAME_BYTES + 4, backlog=MAX_HANDLERS)
            self.started = True
        except BaseException:
            if self.lock_fd == lock:
                self.lock_fd = None
            os.close(lock)
            raise

    def accept(self, reader, writer):
        if self.closed or len(self.handlers) >= MAX_HANDLERS:
            writer.transport.abort()
            return
        task = asyncio.create_task(self.handle(reader, writer))
        self.handlers.add(task)
        task.add_done_callback(self.handler_done)

    def handler_done(self, task):
        self.handlers.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self.failure = type(task.exception()).__name__
            self.stop_event.set()

    async def handle(self, reader, writer):
        request_id, peer, operation = uuid.uuid4().hex, None, None
        try:
            async with asyncio.timeout(REQUEST_SECONDS):
                self.network.address(writer.get_extra_info("peername")[0])
                peer, certificate = authenticate(writer.get_extra_info("ssl_object"), self.policy)
                require(peer != self.device_id and peer in self.identities, "Experience caller is not another enrolled peer")
                request = await read_frame(reader)
                operation = request.get("operation") if isinstance(request, dict) else None
                require(operation in ("inventory", "deliver", "review", "commit_inventory"), "Unsupported experience operation")
                fields = ("operation",) + ({"deliver": ("record",), "review": ("proposal",)}.get(operation, ()))
                self.check_envelope(request, fields)
                request_id = request["request_id"]
                approval = None
                if operation == "inventory":
                    result = {"note_ids": self.store.inventory(), "store_id": self.store.store_id,
                              "device_id": self.device_id, "individual_id": self.config.individual_id,
                              "source_reviewer": ({"checker_id": self.reviews.checker_id,
                                                   "inventory_commit": self.reviews.inventory_key is not None}
                                                  if self.reviews is not None else None)}
                elif operation == "deliver":
                    require(isinstance(request["record"], str), "Experience transfer has no encoded record")
                    raw = base64.b64decode(request["record"], validate=True)
                    acknowledgement = self.store.receive(raw, sender_id=peer, request_id=request_id)
                    result = {"receipt": base64.b64encode(acknowledgement).decode()}
                elif operation == "commit_inventory":
                    require(self.reviews is not None, "Source review is not configured")
                    raw = await read_blob(reader, MAX_INVENTORY_BYTES)
                    decision = self.reviews.commit_inventory(peer, request_id, raw)
                    result = {"receipt": base64.b64encode(decision["receipt"]).decode(),
                              "cached": decision["cached"]}
                else:
                    require(self.reviews is not None, "Source review is not configured")
                    require(isinstance(request["proposal"], str)
                            and len(request["proposal"]) <= (MAX_PROPOSAL_BYTES + 2) // 3 * 4,
                            "Source review has no bounded proposal")
                    proposal = base64.b64decode(request["proposal"], validate=True)
                    # Authenticate the signed request before reading larger blobs.
                    self.reviews.validate_request(peer, request_id, proposal)
                    observation = await read_blob(reader, MAX_OBSERVATION_BYTES)
                    job = await read_blob(reader, MAX_JOB_TRANSFER_BYTES)
                    decision = self.reviews.submit(peer, request_id, proposal, observation, job)
                    approval = decision["approval"] or None
                    result = {"status": decision["status"], "cached": decision["cached"],
                              "approval_bytes": len(approval) if approval else 0}
                self.audit.append(request_id=request_id, phase="served", peer_id=peer,
                                  operation="experience-" + operation, detail={"certificate_sha256": certificate})
                await write_frame(writer, self.envelope(request_id, ok=True, result=result, error=None))
                if approval is not None:
                    await write_blob(writer, approval, MAX_APPROVAL_BYTES)
        except asyncio.CancelledError:
            raise
        except (ValueError, OSError, EOFError, asyncio.IncompleteReadError) as exc:
            # Payloads, record contents, paths and exception messages are not echoed.
            try:
                self.audit.append(request_id=request_id, phase="rejected", peer_id=peer,
                                  operation="experience-request", detail={"error_type": type(exc).__name__})
                async with asyncio.timeout(1):
                    await write_frame(writer, self.envelope(request_id, ok=False, result=None, error="rejected"))
            except (ValueError, OSError, asyncio.CancelledError):
                pass
        finally:
            await close_writer(writer)

    async def request(self, peer, operation, *, request_id=None, **fields):
        require(peer in self.identities and not self.closed, "Experience destination is not an active enrolled peer")
        endpoint = self.book.experience_endpoint(peer)
        request_id = request_id or uuid.uuid4().hex
        writer = None
        try:
            async with asyncio.timeout(REQUEST_SECONDS):
                reader, writer = await asyncio.open_connection(
                    self.network.address(endpoint[1]), endpoint[2], ssl=self.client_tls,
                    server_hostname=device_hostname(peer), limit=MAX_FRAME_BYTES + 4,
                    ssl_handshake_timeout=3, ssl_shutdown_timeout=1)
                authenticate(writer.get_extra_info("ssl_object"), self.policy, expected=peer)
                await write_frame(writer, self.envelope(request_id, operation=operation, **fields))
                value = await read_frame(reader)
                self.check_envelope(value, ("ok", "result", "error"))
                require(value["request_id"] == request_id and value["ok"] is True and value["error"] is None
                        and isinstance(value["result"], dict), "Experience peer rejected or mismatched the request")
                return value["result"]
        finally:
            if writer is not None:
                await close_writer(writer)

    async def inventory(self, peer):
        result = await self.request(peer, "inventory")
        exact(result, ("note_ids", "store_id", "device_id", "individual_id", "source_reviewer"))
        notes = result["note_ids"]
        require(isinstance(notes, list) and len(notes) <= MAX_RECORDS
                and all(hash_value(note) for note in notes) and notes == sorted(set(notes))
                and token(result["store_id"]) and token(result["individual_id"])
                and result["device_id"] == peer, "Invalid authenticated experience inventory")
        if result["source_reviewer"] is not None:
            exact(result["source_reviewer"], ("checker_id", "inventory_commit"))
            require(hash_value(result["source_reviewer"]["checker_id"])
                    and type(result["source_reviewer"]["inventory_commit"]) is bool,
                    "Invalid advertised source checker")
        return result

    async def commit_inventory(self, peer, request_id, raw):
        require(peer in self.identities and not self.closed and token(request_id)
                and isinstance(raw, bytes) and 0 < len(raw) <= MAX_INVENTORY_BYTES,
                "Invalid inventory commitment destination or payload")
        key, identity = (peer, request_id), sha(raw)
        previous, count = self.inventory_attempts.get(key, (identity, 0))
        require(previous == identity and count < MAX_REVIEW_ATTEMPTS
                and (key in self.inventory_attempts or len(self.inventory_attempts) < MAX_REVIEW_REQUESTS),
                "Inventory commitment changed or exhausted its allowance")
        self.audit.append(request_id=request_id, phase="attempted", peer_id=peer,
                          operation="experience-source-inventory",
                          detail={"inventory_sha256": identity, "attempt": count + 1})
        self.inventory_attempts[key] = identity, count + 1
        endpoint, writer = self.book.experience_endpoint(peer), None
        try:
            async with asyncio.timeout(REQUEST_SECONDS):
                reader, writer = await asyncio.open_connection(
                    self.network.address(endpoint[1]), endpoint[2], ssl=self.client_tls,
                    server_hostname=device_hostname(peer), limit=MAX_FRAME_BYTES + 4,
                    ssl_handshake_timeout=3, ssl_shutdown_timeout=1)
                authenticate(writer.get_extra_info("ssl_object"), self.policy, expected=peer)
                await write_frame(writer, self.envelope(request_id, operation="commit_inventory"))
                await write_blob(writer, raw, MAX_INVENTORY_BYTES)
                response = await read_frame(reader)
                self.check_envelope(response, ("ok", "result", "error"))
                require(response["request_id"] == request_id and response["ok"] is True
                        and response["error"] is None, "Inventory peer rejected or mismatched the request")
                result = response["result"]
                exact(result, ("receipt", "cached"))
                require(type(result["cached"]) is bool and isinstance(result["receipt"], str)
                        and len(result["receipt"]) <= (MAX_COMMIT_BYTES + 2) // 3 * 4,
                        "Inventory commitment response exceeds its bound")
                receipt = base64.b64decode(result["receipt"], validate=True)
                check_commit(receipt, raw, self.policy, release_id=self.store.release_id,
                             producer_id=self.device_id, request_id=request_id, verifier_id=peer)
                self.audit.append(request_id=request_id, phase="observed", peer_id=peer,
                                  operation="experience-source-inventory",
                                  detail={"commit_sha256": sha(receipt), "cached": result["cached"]})
                return receipt
        finally:
            if writer is not None:
                await close_writer(writer)

    async def prepare_source(self, capture):
        """Freeze task inputs and request one verifier's durable commitment."""
        began = time.monotonic()
        view = freeze(self.store, self.config, capture.request_id)
        capture.inventory(view.raw)

        async def eligible(peer):
            async with asyncio.timeout(2):
                remote = await self.inventory(peer)
                reviewer = remote["source_reviewer"]
                return peer if (reviewer is not None and reviewer["inventory_commit"]
                                and reviewer["checker_id"] in self.store.allowed_checkers) else None

        try:
            async with asyncio.timeout(10):
                observed = await asyncio.gather(*(eligible(peer) for peer in self.identities), return_exceptions=True)
                for value in observed:
                    if isinstance(value, BaseException) and not isinstance(
                            value, (ValueError, OSError, EOFError, asyncio.IncompleteReadError)):
                        raise value
                peers = sorted(value for value in observed if isinstance(value, str))
                if peers:
                    peer = peers[int(sha(capture.request_id.encode()), 16) % len(peers)]
                    for _ in range(MAX_REVIEW_ATTEMPTS):
                        try:
                            receipt = await self.commit_inventory(peer, capture.request_id, view.raw)
                        except (ValueError, OSError, EOFError, asyncio.IncompleteReadError):
                            continue
                        context = {"inventory_sha256": sha(view.raw), "commit_sha256": sha(receipt),
                                   "model_started_unix_seconds": self.store.clock(),
                                   "preparation_seconds": round(time.monotonic() - began, 6)}
                        check_source_inventory(
                            view.raw, receipt, context, self.policy, release_id=self.store.release_id,
                            allowed_checkers=self.store.allowed_checkers,
                            require_live_source=self.store.require_live_source, producer_id=self.device_id,
                            request_id=capture.request_id, verifier_id=peer)
                        capture.inventory(view.raw, receipt, context)
                        view.source_context = context
                        break
        except (ValueError, OSError, EOFError, asyncio.IncompleteReadError):
            pass
        finally:
            view.preparation_seconds = round(time.monotonic() - began, 6)
        return view

    def capture_inventory_peer(self, source):
        inventory = source.get("source_inventory")
        if inventory is None or inventory["context"] is None:
            return None
        peer = json.loads(inventory["receipt"])["body"]["verifier_device_id"]
        require(source["job"]["record"]["trace"].get("source_inventory") == inventory["context"],
                "Source trace differs from its captured inventory")
        check_source_inventory(
            inventory["snapshot"], inventory["receipt"], inventory["context"], self.policy,
            release_id=self.store.release_id, allowed_checkers=self.store.allowed_checkers,
            require_live_source=self.store.require_live_source, producer_id=self.device_id,
            request_id=source["request_id"], verifier_id=peer)
        return peer

    async def request_review(self, peer, proposal, observation, job):
        """One charged attempt; callers can retry the same signed request."""
        require(peer in self.identities and not self.closed, "Review destination is not an active enrolled peer")
        source = check_proposal(proposal, self.policy, release_id=self.store.release_id)
        require(source["producer_device_id"] == self.device_id and source["verifier_device_id"] == peer,
                "Review proposal belongs to another producer or verifier")
        require(isinstance(observation, bytes) and 0 < len(observation) <= MAX_OBSERVATION_BYTES,
                "Review observation exceeds its allowance")
        raw_job = canonical(job)
        require(len(raw_job) <= MAX_JOB_TRANSFER_BYTES, "Review job exceeds its allowance")
        request_id, identity = source["request_id"], sha(proposal)
        key = peer, request_id
        prior_identity, count = self.review_attempts.get(key, (identity, 0))
        require(prior_identity == identity and count < MAX_REVIEW_ATTEMPTS,
                "Review request changed identity or exhausted its attempts")
        require(key in self.review_attempts or len(self.review_attempts) < MAX_REVIEW_REQUESTS,
                "Source-review request capacity exhausted")
        self.audit.append(request_id=request_id, phase="attempted", peer_id=peer,
                          operation="experience-source-review",
                          detail={"proposal_sha256": identity, "attempt": count + 1})
        self.review_attempts[key] = identity, count + 1
        endpoint, writer = self.book.experience_endpoint(peer), None
        try:
            async with asyncio.timeout(REQUEST_SECONDS):
                reader, writer = await asyncio.open_connection(
                    self.network.address(endpoint[1]), endpoint[2], ssl=self.client_tls,
                    server_hostname=device_hostname(peer), limit=MAX_FRAME_BYTES + 4,
                    ssl_handshake_timeout=3, ssl_shutdown_timeout=1)
                authenticate(writer.get_extra_info("ssl_object"), self.policy, expected=peer)
                await write_frame(writer, self.envelope(request_id, operation="review",
                                                        proposal=base64.b64encode(proposal).decode()))
                await write_blob(writer, observation, MAX_OBSERVATION_BYTES)
                await write_blob(writer, raw_job, MAX_JOB_TRANSFER_BYTES)
                response = await read_frame(reader)
                self.check_envelope(response, ("ok", "result", "error"))
                require(response["request_id"] == request_id and response["ok"] is True
                        and response["error"] is None, "Source-review peer rejected or mismatched the request")
                result = response["result"]
                exact(result, ("status", "cached", "approval_bytes"))
                require(result["status"] in ("approved", "rejected") and type(result["cached"]) is bool
                        and type(result["approval_bytes"]) is int, "Invalid source-review response")
                if result["status"] == "approved":
                    require(0 < result["approval_bytes"] <= MAX_APPROVAL_BYTES, "Review approval exceeds its allowance")
                    approval = await read_blob(reader, MAX_APPROVAL_BYTES)
                    require(len(approval) == result["approval_bytes"], "Review approval size differs")
                    check_approval(proposal, approval, policy=self.policy, release_id=self.store.release_id,
                                   allowed_checkers=self.store.allowed_checkers, now=self.store.now(),
                                   require_live_source=self.store.require_live_source)
                else:
                    require(result["approval_bytes"] == 0, "Rejected review contains an approval")
                    approval = None
                self.audit.append(request_id=request_id, phase="observed", peer_id=peer,
                                  operation="experience-source-review",
                                  detail={"status": result["status"], "cached": result["cached"]})
                return {"status": result["status"], "approval": approval, "cached": result["cached"]}
        finally:
            if writer is not None:
                await close_writer(writer)

    def capture_proposal(self, peer, source):
        """Bind a retained live capture to one enrolled verifier."""
        require(source["state"] == "awaiting_review" and source["independently_verified"] is False,
                "Diagnostic source is not awaiting independent review")
        committed_peer = self.capture_inventory_peer(source)
        require(committed_peer is None or peer == committed_peer,
                "Source review must use the peer that committed its inventory")
        runtime = source["runtime_context"]
        require(all(runtime[field] == getattr(self.config, field) for field in (
            "device_id", "individual_id", "release_id", "model_sha256", "model_bytes",
            "executable_sha256", "native", "trained_for_organism")),
            "Source capture belongs to another runtime")
        observation = source["observations"][source["source_sha256"]]
        job = source["job"]
        reads = [event["result"] for event in job["record"]["trace"]["tool_events"]
                 if event.get("tool") == "read_equipment" and event.get("result", {}).get("status") == "observed"]
        require(reads and job["record"]["request_id"] == source["request_id"], "Captured source identity differs")
        return propose(observation, job, policy=self.policy, producer_key=self.store.device_key,
                           producer_individual_id=self.config.individual_id, verifier_device_id=peer,
                           release_id=self.store.release_id, request_id=source["request_id"],
                           source_clock=reads[-1]["read_unix_seconds"], live=True,
                           clock_window=[source["started"], source["ended"]])

    async def review_capture(self, peer, source):
        """Submit one retained live capture and retain an approved procedure."""
        proposal = self.capture_proposal(peer, source)
        observation, job = source["observations"][source["source_sha256"]], source["job"]
        decision = await self.request_review(peer, proposal, observation, job)
        if decision["status"] == "rejected":
            return {"status": "rejected", "note_id": None}
        record = accept_approval(proposal, decision["approval"], policy=self.policy,
                                 producer_key=self.store.device_key, release_id=self.store.release_id,
                                 allowed_checkers=self.store.allowed_checkers, now=self.store.now(),
                                 require_live_source=True)
        return {"status": "approved", "note_id": self.store.add(record)}

    async def deliver(self, peer, note_id, *, request_id):
        record = self.store.get(note_id)
        result = await self.request(peer, "deliver", request_id=request_id,
                                    record=base64.b64encode(record.raw).decode())
        exact(result, ("receipt",))
        require(isinstance(result["receipt"], str), "Experience peer returned no copy receipt")
        receipt = base64.b64decode(result["receipt"], validate=True)
        return self.store.confirm_copy(note_id, receiver_id=peer, request_id=request_id, receipt=receipt)

    async def round(self):
        require(self.started and not self.closed, "Experience exchange is not running")
        async with self.round_lock:
            notes = self.store.inventory()
            copies = {note: set(self.store.copy_state(note)["historically_acknowledged_other_devices"]) for note in notes}
            # Rotate first contact so an unavailable peer or large note inventory
            # cannot permanently keep the other enrolled peers behind it.
            offset = self.round_number % len(self.identities) if self.identities else 0
            peers = self.identities[offset:] + self.identities[:offset]
            self.round_number += 1
            deliveries = 0
            reviewers = []
            for peer in peers:
                if self.closed:
                    return
                probe_id = uuid.uuid4().hex
                try:
                    remote = await self.inventory(peer)
                    self.peer_states[peer] = {"state": "reachable", "observed_unix_seconds": self.store.now(),
                                              "store_id": remote["store_id"], "notes": len(remote["note_ids"])}
                    self.audit.append(request_id=probe_id, phase="observed", peer_id=peer,
                                      operation="experience-probe", detail={"store_id": remote["store_id"]})
                    reviewer = remote["source_reviewer"]
                    if reviewer is not None and reviewer["checker_id"] in self.store.allowed_checkers:
                        reviewers.append(peer)
                except (ValueError, OSError, EOFError, asyncio.IncompleteReadError) as exc:
                    self.peer_states[peer] = {"state": "unreachable", "observed_unix_seconds": self.store.now(),
                                              "error_type": type(exc).__name__}
                    self.audit.append(request_id=probe_id, phase="failed", peer_id=peer,
                                      operation="experience-probe", detail={"error_type": type(exc).__name__})
                    continue
                for note in notes:
                    key = peer, note
                    if ((note in remote["note_ids"] and peer in copies[note])
                            or self.attempts.get(key, 0) >= MAX_DELIVERY_ATTEMPTS
                            or deliveries >= MAX_DELIVERIES_PER_ROUND):
                        continue
                    attempt, request_id = self.attempts.get(key, 0) + 1, uuid.uuid4().hex
                    self.audit.append(request_id=request_id, phase="attempted", peer_id=peer,
                        operation="experience-delivery", detail={"note_id": note, "attempt": attempt})
                    self.attempts[key] = attempt
                    deliveries += 1
                    try:
                        await self.deliver(peer, note, request_id=request_id)
                        phase, detail = "confirmed", {"note_id": note}
                    except asyncio.CancelledError:
                        self.audit.append(request_id=request_id, phase="interrupted", peer_id=peer,
                                          operation="experience-delivery", detail={"note_id": note})
                        raise
                    except (ValueError, OSError, EOFError, asyncio.IncompleteReadError) as exc:
                        phase, detail = "failed", {"note_id": note, "error_type": type(exc).__name__}
                    self.audit.append(request_id=request_id, phase=phase, peer_id=peer,
                                      operation="experience-delivery", detail=detail)
            if self.source_queue is not None and not self.closed:
                await self.source_queue.round(reviewers)

    def snapshot(self):
        return {"enabled": True, "running": self.started and not self.closed and self.listener is not None,
                "book_id": self.book.book_id, "local_note_ids": self.store.inventory(),
                "peers": {peer: dict(value) for peer, value in self.peer_states.items()},
                "exhausted_deliveries": sum(count >= MAX_DELIVERY_ATTEMPTS for count in self.attempts.values()),
                "review_inbox": self.reviews.snapshot() if self.reviews is not None else None,
                "source_review_queue": self.source_queue.snapshot() if self.source_queue is not None else None,
                "source_review_requests": len(self.review_attempts),
                "exhausted_reviews": sum(count >= MAX_REVIEW_ATTEMPTS for _, count in self.review_attempts.values()),
                "peer_observations_are_historical": True, "qualification_passed": False}

    async def run(self):
        require(self.started and not self.closed, "Start the experience service before its exchange loop")
        require(self.run_task is None, "Experience exchange loop is already running")
        self.run_task = asyncio.current_task()
        try:
            while not self.closed:
                await self.round()
                if self.failure:
                    raise RuntimeError("Experience handler failed: " + self.failure)
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=self.interval)
                except TimeoutError:
                    pass
                if self.failure:
                    raise RuntimeError("Experience handler failed: " + self.failure)
        finally:
            self.run_task = None
            await self.close()

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        if self.run_task is not None and self.run_task is not asyncio.current_task():
            self.run_task.cancel()
            await asyncio.gather(self.run_task, return_exceptions=True)
        if self.listener is not None:
            self.listener.close()
        active = list(self.handlers)
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        if self.listener is not None:
            await self.listener.wait_closed()
            self.listener = None
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
