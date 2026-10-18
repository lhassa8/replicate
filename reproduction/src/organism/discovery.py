"""Bounded authenticated discovery inside each individual, without model calls."""

import asyncio
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import time
import uuid

from .capsule import canonical
from .enrollment import unique_object
from .peer_audit import PeerAudit
from .peer_identity import PROTOCOL, authenticate, device_hostname
from .peer_transport import PeerClient


POLL_SECONDS = 60
PROBE_SECONDS = 5
MAX_RESPONSE_BYTES = 8192
PARALLEL_PROBES = 2
OBSERVATION_SECONDS = 30


def elapsed_clock():
    return time.clock_gettime(time.CLOCK_BOOTTIME) if hasattr(time, "CLOCK_BOOTTIME") else time.monotonic()


class DiscoveryError(ValueError):
    pass


def public_observation(result, policy, identity):
    expected = {"device_id", "policy_id", "platform", "revision", "owner_mode", "available",
                "admission_ticket", "observation_valid_seconds", "transport_receipt"}
    if not isinstance(result, dict) or set(result) != expected:
        raise DiscoveryError("Unexpected peer observation fields")
    if (result["device_id"] != identity or result["policy_id"] != policy.policy_id
            or result["platform"] != policy.platform(identity)):
        raise DiscoveryError("Peer observation differs from its authenticated enrollment")
    if (type(result["revision"]) is not int or not 0 <= result["revision"] < 2**53
            or result["owner_mode"] not in ("enabled", "stopped", "retired")
            or type(result["available"]) is not bool
            or result["available"] and result["owner_mode"] != "enabled"
            or result["observation_valid_seconds"] != OBSERVATION_SECONDS):
        raise DiscoveryError("Invalid peer admission observation")
    ticket = result["admission_ticket"]
    if not isinstance(ticket, str) or not 1 <= len(ticket) <= 2048:
        raise DiscoveryError("Invalid observation ticket envelope")
    PeerClient.check_anchor(result)
    # Discovery identifies candidates. A later birth operation must obtain its
    # own current admission ticket; background discovery never caches tickets.
    return {key: value for key, value in result.items() if key != "admission_ticket"}


async def observe_peer(book, identity, tls):
    """One complete, read-only TLS exchange under one wall-clock deadline."""
    _, address, port = book.endpoint(identity)
    writer = None
    try:
        async with asyncio.timeout(PROBE_SECONDS):
            reader, writer = await asyncio.open_connection(address, port, ssl=tls,
                                                          server_hostname=device_hostname(identity),
                                                          limit=MAX_RESPONSE_BYTES + 4)
            negotiated = writer.get_extra_info("ssl_object")
            try:
                authenticated, certificate = authenticate(negotiated, book.policy, expected=identity)
            except Exception as exc:
                raise DiscoveryError("Peer identity could not be verified") from exc
            message = canonical({"protocol": PROTOCOL, "policy_id": book.policy.policy_id, "operation": "observe"})
            writer.write(struct.pack("!I", len(message)) + message)
            await writer.drain()
            count = struct.unpack("!I", await reader.readexactly(4))[0]
            if not 0 < count <= MAX_RESPONSE_BYTES:
                raise DiscoveryError("Discovery response exceeds its bound")
            raw = await reader.readexactly(count)
            response = json.loads(raw, object_pairs_hook=unique_object)
            if (not isinstance(response, dict) or set(response) != {"protocol", "policy_id", "ok", "result", "error"}
                    or response["protocol"] != PROTOCOL or response["policy_id"] != book.policy.policy_id
                    or response["ok"] is not True or response["error"] is not None):
                raise DiscoveryError("Peer did not complete the authenticated observation")
            observation = public_observation(response["result"], book.policy, identity)
            return {"observation": observation, "certificate_sha256": certificate, "tls": negotiated.version(),
                    "authenticated_device_id": authenticated}
    finally:
        if writer is not None:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=1)
            except (OSError, TimeoutError):
                pass


class Discovery:
    def __init__(self, directory, book, local_identity, tls, *, boot_id, start_token,
                 clock=elapsed_clock, interval=POLL_SECONDS, probe=observe_peer):
        self.directory = Path(directory).absolute()
        self.book, self.local_identity, self.tls = book, local_identity, tls
        book.policy.peer_key(local_identity)
        if (not isinstance(boot_id, str) or not 1 <= len(boot_id) <= 128
                or not isinstance(start_token, str) or not re.fullmatch("[0-9a-f]{32}", start_token)):
            raise DiscoveryError("Discovery requires a local boot and incarnation identity")
        if type(interval) not in (int, float) or not math.isfinite(interval) or not 0.01 <= interval <= POLL_SECONDS:
            raise DiscoveryError("Invalid discovery interval")
        self.boot_id, self.start_token, self.clock, self.interval, self.probe = boot_id, start_token, clock, interval, probe
        self.directory.mkdir(mode=0o700, exist_ok=True)
        info = self.directory.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077):
            raise DiscoveryError("Discovery state must be private to the individual")
        marker = self.directory / "discovery.json"
        identity = {"schema": "organism-discovery-state-v1", "book_id": book.book_id,
                    "policy_id": book.policy.policy_id, "local_device_id": local_identity}
        new_state = not os.path.lexists(marker)
        if new_state:
            if any(self.directory.iterdir()):
                raise DiscoveryError("Existing discovery state has lost its identity marker")
            with marker.open("xb") as output:
                output.write(canonical(identity)); output.flush(); os.fsync(output.fileno())
            marker.chmod(0o600)
            fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        entry = marker.lstat()
        if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.geteuid() or entry.st_nlink != 1
                or entry.st_mode & 0o077 or entry.st_size > 8192 or json.loads(marker.read_bytes()) != identity):
            raise DiscoveryError("Discovery state belongs to another local identity or address book")
        if not new_state and any(not os.path.lexists(self.directory / name)
                                 for name in ("peer-audit.sqlite3", "peer-audit.enrolled")):
            raise DiscoveryError("Discovery history is missing; do not silently reset it")
        self.audit = PeerAudit(self.directory)
        self.identities = tuple(sorted(set(book.body["endpoints"]) - {local_identity}))
        self.latest = {}
        self.revisions = {}
        self.pending = set()
        # Retain prior evidence for comparison, but no previous process/boot's
        # observations become currently actionable until probed again.
        for item in self.audit.iter_records():
            record = item["record"]
            if record["operation"] == "discover" and record["phase"] in ("observed", "unreachable", "interrupted"):
                if record["peer_id"] not in self.identities:
                    raise DiscoveryError("Discovery history contains an unenrolled destination")
                self.latest[record["peer_id"]] = {**record["detail"], "audit_receipt": {
                    "sequence": item["sequence"], "digest": item["digest"]}}
                if record["detail"].get("observation"):
                    self.revisions[record["peer_id"]] = max(self.revisions.get(record["peer_id"], -1),
                                                           record["detail"]["observation"]["revision"])
        self.events = asyncio.Queue(maxsize=len(self.identities) or 1)
        self.stop_requested = asyncio.Event()
        self.tasks = set()
        self.semaphore = asyncio.Semaphore(PARALLEL_PROBES)
        self.failure = None
        self.lock_fd = None
        self.closed = False

    def snapshot(self):
        now = self.clock()
        rows = []
        for identity in self.identities:
            previous = self.latest.get(identity)
            valid = previous and previous.get("boot_id") == self.boot_id and previous.get("start_token") == self.start_token
            age = now - previous["observed_clock"] if valid else None
            fresh = valid and age is not None and 0 <= age < OBSERVATION_SECONDS
            observation = previous.get("observation") if previous else None
            rows.append({"device_id": identity, "endpoint": self.book.endpoint(identity)[1:],
                         "state": ("current" if fresh and observation else
                                   ("unreachable" if fresh else ("stale" if previous else "unseen"))),
                         "available_observed": observation["available"] if observation else False,
                         "admission_requires_fresh_observation": True,
                         "age_seconds": age, "observation": observation,
                         "audit_receipt": previous.get("audit_receipt") if previous else None})
        return rows

    async def probe_one(self, identity):
        async with self.semaphore:
            if self.stop_requested.is_set():
                return
            request = uuid.uuid4().hex
            common = {"boot_id": self.boot_id, "start_token": self.start_token}
            self.audit.append(request_id=request, phase="started", peer_id=identity, operation="discover",
                              detail={**common, "endpoint": self.book.endpoint(identity)[1:]})
            previous = self.latest.get(identity)
            cancelled = False
            try:
                result = await self.probe(self.book, identity, self.tls)
                # Validate again even when a development probe adapter is injected.
                observation = result["observation"]
                if (result["authenticated_device_id"] != identity or observation["device_id"] != identity
                        or observation["policy_id"] != self.book.policy.policy_id):
                    raise DiscoveryError("Probe result has another identity")
                if observation["revision"] < self.revisions.get(identity, -1):
                    raise DiscoveryError("Peer revision moved backwards")
                detail, phase = {**common, **result}, "observed"
            except asyncio.CancelledError:
                detail, phase, cancelled = {**common, "observation": None, "error_type": "CancelledError"}, "interrupted", True
            except (OSError, ValueError, TypeError, KeyError, TimeoutError, asyncio.IncompleteReadError) as exc:
                detail, phase = {**common, "observation": None, "error_type": type(exc).__name__}, "unreachable"
            detail["observed_clock"] = self.clock()
            detail["observed_at"] = datetime.now(timezone.utc).isoformat()
            anchor = self.audit.append(request_id=request, phase=phase, peer_id=identity, operation="discover", detail=detail)
            self.latest[identity] = {**detail, "audit_receipt": anchor}
            observation = detail.get("observation")
            if observation:
                self.revisions[identity] = observation["revision"]
            old_observation = previous.get("observation") if previous else None
            changed = (not old_observation or previous.get("start_token") != self.start_token
                       or any(old_observation[name] != observation[name] for name in
                              ("revision", "available", "owner_mode", "platform"))) if observation else False
            if observation and observation["available"] and changed and identity not in self.pending:
                self.pending.add(identity)
                self.events.put_nowait(identity)
            if cancelled:
                raise asyncio.CancelledError()

    async def next_candidate(self):
        identity = await self.events.get()
        self.pending.discard(identity)
        return next(item for item in self.snapshot() if item["device_id"] == identity)

    async def _worker(self, identity):
        while not self.stop_requested.is_set():
            started = self.clock()
            await self.probe_one(identity)
            delay = max(0, self.interval - (self.clock() - started))
            try:
                await asyncio.wait_for(self.stop_requested.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def run(self):
        if self.tasks or self.lock_fd is not None or self.closed:
            raise DiscoveryError("Discovery loop is already running")
        fd = os.open(self.directory / ".discovery.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            entry = os.fstat(fd)
            if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.geteuid() or entry.st_nlink != 1
                    or entry.st_mode & 0o077):
                raise DiscoveryError("Invalid discovery ownership lock")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise
        self.lock_fd = fd
        try:
            self.tasks = {asyncio.create_task(self._worker(identity)) for identity in self.identities}
            if self.tasks:
                await asyncio.gather(*self.tasks)
            else:
                await self.stop_requested.wait()
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self.failure = type(exc).__name__ + ": " + str(exc)[:256]
            raise
        finally:
            self.stop_requested.set()
            for task in self.tasks:
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.tasks.clear()
            self.closed = True
            os.close(self.lock_fd)
            self.lock_fd = None

    async def close(self):
        self.stop_requested.set()
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
