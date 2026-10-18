"""Framed TLS peer transfer on an explicitly declared private lab network.

The protocol exposes observation, reservation, signed component transfer, commit,
status and cancellation. Owner controls, readiness claims and process-start
tokens are deliberately absent from its wire interface.
"""

from contextlib import contextmanager
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import socketserver
import sqlite3
import ssl
import stat
import struct
import tempfile
import threading
import time
import uuid

from .capsule import CapsuleError, MAX_MANIFEST_BYTES, _read_beneath, canonical, verify_manifest
from .enrollment import EnrollmentError, unique_object
from .peer_audit import MAX_RECORDS, PeerAudit, PeerAuditError
from .peer_identity import LabNetwork, PROTOCOL, PeerIdentityError, authenticate, context, device_hostname
from .receiver import stored_bytes
from .receiver_state import AdmissionError


MAX_CONTROL = 2 * 1024**2
BUFFER_BYTES = 64 * 1024
CONTROL_SECONDS = 10
TRANSFER_SECONDS = 120
# Keep the self-contained application aligned with the owner network profile.
# A focused test checks this value against network_budget.RATE.
NETWORK_BYTES_PER_SECOND = 8 * 1024**2
ID = re.compile(r"[0-9a-f]{32}\Z")


class PeerProtocolError(ValueError):
    pass


def payload_seconds(byte_count, policy):
    """Bound a signed payload by its size and the existing enrollment lifetime."""
    if (type(byte_count) is not int or not 0 < byte_count <= policy["max_payload_bytes"]
            or type(policy["birth_seconds"]) is not int or not 1 <= policy["birth_seconds"] <= 1200):
        raise PeerProtocolError("Payload exceeds its enrolled size or lifetime")
    # Reserve half the shared edge's rate for payload progress. Control traffic,
    # acknowledgments and retransmission still count against its kernel limit.
    # Receiver state independently retains the original birth deadline across
    # every component and retry; this is never a fresh birth allowance.
    size_seconds = (2 * byte_count + NETWORK_BYTES_PER_SECOND - 1) // NETWORK_BYTES_PER_SECOND
    return min(policy["birth_seconds"], max(TRANSFER_SECONDS, size_seconds))


def read_exact(connection, count, deadline, *, idle_timeout=CONTROL_SECONDS):
    chunks, remaining = [], count
    while remaining:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("Peer operation deadline exceeded")
        connection.settimeout(min(idle_timeout, left))
        block = connection.recv(min(remaining, BUFFER_BYTES))
        if not block:
            raise PeerProtocolError("Peer disconnected before completing its frame")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def receive_control(connection, *, timeout=CONTROL_SECONDS):
    deadline = time.monotonic() + timeout
    length = struct.unpack("!I", read_exact(connection, 4, deadline, idle_timeout=timeout))[0]
    if not 0 < length <= MAX_CONTROL:
        raise PeerProtocolError("Control frame exceeds its size bound")
    try:
        message = json.loads(read_exact(connection, length, deadline, idle_timeout=timeout), object_pairs_hook=unique_object)
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise PeerProtocolError("Malformed control frame") from exc
    if not isinstance(message, dict):
        raise PeerProtocolError("A control object is required")
    return message


def send_control(connection, value):
    body = canonical(value)
    if not 0 < len(body) <= MAX_CONTROL:
        raise PeerProtocolError("Control response exceeds its size bound")
    connection.settimeout(CONTROL_SECONDS)
    connection.sendall(struct.pack("!I", len(body)) + body)


def exact_fields(value, extra):
    if set(value) != {"protocol", "policy_id", "operation"} | set(extra):
        raise PeerProtocolError("Unexpected peer request fields")


def flush_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def file_identity(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise PeerProtocolError("Transfer file is not regular")
        digest, size = hashlib.sha256(), 0
        while block := source.read(BUFFER_BYTES):
            size += len(block)
            digest.update(block)
        return {"bytes": size, "sha256": digest.hexdigest()}


class PeerServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = False
    block_on_close = True

    def __init__(self, receiver, identity_directory, *, bind=("127.0.0.1", 0), network=None,
                 incoming_directory=None):
        self.receiver, self.state = receiver, receiver.state
        self.policy = self.state.policy
        self.network = network or LabNetwork()
        address = self.network.address(bind[0])
        if type(bind[1]) is not int or not 0 <= bind[1] <= 65535:
            raise PeerProtocolError("Invalid listening port")
        self.tls = context(self.policy, self.state.device_id, identity_directory, server=True)
        self._service_lock = os.open(self.state.directory / ".peer-service.lock",
                                     os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(self._service_lock).st_mode):
                raise PeerProtocolError("Invalid peer service lock")
            fcntl.flock(self._service_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.incoming = self.state.directory / "incoming" if incoming_directory is None else Path(incoming_directory)
            if (not self.incoming.is_absolute() or ".." in self.incoming.parts
                    or any(ord(char) < 32 or ord(char) == 127 for char in str(self.incoming))):
                raise PeerProtocolError("Incoming-transfer directory must be an explicit absolute path")
            self.incoming.mkdir(mode=0o700, exist_ok=True)
            incoming = self.incoming.lstat()
            if (not stat.S_ISDIR(incoming.st_mode) or incoming.st_uid != os.geteuid()
                    or incoming.st_mode & 0o077):
                raise PeerProtocolError("Incoming-transfer directory must be private to the owner receiver")
            self.audit = PeerAudit(self.state.directory)
            self._slots = threading.BoundedSemaphore(4)
            self._transfer = threading.Lock()
            self._connections_lock = threading.Lock()
            self._connections = set()
            self._stopping = threading.Event()
            self._thread = None
            super().__init__((address, bind[1]), _Handler)
        except BaseException:
            os.close(self._service_lock)
            self._service_lock = None
            raise

    def process_request(self, request, client_address):
        if self._stopping.is_set():
            self.shutdown_request(request)
            return
        try:
            self.network.address(client_address[0])
        except PeerIdentityError:
            self.shutdown_request(request)
            return
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        with self._connections_lock:
            self._connections.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            with self._connections_lock:
                self._connections.discard(request)
            self.shutdown_request(request)
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._connections_lock:
                self._connections.discard(request)
            self._slots.release()

    def start(self):
        if self._thread is not None:
            raise PeerProtocolError("Peer service is already started")
        self._thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.05},
                                        name="organism-peer-listener", daemon=False)
        self._thread.start()
        return self

    def close(self):
        self._stopping.set()
        with self._connections_lock:
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._thread is not None and self._thread.is_alive():
            self.shutdown()
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                raise PeerProtocolError("Owned peer listener did not stop")
        self.server_close()
        if self._service_lock is not None:
            os.close(self._service_lock)
            self._service_lock = None

    def reply(self, connection, result=None, error=None):
        send_control(connection, {"protocol": PROTOCOL, "policy_id": self.policy.policy_id,
                                  "ok": error is None, "result": result, "error": error})

    def receipt(self, peer, operation_id):
        if not isinstance(operation_id, str) or not ID.fullmatch(operation_id):
            raise PeerProtocolError("Invalid birth operation identity")
        return self.state.peer_receipt(peer, operation_id)

    def active_transfer(self, peer, operation_id):
        receipt = self.receipt(peer, operation_id)
        snapshot = self.state.snapshot()
        if snapshot["owner_mode"] != "enabled" or receipt["status"] != "receiving":
            raise AdmissionError("This birth is no longer accepting payloads")
        if not snapshot["current_birth"] or snapshot["current_birth"]["operation_id"] != operation_id:
            raise AdmissionError("Birth reservation no longer owns the receiving slot")
        return receipt

    @contextmanager
    def transfer_lock(self):
        if not self._transfer.acquire(blocking=False):
            raise AdmissionError("Another payload operation is in progress; retry the same birth")
        try:
            if self._stopping.is_set():
                raise AdmissionError("Peer service is stopping")
            yield
        finally:
            self._transfer.release()

    def cleanup_inactive(self, request_id):
        """Caller holds the transfer lock; only known inactive input directories."""
        for directory in sorted(self.incoming.iterdir()):
            if re.fullmatch(r"\.session-[a-z0-9_]{8}", directory.name) and not directory.is_symlink() and directory.is_dir():
                # This name is never published or used as an active transfer.
                if any(path.name not in (".manifest.json", ".intent.json") for path in directory.iterdir()):
                    raise PeerProtocolError("Unexpected unfinished session content")
                size = stored_bytes(directory)
                if size > MAX_MANIFEST_BYTES + 4096:
                    raise PeerProtocolError("Unfinished session exceeds its bound")
                self.audit.append(request_id=request_id, phase="unfinished_session_discarded",
                                  detail={"directory": directory.name, "discarded_input_bytes": size})
                shutil.rmtree(directory)
                flush_directory(self.incoming)
                continue
            if not ID.fullmatch(directory.name) or directory.is_symlink() or not directory.is_dir():
                raise PeerProtocolError("Unrecognized incoming state requires owner inspection")
            intent = self.load_intent(directory)
            receipt = self.receipt(intent["peer_id"], directory.name)
            if receipt["status"] == "receiving":
                continue
            size = stored_bytes(directory)
            self.audit.append(request_id=request_id, phase="cleanup", peer_id=intent["peer_id"],
                              operation_id=directory.name, detail={"discarded_input_bytes": size, "status": receipt["status"]})
            shutil.rmtree(directory)
            flush_directory(self.incoming)

    @staticmethod
    def load_intent(directory):
        fd = os.open(directory / ".intent.json", os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as source:
            value = source.read(4097)
        if len(value) > 4096:
            raise PeerProtocolError("Stored transfer intent exceeds bounds")
        intent = json.loads(value, object_pairs_hook=unique_object)
        if set(intent) != {"peer_id", "manifest_sha256"}:
            raise PeerProtocolError("Unexpected stored transfer intent")
        return intent

    def manifest(self, peer, operation_id):
        directory = self.incoming / operation_id
        if directory.is_symlink() or not directory.is_dir():
            raise PeerProtocolError("Incoming transfer is unavailable")
        intent = self.load_intent(directory)
        if intent["peer_id"] != peer:
            raise AdmissionError("Incoming transfer belongs to another peer")
        fd = os.open(directory / ".manifest.json", os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as source:
            data = source.read(MAX_MANIFEST_BYTES + 1)
        if hashlib.sha256(data).hexdigest() != intent["manifest_sha256"]:
            raise PeerProtocolError("Stored transfer manifest changed")
        policy = self.policy.body
        envelope = verify_manifest(data, self.policy.release_keys, require_trained=policy["require_trained"],
                                   max_payload_bytes=policy["max_payload_bytes"])
        return directory, data, envelope

    def reserve(self, peer, message, request_id):
        exact_fields(message, {"operation_id", "observed_revision", "admission_ticket", "manifest"})
        try:
            manifest = base64.b64decode(message["manifest"], validate=True)
        except (TypeError, ValueError) as exc:
            raise PeerProtocolError("Invalid manifest encoding") from exc
        birth = self.state.reserve_birth(peer, message["operation_id"], message["observed_revision"], manifest,
                                         admission_ticket=message["admission_ticket"])
        with self.transfer_lock():
            self.cleanup_inactive(request_id)
            if birth["status"] != "receiving":
                return self.receipt(peer, message["operation_id"])
            self.active_transfer(peer, message["operation_id"])
            directory = self.incoming / message["operation_id"]
            if directory.exists():
                _, existing, _ = self.manifest(peer, message["operation_id"])
                if existing != manifest:
                    raise PeerProtocolError("Retried manifest differs from the reserved transfer")
            else:
                if stored_bytes(self.incoming) + len(manifest) + birth["payload_bytes"] > self.policy.body["max_payload_bytes"] + MAX_MANIFEST_BYTES:
                    raise AdmissionError("Incoming-transfer storage allowance exhausted")
                temporary = Path(tempfile.mkdtemp(prefix=".session-", dir=self.incoming))
                try:
                    for name, data in {
                        ".manifest.json": manifest,
                        ".intent.json": canonical({"peer_id": peer, "manifest_sha256": birth["manifest_sha256"]}),
                    }.items():
                        with (temporary / name).open("xb") as target:
                            target.write(data)
                            target.flush()
                            os.fsync(target.fileno())
                    flush_directory(temporary)
                    os.rename(temporary, directory)
                    flush_directory(self.incoming)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            return self.receipt(peer, message["operation_id"])

    def receive_payload(self, connection, peer, message, request_id):
        exact_fields(message, {"operation_id", "payload", "bytes"})
        operation_id = message["operation_id"]
        with self.transfer_lock():
            self.active_transfer(peer, operation_id)
            directory, _, envelope = self.manifest(peer, operation_id)
            name = message["payload"]
            if not isinstance(name, str) or name not in envelope["body"]["files"]:
                raise PeerProtocolError("Payload is not in the signed manifest")
            expected = envelope["body"]["files"][name]
            if type(message["bytes"]) is not int or message["bytes"] != expected["bytes"]:
                raise PeerProtocolError("Payload size differs from its signed declaration")
            target = directory / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if os.path.lexists(target):
                if file_identity(target) != expected:
                    raise PeerProtocolError("Previously accepted input payload is corrupt")
                return {"stored": True, "already_present": True, "payload": name, **expected}
            temporary = target.with_name(target.name + ".part")
            if os.path.lexists(temporary):
                partial = file_identity(temporary)
                self.audit.append(request_id=request_id, phase="partial_discarded", peer_id=peer,
                                  operation_id=operation_id, detail={"payload": name, **partial})
                temporary.unlink()
            if stored_bytes(self.incoming) + expected["bytes"] > self.policy.body["max_payload_bytes"] + MAX_MANIFEST_BYTES:
                raise AdmissionError("Incoming-transfer storage allowance exhausted")
            self.reply(connection, {"ready_for_payload": True, "bytes": expected["bytes"]})
            deadline = time.monotonic() + payload_seconds(expected["bytes"], self.policy.body)
            size, digest, checked_at = 0, hashlib.sha256(), 0.0
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as output:
                while size < expected["bytes"]:
                    if self._stopping.is_set():
                        raise AdmissionError("Peer service is stopping")
                    if time.monotonic() - checked_at >= 0.25:
                        self.active_transfer(peer, operation_id)
                        checked_at = time.monotonic()
                    block = read_exact(connection, min(BUFFER_BYTES, expected["bytes"] - size), deadline)
                    output.write(block)
                    digest.update(block)
                    size += len(block)
                output.flush()
                os.fsync(output.fileno())
            if digest.hexdigest() != expected["sha256"]:
                raise CapsuleError("Received payload hash differs from the signed manifest")
            self.active_transfer(peer, operation_id)
            os.rename(temporary, target)
            flush_directory(target.parent)
            return {"stored": True, "already_present": False, "payload": name, "bytes": size, "sha256": digest.hexdigest()}

    def dispatch(self, connection, peer, message, request_id):
        if message.get("protocol") != PROTOCOL or message.get("policy_id") != self.policy.policy_id:
            raise PeerProtocolError("Peer protocol or enrollment policy differs")
        operation = message.get("operation")
        if operation == "self_status":
            exact_fields(message, set())
            return self.state.self_receipt(peer)
        if operation == "observe":
            exact_fields(message, set())
            return self.state.advertisement()
        if operation == "reserve":
            return self.reserve(peer, message, request_id)
        if operation == "put":
            return self.receive_payload(connection, peer, message, request_id)
        if operation == "status":
            exact_fields(message, {"operation_id"})
            return self.receipt(peer, message["operation_id"])
        if operation == "cancel":
            exact_fields(message, {"operation_id", "reason"})
            self.state.abort_birth(peer, message["operation_id"], message["reason"])
            with self.transfer_lock():
                self.cleanup_inactive(request_id)
            return self.receipt(peer, message["operation_id"])
        if operation == "commit":
            exact_fields(message, {"operation_id"})
            receipt = self.receipt(peer, message["operation_id"])
            if receipt["status"] != "receiving":
                return receipt
            with self.transfer_lock():
                self.active_transfer(peer, message["operation_id"])
                directory, manifest, _ = self.manifest(peer, message["operation_id"])
                # The reservation already exists. An expired original observation
                # does not renew or duplicate it; no new ticket is minted here.
                result = self.receiver.receive(peer, message["operation_id"], 0, manifest, directory,
                                               admission_ticket="")
                self.cleanup_inactive(request_id)
                return result
        raise PeerProtocolError("Operation is not exposed by the peer interface")


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        server = self.server
        request_id = uuid.uuid4().hex
        peer = operation = operation_id = None
        connection = None
        completed = False
        started = False
        try:
            self.request.settimeout(CONTROL_SECONDS)
            connection = server.tls.wrap_socket(self.request, server_side=True, do_handshake_on_connect=False)
            with server._connections_lock:
                server._connections.discard(self.request)
                server._connections.add(connection)
            connection.do_handshake()
            peer, certificate_hash = authenticate(connection, server.policy)
            message = receive_control(connection)
            operation = message.get("operation")
            if (peer == server.state.device_id) != (operation == "self_status"):
                raise PeerIdentityError("Only self_status is available through this device's own identity")
            operation_id = message.get("operation_id")
            # Bound logged identifiers before appending any request-supplied text.
            if not isinstance(operation, str) or len(operation) > 32:
                raise PeerProtocolError("Invalid peer operation name")
            if operation_id is not None and (not isinstance(operation_id, str) or not ID.fullmatch(operation_id)):
                raise PeerProtocolError("Invalid birth operation identity")
            server.audit.append(request_id=request_id, phase="started", peer_id=peer, operation=operation,
                                operation_id=operation_id, detail={"certificate_sha256": certificate_hash, "tls": connection.version()})
            started = True
            result = server.dispatch(connection, peer, message, request_id)
            detail = {name: result[name] for name in ("status", "release_id", "payload", "bytes", "sha256") if name in result}
            # Bind the public response to its receiver-owned transport history
            # without retaining admission tickets or credential material.
            detail["response_sha256"] = hashlib.sha256(canonical(
                {key: value for key, value in result.items() if key != "admission_ticket"})).hexdigest()
            anchor = server.audit.append(request_id=request_id, phase="completed", peer_id=peer, operation=operation,
                                         operation_id=operation_id, detail=detail)
            completed = True
            server.reply(connection, {**result, "transport_receipt": anchor})
        except Exception as exc:
            # A rejected operation never receives an affirmative completion.
            detail = str(exc)[:384]
            try:
                phase = "response_lost" if completed else ("failed" if started else "rejected")
                server.audit.append(request_id=request_id, phase=phase, peer_id=peer,
                                    operation=operation if isinstance(operation, str) and len(operation) <= 32 else None,
                                    operation_id=operation_id if isinstance(operation_id, str) and ID.fullmatch(operation_id) else None,
                                    detail={"error_type": type(exc).__name__, "message": detail})
            except (OSError, ValueError, sqlite3.Error):
                pass
            if connection is not None:
                try:
                    server.reply(connection, error=detail)
                except (OSError, ValueError):
                    pass
        finally:
            if connection is not None:
                with server._connections_lock:
                    server._connections.discard(connection)
                connection.close()


class PeerClient:
    def __init__(self, policy, identity, identity_directory, *, network=None, identity_owner_uid=None, identity_read_group=None):
        self.policy, self.identity = policy, identity
        self.network = network or LabNetwork()
        self.tls = context(policy, identity, identity_directory, owner_uid=identity_owner_uid, read_group=identity_read_group)

    @contextmanager
    def connect(self, device_id, address, port):
        self.policy.peer_key(device_id)
        if device_id == self.identity or type(port) is not int or not 1 <= port <= 65535:
            raise PeerIdentityError("Invalid peer destination")
        address = self.network.address(address)
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.settimeout(CONTROL_SECONDS)
        connection = None
        try:
            raw.connect((address, port))
            connection = self.tls.wrap_socket(raw, server_hostname=device_hostname(device_id))
            authenticate(connection, self.policy, expected=device_id)
            yield connection
        finally:
            if connection is not None:
                connection.close()
            else:
                raw.close()

    def response(self, connection, *, timeout=CONTROL_SECONDS):
        value = receive_control(connection, timeout=timeout)
        if set(value) != {"protocol", "policy_id", "ok", "result", "error"} or value["protocol"] != PROTOCOL or value["policy_id"] != self.policy.policy_id:
            raise PeerProtocolError("Unexpected peer response")
        if value["ok"] is not True:
            raise PeerProtocolError(str(value["error"])[:384])
        if not isinstance(value["result"], dict):
            raise PeerProtocolError("Peer response is not an object")
        return value["result"]

    def request(self, endpoint, operation, **fields):
        with self.connect(*endpoint) as connection:
            send_control(connection, {"protocol": PROTOCOL, "policy_id": self.policy.policy_id,
                                      "operation": operation, **fields})
            result = self.response(connection, timeout=TRANSFER_SECONDS if operation == "commit" else CONTROL_SECONDS)
            self.check_anchor(result)
            return result

    @staticmethod
    def check_anchor(result):
        anchor = result.get("transport_receipt")
        if (not isinstance(anchor, dict) or set(anchor) != {"sequence", "digest"}
                or type(anchor["sequence"]) is not int or not 1 <= anchor["sequence"] <= MAX_RECORDS
                or not isinstance(anchor["digest"], str) or not re.fullmatch(r"[0-9a-f]{64}", anchor["digest"])):
            raise PeerProtocolError("Completed peer response lacks its audit identity")

    def observe(self, endpoint):
        return self.request(endpoint, "observe")

    def reserve(self, endpoint, operation_id, manifest, observation):
        return self.request(endpoint, "reserve", operation_id=operation_id, manifest=base64.b64encode(manifest).decode(),
                            observed_revision=observation["revision"], admission_ticket=observation["admission_ticket"])

    def upload(self, endpoint, operation_id, name, expected, source):
        with self.connect(*endpoint) as connection:
            send_control(connection, {"protocol": PROTOCOL, "policy_id": self.policy.policy_id, "operation": "put",
                                      "operation_id": operation_id, "payload": name, "bytes": expected["bytes"]})
            result = self.response(connection, timeout=TRANSFER_SECONDS)
            if result.get("stored") is True:
                if result.get("payload") != name or result.get("sha256") != expected["sha256"] or result.get("bytes") != expected["bytes"]:
                    raise PeerProtocolError("Receiver's existing payload differs from the signed input")
                self.check_anchor(result)
                return result
            if result != {"ready_for_payload": True, "bytes": expected["bytes"]}:
                raise PeerProtocolError("Receiver did not accept the declared transfer")
            source_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with _read_beneath(source_fd, name) as input_file:
                    if os.fstat(input_file.fileno()).st_size != expected["bytes"]:
                        raise CapsuleError("Source payload length changed")
                    deadline = time.monotonic() + payload_seconds(expected["bytes"], self.policy.body)
                    sent, digest = 0, hashlib.sha256()
                    while sent < expected["bytes"]:
                        left = deadline - time.monotonic()
                        if left <= 0:
                            raise TimeoutError("Peer upload deadline exceeded")
                        block = input_file.read(min(BUFFER_BYTES, expected["bytes"] - sent))
                        if not block:
                            raise CapsuleError("Source payload was truncated")
                        connection.settimeout(min(CONTROL_SECONDS, left))
                        connection.sendall(block)
                        sent += len(block)
                        digest.update(block)
                    if digest.hexdigest() != expected["sha256"]:
                        raise CapsuleError("Source payload hash changed")
            finally:
                os.close(source_fd)
            result = self.response(connection, timeout=TRANSFER_SECONDS)
            if (result.get("stored") is not True or result.get("payload") != name
                    or result.get("sha256") != expected["sha256"] or result.get("bytes") != expected["bytes"]):
                raise PeerProtocolError("Receiver did not acknowledge the signed payload")
            self.check_anchor(result)
            return result

    def transfer(self, endpoint, operation_id, manifest, source, *, observation=None):
        policy = self.policy.body
        envelope = verify_manifest(manifest, self.policy.release_keys, require_trained=policy["require_trained"],
                                   max_payload_bytes=policy["max_payload_bytes"])
        observation = observation or self.observe(endpoint)
        result = self.reserve(endpoint, operation_id, manifest, observation)
        if result["status"] != "receiving":
            return result
        for name, expected in envelope["body"]["files"].items():
            self.upload(endpoint, operation_id, name, expected, source)
        return self.request(endpoint, "commit", operation_id=operation_id)
