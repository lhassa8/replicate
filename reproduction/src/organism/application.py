"""Persistent local diagnostics and discovery; reproduction decisions are separate."""

import asyncio
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import stat

from .application_store import ApplicationStore, JobError
from .capsule import canonical
from .diagnostic_agent import assess
from .equipment import EquipmentDirectory
from .equipment_diagnostic_agent import diagnose
from .equipment_interfaces import InterfaceDirectory
from .source_journal import SourceJournal
from .inference_server import InferenceServer


MAX_RPC_BYTES = 32768
MAX_CLIENTS = 4


class AgentApplication:
    def __init__(self, config, work, *, inference_factory=InferenceServer, assessor=assess,
                 diagnoser=diagnose, discovery=None, reproduction=None, experience=None):
        self.config, self.work = config, Path(work).absolute()
        info = self.work.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("Application work directory must be private to its service account")
        self.socket = self.work / "agent.sock"
        if len(os.fsencode(self.socket)) >= 104:
            raise ValueError("Application socket path exceeds the portable Unix socket bound")
        self.inference = inference_factory(config, self.work)
        self.assessor = assessor
        self.diagnoser = diagnoser
        self.store, self.observations = None, None
        self.equipment_interfaces = None
        self.source_journal = None
        self.lock_fd, self.listener = None, None
        self.handlers, self.job_tasks = set(), set()
        self.busy, self.closed = False, False
        self.failure = None
        if discovery is not None and (discovery.local_identity != config.device_id or discovery.start_token != config.start_token):
            raise ValueError("Discovery belongs to another device or incarnation")
        self.discovery, self.discovery_task = discovery, None
        if experience is not None and experience.config != config:
            raise ValueError("Experience exchange belongs to another local application")
        self.experience, self.experience_task = experience, None
        if reproduction is not None and (discovery is None or reproduction.discovery is not discovery
                                          or reproduction.config != config):
            raise ValueError("Reproduction belongs to another local application")
        self.reproduction, self.reproduction_task = reproduction, None
        self.stop_event = asyncio.Event()
        self.receipt = {"started_at": datetime.now(timezone.utc).isoformat(),
                        "start_token": config.start_token, "device_id": config.device_id,
                        "individual_id": config.individual_id, "release_id": config.release_id,
                        "native": config.native, "trained_for_organism": config.trained_for_organism,
                        "useful_readiness_verified": False, "autonomous_reproduction_verified": False,
                        "experience_configured": experience is not None, "qualification_passed": False}

    def write_receipt(self):
        path = self.work / ("application-" + self.config.start_token + ".json")
        temporary = self.work / ".application-receipt.pending"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        with os.fdopen(fd, "wb") as target:
            info = os.fstat(target.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_nlink != 1 or info.st_mode & 0o077):
                raise ValueError("Application receipt is not an owned regular file")
            os.ftruncate(target.fileno(), 0)
            target.write(canonical(self.receipt))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory = os.open(self.work, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    async def start(self):
        if self.lock_fd is not None:
            raise RuntimeError("Application has already acquired its work directory")
        fd = os.open(self.work / ".application.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1 or info.st_mode & 0o077:
                raise ValueError("Application lock is not private")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise
        self.lock_fd = fd
        self.store = ApplicationStore(self.work, self.config.start_token)
        (self.work / "equipment").mkdir(mode=0o700, exist_ok=True)
        self.observations = EquipmentDirectory(self.work / "equipment")
        (self.work / "equipment-interfaces").mkdir(mode=0o700, exist_ok=True)
        self.equipment_interfaces = InterfaceDirectory(self.work / "equipment-interfaces")
        self.source_journal = SourceJournal(self.work / "source-evidence", self.config)
        if self.experience is not None:
            self.experience.bind_sources(self.source_journal)
        if os.path.lexists(self.socket):
            info = self.socket.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid():
                raise ValueError("Unexpected existing application socket path")
            self.socket.unlink()
        self.write_receipt()
        self.receipt["inference"] = await self.inference.start()
        self.write_receipt()
        if self.experience is not None:
            await self.experience.start()
        self.listener = await asyncio.start_unix_server(self.handle, path=str(self.socket), limit=MAX_RPC_BYTES + 1)
        self.socket.chmod(0o600)
        if self.discovery is not None:
            self.discovery_task = asyncio.create_task(self.discovery.run())
            self.discovery_task.add_done_callback(self.discovery_done)
        if self.reproduction is not None:
            self.reproduction_task = asyncio.create_task(self.reproduction.run(self))
            self.reproduction_task.add_done_callback(self.reproduction_done)
        if self.experience is not None:
            self.experience_task = asyncio.create_task(self.experience.run())
            self.experience_task.add_done_callback(self.experience_done)
        return self.health()

    def health(self):
        process = self.inference.process
        ready = (self.inference.ready and (process is None or process.returncode is None)
                 and not self.inference.failure and not self.closed)
        return {"status": "serving" if ready else "unavailable", "inference_ready": ready,
                "busy": self.busy, "start_token": self.config.start_token,
                "trained_for_organism": self.config.trained_for_organism,
                "native": self.config.native, "useful_readiness_verified": False,
                "discovery_running": self.discovery_task is not None and not self.discovery_task.done(),
                "reproduction_running": self.reproduction_task is not None and not self.reproduction_task.done(),
                "experience_running": self.experience_task is not None and not self.experience_task.done()}

    def experience_done(self, task):
        if self.closed:
            return
        self.failure = ("Experience exchange stopped unexpectedly" if task.cancelled() or task.exception() is None else
                        "Experience exchange failed: " + str(task.exception())[:512])
        self.stop_event.set()

    def reproduction_done(self, task):
        if self.closed:
            return
        self.failure = ("Reproduction stopped unexpectedly" if task.cancelled() or task.exception() is None else
                        "Reproduction failed: " + str(task.exception())[:512])
        self.stop_event.set()

    def discovery_done(self, task):
        if self.closed:
            return
        self.failure = ("Discovery stopped unexpectedly" if task.cancelled() or task.exception() is None else
                        "Discovery failed: " + str(task.exception())[:512])
        self.stop_event.set()

    async def execute(self, request_id, device_id, task_kind="temperature_v1"):
        capture = None
        try:
            if task_kind == "equipment_interface_v1":
                directory = self.equipment_interfaces
                if self.source_journal is not None:
                    capture = self.source_journal.begin(request_id, device_id)
                    if capture is not None:
                        directory = InterfaceDirectory(directory.directory, clock=directory.clock,
                                                       observer=capture.observe)
                experience_store = self.experience.store if self.experience is not None else None
                if self.experience is not None and capture is not None:
                    experience_store = await self.experience.prepare_source(capture)
                arguments = {} if experience_store is None else {"experience_store": experience_store}
                await self.diagnoser(self.inference, directory, self.store, request_id, device_id, **arguments)
            else:
                await self.assessor(self.inference, self.observations, self.store, request_id, device_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            record = self.store.get(request_id)
            if record and record["record"]["status"] == "running":
                self.store.update(request_id, trace=record["record"]["trace"], status="failed",
                                  failure=type(exc).__name__ + ": " + str(exc)[:512])
        finally:
            try:
                if capture is not None:
                    self.source_journal.finish(capture, self.store.get(request_id))
            finally:
                self.busy = False

    async def dispatch(self, request):
        if not isinstance(request, dict):
            raise ValueError("Invalid local request")
        operation = request.get("operation")
        fields = {"health": {"operation"}, "peers": {"operation"}, "offspring": {"operation"}, "status": {"operation", "request_id"},
                  "experience": {"operation"}, "sources": {"operation"},
                  "assess": {"operation", "request_id", "device_id"},
                  "diagnose": {"operation", "request_id", "device_id"}}
        if operation not in fields or set(request) != fields[operation]:
            raise ValueError("Unknown local operation or fields")
        if operation == "health":
            return self.health()
        if operation == "experience":
            return self.experience.snapshot() if self.experience is not None else {"enabled": False}
        if operation == "sources":
            return self.source_journal.snapshot() if self.source_journal is not None else {"enabled": False}
        if operation == "peers":
            return {"status": "observed" if self.discovery is not None else "disabled",
                    "peers": self.discovery.snapshot() if self.discovery is not None else [],
                    "autonomous_reproduction_verified": False}
        if operation == "offspring":
            return {"status": "observed" if self.reproduction is not None else "disabled",
                    "reproduction": self.reproduction.public() if self.reproduction is not None else None,
                    "autonomous_reproduction_verified": False}
        if operation == "status":
            return self.store.public(self.store.get(request["request_id"]))
        task_kind = "equipment_interface_v1" if operation == "diagnose" else "temperature_v1"
        existing = self.store.get(request["request_id"])
        if existing:
            if existing["record"]["device_id"] != request["device_id"]:
                raise JobError("Request ID belongs to another device")
            if existing["record"].get("task_kind", "temperature_v1") != task_kind:
                raise JobError("Request ID belongs to another diagnostic interface")
            return self.store.public(existing)
        if self.busy or not self.health()["inference_ready"]:
            return {"status": "busy_or_unavailable"}
        value, fresh = self.store.reserve(request["request_id"], request["device_id"], task_kind=task_kind)
        if fresh:
            self.busy = True
            task = asyncio.create_task(self.execute(request["request_id"], request["device_id"], task_kind))
            self.job_tasks.add(task)
            task.add_done_callback(self.job_done)
        return self.store.public(value)

    def job_done(self, task):
        self.job_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self.failure = "Diagnostic journal or completion failed: " + str(task.exception())[:512]
            self.stop_event.set()

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        if len(self.handlers) >= MAX_CLIENTS or self.closed:
            writer.close()
            await writer.wait_closed()
            return
        self.handlers.add(task)
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=5)
            if not raw.endswith(b"\n") or len(raw) > MAX_RPC_BYTES:
                raise ValueError("Local request frame exceeds bounds")
            response = await self.dispatch(json.loads(raw))
            data = canonical(response) + b"\n"
            if len(data) > MAX_RPC_BYTES:
                raise ValueError("Local response exceeds bounds")
            writer.write(data)
            await asyncio.wait_for(writer.drain(), timeout=5)
        except (ValueError, OSError, asyncio.TimeoutError):
            try:
                writer.write(b'{"status":"rejected"}\n')
                await asyncio.wait_for(writer.drain(), timeout=1)
            except (OSError, asyncio.TimeoutError):
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            self.handlers.discard(task)

    async def run(self):
        try:
            await self.start()
            while not self.stop_event.is_set():
                if not self.health()["inference_ready"]:
                    raise RuntimeError("Owned model process became unavailable")
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass
            if self.failure:
                raise RuntimeError(self.failure)
        except BaseException as exc:
            self.receipt["failure"] = type(exc).__name__ + ": " + str(exc)[:512]
            raise
        finally:
            await self.close()

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.experience is not None:
            if self.experience_task is not None:
                self.experience_task.cancel()
                await asyncio.gather(self.experience_task, return_exceptions=True)
            await self.experience.close()
        if self.reproduction is not None:
            self.reproduction.closed = True
            if self.reproduction_task is not None:
                self.reproduction_task.cancel()
                await asyncio.gather(self.reproduction_task, return_exceptions=True)
        if self.discovery is not None:
            await self.discovery.close()
            if self.discovery_task is not None:
                await asyncio.gather(self.discovery_task, return_exceptions=True)
        if self.listener:
            self.listener.close()
            await self.listener.wait_closed()
        for task in list(self.handlers | self.job_tasks):
            task.cancel()
        if self.handlers or self.job_tasks:
            await asyncio.gather(*list(self.handlers | self.job_tasks), return_exceptions=True)
        try:
            self.receipt["shutdown"] = await self.inference.stop()
            self.receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
            if self.lock_fd is not None:
                self.write_receipt()
        finally:
            if self.lock_fd is not None:
                if os.path.lexists(self.socket) and self.socket.is_socket():
                    self.socket.unlink()
                os.close(self.lock_fd)
                self.lock_fd = None


async def local_request(socket_path, value, *, timeout=10):
    reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(str(socket_path), limit=MAX_RPC_BYTES + 1),
                                            timeout=timeout)
    try:
        data = canonical(value) + b"\n"
        if len(data) > MAX_RPC_BYTES:
            raise ValueError("Local request exceeds bounds")
        writer.write(data)
        await asyncio.wait_for(writer.drain(), timeout=timeout)
        raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not raw.endswith(b"\n") or len(raw) > MAX_RPC_BYTES:
            raise ValueError("Invalid local response frame")
        return json.loads(raw)
    finally:
        writer.close()
        await writer.wait_closed()
