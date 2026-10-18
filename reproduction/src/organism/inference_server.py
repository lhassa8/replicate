"""One owned, authenticated loopback llama.cpp process with bounded output."""

import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import time

import httpx

from .runtime_environment import library_environment


MAX_LOG_BYTES = 16 * 1024**2


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@dataclass(frozen=True)
class RuntimeConfig:
    model: Path
    model_sha256: str
    model_bytes: int
    executable: Path
    executable_sha256: str
    start_token: str
    device_id: str
    individual_id: str
    release_id: str
    trained_for_organism: bool
    native: bool
    library_paths: tuple[Path, ...] = ()

    def verify(self):
        for value, length in ((self.start_token, 32), (self.device_id, 64), (self.individual_id, 32),
                              (self.release_id, 64), (self.model_sha256, 64), (self.executable_sha256, 64)):
            if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{" + str(length) + "}", value):
                raise ValueError("Invalid runtime identity")
        for path, expected in ((self.model, self.model_sha256), (self.executable, self.executable_sha256)):
            info = path.lstat()
            if not path.is_absolute() or not stat.S_ISREG(info.st_mode) or digest(path) != expected:
                raise ValueError("Runtime input identity changed")
        if self.model.stat().st_size != self.model_bytes or not os.access(self.executable, os.X_OK):
            raise ValueError("Runtime model size or executable permission differs")
        if any(not path.is_absolute() or ":" in str(path) or not path.is_dir() for path in self.library_paths):
            raise ValueError("Invalid native library directory")


class InferenceServer:
    def __init__(self, config, work):
        self.config, self.work = config, Path(work)
        self.alias = "organism-" + config.start_token
        self.api_key = secrets.token_urlsafe(32)
        self.headers = {"Authorization": "Bearer " + self.api_key}
        self.process, self.log_task = None, None
        self.failure, self.ready = None, False
        self.base_url, self.properties = None, None
        self.client = None

    async def _drain_log(self, log):
        count = 0
        stopped = False

        def fail(reason):
            nonlocal stopped
            self.failure = reason
            if not stopped and self.process.returncode is None:
                self.process.terminate()
                stopped = True

        try:
            while block := await self.process.stdout.read(16384):
                if self.failure:
                    # Keep draining the pipe after a stop request, without
                    # retaining more bytes. Otherwise a full pipe can prevent
                    # subprocess.wait() from observing completed shutdown.
                    continue
                try:
                    if count + len(block) > MAX_LOG_BYTES:
                        fail("Inference log allowance exhausted")
                    else:
                        log.write(block)
                        count += len(block)
                except OSError as exc:
                    fail("Inference log write failed: " + type(exc).__name__)
        finally:
            try:
                log.flush()
                os.fsync(log.fileno())
            except OSError as exc:
                fail("Inference log flush failed: " + type(exc).__name__)
            finally:
                log.close()

    async def json_get(self, path, *, maximum=1024**2):
        data = bytearray()
        async with self.client.stream("GET", path) as response:
            response.raise_for_status()
            async for block in response.aiter_bytes():
                data.extend(block)
                if len(data) > maximum:
                    raise ValueError("Inference server metadata exceeds its bound")
        return json.loads(data)

    async def start(self):
        self.config.verify()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        command = [
            str(self.config.executable), "--model", str(self.config.model), "--alias", self.alias,
            "--host", "127.0.0.1", "--port", str(port), "--offline",
            "--ctx-size", "4096", "--parallel", "1", "--gpu-layers", "0", "--device", "none",
            "--no-kv-offload", "--no-op-offload", "--threads", "4", "--threads-batch", "4",
            "--poll", "0", "--poll-batch", "0", "--cache-ram", "0", "--jinja",
            "--chat-template-kwargs", '{"enable_thinking":false}',
            "--temp", "0", "--top-p", "1", "--top-k", "0", "--min-p", "0", "--repeat-penalty", "1.0",
            "--no-webui", "--no-ui-mcp-proxy", "--no-slots",
        ]
        environment = {
            "PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LC_ALL": "C.UTF-8",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OTEL_SDK_DISABLED": "true",
            "AWS_EC2_METADATA_DISABLED": "true", "LLAMA_API_KEY": self.api_key,
        }
        environment.update(library_environment(self.config.library_paths))
        log = (self.work / ("model-" + self.config.start_token + ".log")).open("xb")
        try:
            self.process = await asyncio.create_subprocess_exec(
                *command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, env=environment, limit=65536,
            )
        except BaseException:
            log.close()
            raise
        self.log_task = asyncio.create_task(self._drain_log(log))
        self.client = httpx.AsyncClient(base_url=self.base_url, headers=self.headers, trust_env=False,
                                       timeout=httpx.Timeout(2), follow_redirects=False)
        deadline = time.monotonic() + 75
        while time.monotonic() < deadline:
            if self.process.returncode is not None or self.failure:
                raise RuntimeError(self.failure or "Owned inference server exited before readiness")
            try:
                if (await self.json_get("/health", maximum=4096)).get("status") == "ok":
                    self.properties = await self.json_get("/props")
                    break
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(0.25)
        else:
            raise TimeoutError("Owned inference server readiness timeout")
        props = self.properties
        if (props.get("model_alias") != self.alias or props.get("model_path") != str(self.config.model)
                or props.get("total_slots") != 1 or props.get("default_generation_settings", {}).get("n_ctx") != 4096
                or props.get("ui") is not False or props.get("cors_proxy_enabled") is not False):
            raise RuntimeError("Inference server identity or configuration differs")
        self.ready = True
        return {"pid": self.process.pid, "alias": self.alias, "command": command,
                "properties": props, "native_execution": self.config.native,
                "trained_for_organism": self.config.trained_for_organism,
                "useful_readiness_verified": False}

    async def stop(self):
        self.ready = False
        if self.client:
            await self.client.aclose()
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.process.kill()
                await asyncio.wait_for(self.process.wait(), timeout=5)
        if self.log_task:
            await asyncio.wait_for(self.log_task, timeout=5)
        return {"owned_server_stopped": self.process is None or self.process.returncode is not None,
                "server_pid": self.process.pid if self.process else None,
                "server_return_code": self.process.returncode if self.process else None,
                "failure": self.failure}
