"""Cancellable TLS operations for an organism's own reproduction controller."""

from contextlib import asynccontextmanager
import asyncio
import base64
import hashlib
import json
import os
import struct

from .capsule import CapsuleError, _read_beneath, canonical, verify_manifest
from .enrollment import unique_object
from .peer_identity import PeerIdentityError, PROTOCOL, authenticate, device_hostname
from .peer_transport import (
    BUFFER_BYTES, CONTROL_SECONDS, MAX_CONTROL, TRANSFER_SECONDS,
    PeerClient, PeerProtocolError, payload_seconds,
)


class AsyncPeerClient(PeerClient):
    """Same framed protocol; cancellation closes the owned connection."""

    @asynccontextmanager
    async def connection(self, endpoint, *, self_status=False):
        identity, address, port = endpoint
        self.policy.peer_key(identity)
        if (identity == self.identity) != self_status or type(port) is not int or not 1 <= port <= 65535:
            raise PeerIdentityError("Invalid async peer destination or local operation")
        address = self.network.address(address)
        writer = None
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(
                address, port, ssl=self.tls, server_hostname=device_hostname(identity), limit=MAX_CONTROL + 4),
                timeout=CONTROL_SECONDS)
            authenticate(writer.get_extra_info("ssl_object"), self.policy, expected=identity)
            yield reader, writer
        finally:
            if writer is not None:
                writer.close()
                try:
                    await asyncio.wait_for(writer.wait_closed(), timeout=1)
                except (OSError, TimeoutError):
                    pass

    async def send(self, writer, operation, **fields):
        raw = canonical({"protocol": PROTOCOL, "policy_id": self.policy.policy_id, "operation": operation, **fields})
        if len(raw) > MAX_CONTROL:
            raise PeerProtocolError("Async control frame exceeds its bound")
        writer.write(struct.pack("!I", len(raw)) + raw)
        await asyncio.wait_for(writer.drain(), timeout=CONTROL_SECONDS)

    async def receive(self, reader, *, timeout=CONTROL_SECONDS, anchored=True):
        async with asyncio.timeout(timeout):
            count = struct.unpack("!I", await reader.readexactly(4))[0]
            if not 0 < count <= MAX_CONTROL:
                raise PeerProtocolError("Async response exceeds its bound")
            value = json.loads(await reader.readexactly(count), object_pairs_hook=unique_object)
        if (not isinstance(value, dict) or set(value) != {"protocol", "policy_id", "ok", "result", "error"}
                or value["protocol"] != PROTOCOL or value["policy_id"] != self.policy.policy_id):
            raise PeerProtocolError("Unexpected async response envelope")
        if value["ok"] is not True or value["error"] is not None:
            raise PeerProtocolError(str(value["error"])[:384])
        if not isinstance(value["result"], dict):
            raise PeerProtocolError("Async response is not an object")
        result = value["result"]
        if anchored:
            self.check_anchor(result)
        return result

    async def request(self, endpoint, operation, **fields):
        if operation not in ("observe", "self_status", "reserve", "status", "commit", "cancel"):
            raise PeerProtocolError("Unsupported async control operation")
        async with self.connection(endpoint, self_status=operation == "self_status") as (reader, writer):
            await self.send(writer, operation, **fields)
            return await self.receive(reader, timeout=TRANSFER_SECONDS if operation == "commit" else CONTROL_SECONDS)

    async def observe(self, endpoint):
        return await self.request(endpoint, "observe")

    async def reserve(self, endpoint, operation_id, manifest, observation):
        return await self.request(endpoint, "reserve", operation_id=operation_id,
                                  manifest=base64.b64encode(manifest).decode(),
                                  observed_revision=observation["revision"], admission_ticket=observation["admission_ticket"])

    async def upload(self, endpoint, operation_id, name, expected, source):
        async with self.connection(endpoint) as (reader, writer):
            await self.send(writer, "put", operation_id=operation_id, payload=name, bytes=expected["bytes"])
            response = await self.receive(reader, timeout=TRANSFER_SECONDS, anchored=False)
            if response.get("stored") is not True:
                if response != {"ready_for_payload": True, "bytes": expected["bytes"]}:
                    raise PeerProtocolError("Receiver did not accept the declared async transfer")
                # Match the synchronous sender's signed-size payload allowance.
                # This clock is absolute; progress does not reset it. The receiver
                # still enforces the original shared birth deadline and owner state.
                async with asyncio.timeout(payload_seconds(expected["bytes"], self.policy.body)):
                    directory = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        with _read_beneath(directory, name) as input_file:
                            if os.fstat(input_file.fileno()).st_size != expected["bytes"]:
                                raise CapsuleError("Async source payload length changed")
                            count, digest = 0, hashlib.sha256()
                            while count < expected["bytes"]:
                                block = input_file.read(min(BUFFER_BYTES, expected["bytes"] - count))
                                if not block:
                                    raise CapsuleError("Async source payload was truncated")
                                writer.write(block)
                                await asyncio.wait_for(writer.drain(), timeout=CONTROL_SECONDS)
                                count += len(block)
                                digest.update(block)
                                # Cancellation remains possible even on an
                                # uncongested local socket with a large payload.
                                await asyncio.sleep(0)
                            if digest.hexdigest() != expected["sha256"]:
                                raise CapsuleError("Async source payload hash changed")
                    finally:
                        os.close(directory)
                response = await self.receive(reader, timeout=TRANSFER_SECONDS)
            self.check_anchor(response)
            if (response.get("stored") is not True or response.get("payload") != name
                    or response.get("sha256") != expected["sha256"] or response.get("bytes") != expected["bytes"]):
                raise PeerProtocolError("Receiver did not acknowledge the exact async payload")
            return response

    async def transfer(self, endpoint, operation_id, manifest, source, *, observation=None):
        policy = self.policy.body
        envelope = verify_manifest(manifest, self.policy.release_keys, require_trained=policy["require_trained"],
                                   max_payload_bytes=policy["max_payload_bytes"])
        observation = observation or await self.observe(endpoint)
        result = await self.reserve(endpoint, operation_id, manifest, observation)
        if result["status"] != "receiving":
            return result
        for name, expected in envelope["body"]["files"].items():
            await self.upload(endpoint, operation_id, name, expected, source)
        return await self.request(endpoint, "commit", operation_id=operation_id)
