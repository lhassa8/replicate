"""Ephemeral Mac TLS peers for a stock-model discovery probe, not native devices."""

import asyncio
import base64
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .discovery import Discovery, POLL_SECONDS, PROBE_SECONDS
from .discovery_book import DiscoveryBook, SCHEMA as BOOK_SCHEMA, sign_discovery_book
from .enrollment import Enrollment, SCHEMA, key_id, public_bytes, sign_enrollment
from .node_context import NodeContextProvisioner
from .peer_identity import device_certificate, owner_certificate, write_identity
from .peer_transport import PeerServer
from .receiver import LocalReceiver
from .receiver_state import ReceiverState


class ProbeHabitat:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="od-habitat-", dir="/private/tmp")
        self.root = Path(self.temporary.name)
        self.owner = Ed25519PrivateKey.generate()
        self.public = public_bytes(self.owner.public_key())
        self.keys = [Ed25519PrivateKey.generate() for _ in range(3)]
        self.ids = [key_id(public_bytes(key.public_key())) for key in self.keys]
        self.local_id, self.existing_id, self.new_id = self.ids
        release = Ed25519PrivateKey.generate()
        release_public = public_bytes(release.public_key())
        body = {
            "schema": SCHEMA, "experiment_id": "mac-discovery-development-not-native-qualification", "epoch": 1,
            "devices": {identity: {"public_key": base64.b64encode(public_bytes(key.public_key())).decode(),
                                  "platform": "linux-arm64" if index < 2 else "linux-x86_64"}
                        for index, (identity, key) in enumerate(zip(self.ids, self.keys))},
            "release_keys": {key_id(release_public): base64.b64encode(release_public).decode()},
            "require_trained": False, "max_payload_bytes": 8 * 1024**3, "birth_seconds": 1200,
            "max_births_per_device": 6, "max_process_starts": 3,
        }
        self.policy = Enrollment(sign_enrollment(body, self.owner), self.public)
        owner_cert = owner_certificate(self.owner)
        self.identities, self.receivers, self.servers = {}, {}, {}
        for identity, key in zip(self.ids, self.keys):
            cert = device_certificate(self.owner, owner_cert, key.public_key())
            self.identities[identity] = write_identity(self.root / ("identity-" + identity[:12]), key, cert, owner_cert)
        endpoints = {self.local_id: {"address": "127.0.0.1", "port": 7443}}
        for index, identity in enumerate((self.existing_id, self.new_id)):
            state = ReceiverState.enroll(self.root / ("state-" + str(index)), self.policy, identity,
                                         boot_id="mac-fixture-peer-" + str(index))
            receiver = LocalReceiver(state, self.root / ("capsules-" + str(index)))
            server = PeerServer(receiver, self.identities[identity])
            endpoints[identity] = {"address": server.server_address[0], "port": server.server_address[1]}
            self.receivers[identity] = receiver
            if identity == self.existing_id:
                self.servers[identity] = server.start()
            else:
                server.close()  # Its endpoint is unavailable until the recorded arrival.
        book_body = {"schema": BOOK_SCHEMA, "policy_id": self.policy.policy_id, "epoch": 1,
                     "lab_cidrs": ["127.0.0.0/24"], "endpoints": endpoints}
        self.book = DiscoveryBook(sign_discovery_book(book_body, self.owner, self.policy), self.policy, self.public)
        self.report = {"scope": "One real Mac stock-model application and two local TLS receiver fixtures",
                       "poll_seconds": POLL_SECONDS, "probe_seconds": PROBE_SECONDS,
                       "local_device_id": self.local_id, "existing_peer_id": self.existing_id, "new_peer_id": self.new_id,
                       "platform_fields_are_fixture_declarations": True, "native_execution_verified": False,
                       "autonomous_reproduction_verified": False, "private_keys_exported": False}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for server in self.servers.values():
            server.close()
        # Sealed contexts are this probe's own temporary files, not a release.
        for path in (self.root / "contexts").glob("*") if (self.root / "contexts").exists() else ():
            if path.is_dir():
                path.chmod(0o700)
        self.temporary.cleanup()

    def configure(self, config, work):
        import os
        config = replace(config, device_id=self.local_id)
        contexts = NodeContextProvisioner(self.root / "contexts", self.policy, self.local_id, self.public,
                                          self.book.signed, self.identities[self.local_id], group=os.getegid())
        node = contexts.prepare(config.individual_id, config.release_id)
        discovery = Discovery(work / "discovery", self.book, self.local_id, node.tls,
                              boot_id="mac-development-observer", start_token=config.start_token)
        self.report["context_sha256"] = node.sha256
        return config, discovery

    async def before_arrival(self, discovery):
        async with asyncio.timeout(PROBE_SECONDS * 3 + 5):
            while True:
                rows = {item["device_id"]: item for item in discovery.snapshot()}
                if rows[self.existing_id]["state"] == "current" and rows[self.new_id]["state"] == "unreachable":
                    self.report["before_arrival"] = list(rows.values())
                    return
                await asyncio.sleep(0.05)

    async def introduce_and_observe(self, app):
        self.report["application_health_before_arrival"] = app.health()
        endpoint = self.book.endpoint(self.new_id)
        self.report["arrival_at"] = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        self.servers[self.new_id] = PeerServer(self.receivers[self.new_id], self.identities[self.new_id],
                                              bind=endpoint[1:]).start()
        print(json.dumps({"new_local_receiver_arrived": self.new_id, "default_poll_seconds": POLL_SECONDS}), flush=True)
        async with asyncio.timeout(POLL_SECONDS + PROBE_SECONDS * 3 + 5):
            while True:
                row = next(item for item in app.discovery.snapshot() if item["device_id"] == self.new_id)
                if row["state"] == "current" and row["available_observed"]:
                    self.report["discovered_after_seconds"] = round(time.monotonic() - started, 3)
                    self.report["after_arrival"] = app.discovery.snapshot()
                    self.report["application_health_after_arrival"] = app.health()
                    print(json.dumps({"new_local_receiver_discovered": self.new_id,
                                      "after_seconds": self.report["discovered_after_seconds"]}), flush=True)
                    return
                await asyncio.sleep(0.1)

    def export(self, output, discovery):
        self.report["observation_attempts"] = discovery.audit.records()
        self.report["public_owner_key"] = base64.b64encode(self.public).decode()
        self.report["signed_enrollment"] = json.loads(self.policy.signed)
        self.report["signed_discovery_book"] = json.loads(self.book.signed)
        self.report["receivers"] = {}
        for identity, receiver in self.receivers.items():
            server = self.servers.get(identity)
            if server is not None:
                server.close()
            self.report["receivers"][identity] = {
                "events": receiver.state.events(), "current_birth": receiver.state.snapshot()["current_birth"],
                "transport_records": server.audit.records() if server is not None else [],
                "certificate_pem": (self.identities[identity] / "certificate.pem").read_text(),
            }
        (output / "discovery.json").write_text(json.dumps(self.report, indent=2) + "\n")
