"""Exercise the complete real RPC lifecycle; no model or native lineage claim."""

import asyncio
import hashlib
from pathlib import Path
import tempfile
import time
import unittest

from organism.application import local_request
from organism.audit_equipment_diagnostics import inspect_job
from organism.capsule import canonical
from organism.equipment_development import case
from organism.inference_server import RuntimeConfig
from organism.reproduction_evidence import verify_choice
from organism.research.control_policy import FixedApplication, FixedReproduction
from organism.research.fixed_lifecycle_candidate import (
    FixedLifecycleCandidate, FixedReproductionLifecycleCandidate,
)
from organism.tests import test_reproduction as peer_fixture


class FixedLifecycleCandidateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rep-lifecycle-", dir="/private/tmp")
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.work = root / "work"
        self.work.mkdir(mode=0o700)
        model, executable = root / "unused-model", root / "unused-runtime"
        model.write_bytes(b"Fixture; never loaded.")
        executable.write_bytes(b"Fixture; never executed.")
        self.config = RuntimeConfig(
            model, hashlib.sha256(model.read_bytes()).hexdigest(), model.stat().st_size,
            executable, hashlib.sha256(executable.read_bytes()).hexdigest(),
            "1" * 32, "2" * 64, "3" * 32, "4" * 64, False, False)

    async def test_frozen_application_reproduces_native_lifecycle_defect(self):
        app = FixedApplication(self.config, self.work)
        with self.assertRaisesRegex(RuntimeError, "Owned model process became unavailable"):
            await asyncio.wait_for(app.run(), timeout=5)
        self.assertTrue(app.closed)
        self.assertFalse(app.receipt["inference"]["model_started"])

    async def start_candidate(self):
        app = FixedLifecycleCandidate(self.config, self.work)
        running = asyncio.create_task(app.run())
        async with asyncio.timeout(5):
            while not app.listener:
                if running.done():
                    await running
                await asyncio.sleep(0.01)
        return app, running

    async def test_diagnostic_survives_full_loop_and_stops_with_zero_inference(self):
        app, running = await self.start_candidate()
        try:
            # Cross a complete supervisory interval, rather than testing start()
            # in isolation as the earlier fixture did.
            await asyncio.sleep(1.05)
            self.assertFalse(running.done())
            health = await local_request(app.socket, {"operation": "health"})
            self.assertTrue(health["policy_ready"])
            self.assertFalse(health["inference_ready"])
            item = case("g", "nominal", now=time.time())
            raw = canonical(item["document"])
            (self.work / "equipment-interfaces" / (item["case_id"] + ".json")).write_bytes(raw)
            request = {"operation": "diagnose", "request_id": "5" * 32, "device_id": item["case_id"]}
            await local_request(app.socket, request)
            async with asyncio.timeout(5):
                while app.store.get(request["request_id"])["record"]["status"] == "running":
                    await asyncio.sleep(0.01)
            job = app.store.get(request["request_id"])
            checked = inspect_job(item, raw, job, live=False)
            self.assertTrue(checked["semantic_passed"])
            self.assertEqual(checked["model_calls"], 0)
            self.assertEqual(job["record"]["trace"]["http_requests"], [])
        finally:
            app.stop_event.set()
            await asyncio.wait_for(running, timeout=5)
        self.assertTrue(app.closed)
        self.assertIsNone(app.inference.process)
        self.assertFalse(app.receipt["shutdown"]["model_started"])
        self.assertNotIn("failure", app.receipt)

    async def test_policy_failure_stops_the_candidate(self):
        app, running = await self.start_candidate()
        app.inference.failure = "Injected local policy lifecycle failure"
        with self.assertRaisesRegex(RuntimeError, "Fixed policy application became unavailable"):
            await asyncio.wait_for(running, timeout=3)
        self.assertTrue(app.closed)
        self.assertFalse(app.receipt["shutdown"]["model_started"])


class FixedPeerLifecycleCandidateTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_loop_transfers_with_no_model_and_requires_readiness(self):
        fixture = peer_fixture.ReproductionTests(methodName="runTest")
        await fixture.asyncSetUp()
        original_app = fixture.app
        reproduction = FixedReproductionLifecycleCandidate(
            fixture.config, fixture.work, fixture.discovery, fixture.client,
            fixture.f.source, interval=0.01)
        app = FixedLifecycleCandidate(
            fixture.config, fixture.work, discovery=fixture.discovery, reproduction=reproduction)
        running = asyncio.create_task(app.run())
        try:
            async with asyncio.timeout(8):
                while True:
                    fixture.f.now += 0.02
                    birth = fixture.f.state.snapshot()["current_birth"]
                    if birth and birth["status"] == "staged":
                        break
                    if running.done():
                        await running
                    await asyncio.sleep(0.01)
            self.assertFalse(app.health()["inference_ready"])
            self.assertTrue(app.health()["policy_ready"])
            self.assertEqual(reproduction.public()["completed_children"], 0)
            job = reproduction.journal.decisions.get(birth["operation_id"])
            self.assertEqual(job["record"]["trace"]["http_requests"], [])
            self.assertEqual(job["record"]["trace"]["model_calls"], [])
            with self.assertRaises(ValueError):
                verify_choice(job, peer_id=fixture.f.device, start_token=fixture.config.start_token,
                              envelope=reproduction.envelope)
            # Process/readiness are explicit fixture observations. Transfer and
            # authenticated policy execution above use real sockets and bytes.
            fixture.fixture_child_ready()
            async with asyncio.timeout(5):
                while reproduction.public()["completed_children"] != 1:
                    fixture.f.now += 0.02
                    await asyncio.sleep(0.01)
            self.assertIsNone(app.inference.process)
        finally:
            app.stop_event.set()
            await asyncio.wait_for(running, timeout=5)
            await original_app.close()
            await fixture.discovery.close()
            fixture.close_servers()
            fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
