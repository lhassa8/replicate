"""Real TLS and bytes; model, process and readiness outcomes are explicit fixtures."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from organism.application import AgentApplication, local_request
from organism.async_peer import AsyncPeerClient
from organism.capsule import CapsuleError, sign_manifest
from organism.discovery import Discovery
from organism.discovery_book import DiscoveryBook, SCHEMA, sign_discovery_book
from organism.inference_server import RuntimeConfig
from organism.peer_identity import PeerIdentityError, device_certificate, owner_certificate, write_identity
from organism.peer_transport import BUFFER_BYTES, PeerClient, PeerProtocolError, PeerServer
from organism.receiver import LocalReceiver
from organism.receiver_state import ReceiverState
from organism.reproduction import Reproduction
from organism.reproduction_agent import ReproductionSession, SPECS
from organism.reproduction_protocol import PROFILE, RAW_PROFILE
from organism.reproduction_evidence import verify_choice
from organism.reproduction_journal import ReproductionError, ReproductionJournal
from organism.tests import test_receiver_state as fixture_module
from organism.tests.test_application import FixtureInference


class ReproductionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        f = self.f = fixture_module.ReceiverStateTests(methodName="runTest")
        f.setUp()
        self.addCleanup(f.doCleanups)
        owner = owner_certificate(f.owner_key)
        self.identities = {}
        for identity, key in zip(f.identities, f.device_keys):
            cert = device_certificate(f.owner_key, owner, key.public_key())
            self.identities[identity] = write_identity(f.root / ("identity-" + identity[:12]), key, cert, owner)
        self.parent_state = ReceiverState.enroll(f.root / "parent-state", f.policy, f.parent,
                                                 boot_id="fixture-parent-boot", clock=lambda: f.now)
        observed = self.parent_state.advertisement()
        parent_birth = self.parent_state.reserve_birth(f.other_parent, "f" * 32, observed["revision"], f.manifest,
                                                       admission_ticket=observed["admission_ticket"])
        self.parent_state.record_staged(parent_birth["operation_id"], parent_birth["manifest_sha256"])
        parent_birth = self.parent_state.claim_start(parent_birth["operation_id"])
        self.parent_state.record_running(parent_birth["operation_id"], parent_birth["process_token"],
                                         {"pid": 12345, "start_identity": "EXPLICIT FIXTURE, NO NATIVE PROCESS",
                                          "boot_id": self.parent_state.boot_id})
        self.parent_state.record_ready(parent_birth["operation_id"], parent_birth["process_token"], "1" * 64)
        self.parent_birth = parent_birth
        self.receivers = {
            f.parent: LocalReceiver(self.parent_state, f.root / "parent-capsules"),
            f.device: LocalReceiver(f.state, f.root / "child-capsules"),
        }
        self.servers = {identity: PeerServer(receiver, self.identities[identity]).start()
                        for identity, receiver in self.receivers.items()}
        self.addCleanup(self.close_servers)
        endpoints = {identity: {"address": server.server_address[0], "port": server.server_address[1]}
                     for identity, server in self.servers.items()}
        endpoints[f.other_parent] = {"address": "127.0.0.1", "port": 7443}
        body = {"schema": SCHEMA, "policy_id": f.policy.policy_id, "epoch": 1,
                "lab_cidrs": ["127.0.0.0/24"], "endpoints": endpoints}
        self.book = DiscoveryBook(sign_discovery_book(body, f.owner_key, f.policy), f.policy, f.owner_public)
        self.client = AsyncPeerClient(f.policy, f.parent, self.identities[f.parent])
        (f.source / ".manifest.json").write_bytes(f.manifest)
        binary = f.root / "fixture-inference"
        binary.write_bytes(b"Explicit fixture. Never executed.")
        binary.chmod(0o700)
        model = f.source / f.capsule_body["model"]["path"]
        self.config = RuntimeConfig(model, hashlib.sha256(model.read_bytes()).hexdigest(), model.stat().st_size,
                                    binary, hashlib.sha256(binary.read_bytes()).hexdigest(),
                                    parent_birth["process_token"], f.parent, parent_birth["individual_id"],
                                    parent_birth["release_id"], True, False)
        self.temp = tempfile.TemporaryDirectory(prefix="or-repro-", dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.discovery = Discovery(self.work / "discovery", self.book, f.parent, self.client.tls,
                                   boot_id=self.parent_state.boot_id, start_token=self.config.start_token,
                                   clock=lambda: f.now, interval=0.05)
        self.addAsyncCleanup(self.discovery.close)
        self.planner_calls = 0
        self.action = "establish"
        self.reproduction = Reproduction(self.config, self.work, self.discovery, self.client, f.source,
                                         planner=self.planner, interval=0.01)
        self.app = AgentApplication(self.config, self.work, discovery=self.discovery,
                                    reproduction=self.reproduction, inference_factory=FixtureInference)
        # Most tests step the controller directly. The fixture inference object
        # is marked available without creating a model or process.
        self.app.inference.ready = True
        self.addAsyncCleanup(self.app.close)

    def close_servers(self):
        for server in self.servers.values():
            server.close()

    async def planner(self, server, client, book, envelope, store, operation_id, peer_id):
        self.planner_calls += 1
        profile = getattr(self, "choice_profile", RAW_PROFILE)
        session = ReproductionSession(client, book, peer_id, envelope, tool_profile=profile)
        observed = await session.invoke("inspect_candidate", {"device_id": peer_id})
        await session.invoke("choose_child_action", {
            "action": self.action, "platform": (observed["platform"] if self.action == "establish"
                                               else "none" if profile == PROFILE else None),
            "evidence_id": observed["evidence_id"]})
        trace = {"scope": "Explicit scripted planner; no model call", "tool_events": session.events,
                 "model_calls": [], "model_streams": [], "model_responses": [], "http_requests": []}
        if profile == PROFILE:
            trace["tool_profile"] = profile
        for index, event in enumerate(session.events):
            tool = {"name": event["tool"], "toolUseId": "fixture-" + str(index), "input": event["arguments"]}
            trace["model_calls"].append({"scope": "scripted fixture"})
            trace["model_streams"].append([
                {"contentBlockStart": {"start": {"toolUse": {"name": tool["name"], "toolUseId": tool["toolUseId"]}}}},
                {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tool["input"])}}}},
                {"contentBlockStop": {}}])
            trace["model_responses"].append({"message": {"content": [{"toolUse": tool}]}, "exception": None})
            trace["http_requests"].append({"method": "POST", "path": "/v1/chat/completions", "body": {
                "model": "organism-" + self.config.start_token, "max_tokens": 192, "temperature": 0, "stream": True}})
            if profile == PROFILE:
                trace["http_requests"][-1]["body"]["tools"] = [
                    {"type": "function", "function": {"name": spec["name"],
                     "parameters": deepcopy(spec["inputSchema"]["json"])}} for spec in SPECS]
        store.update(operation_id, trace=trace,
                     submission=session.choice, status="finished")

    def active(self):
        values = self.reproduction.journal.active()
        return values[0] if values else None

    async def test_fractional_clock_preserves_original_reproduction_deadline(self):
        # A retained native discovery clock crosses a floating-point precision
        # boundary when 1200 is added. Subtraction does not recover exactly 1200.
        self.f.now = 1953.423917855
        await self.discover()
        await self.reproduction.step(self.app)
        self.assertEqual(self.planner_calls, 1)
        intent = self.active()
        self.assertIsNotNone(intent)
        self.assertEqual(intent["phase"], "authorized")
        self.assertEqual(intent["issued_clock"], self.f.now)
        self.assertEqual(intent["deadline_clock"], self.f.now + 1200)
        self.assertNotEqual(intent["deadline_clock"] - self.f.now, 1200)
        reopened = ReproductionJournal(
            self.work / "reproduction", self.config, self.book)
        self.assertEqual(reopened.operations[intent["operation_id"]], intent)
        changed = deepcopy(intent)
        changed["deadline_clock"] = math.nextafter(
            changed["deadline_clock"], math.inf)
        with self.assertRaises(ReproductionError):
            reopened.validate(changed)

    async def discover(self):
        await self.discovery.probe_one(self.f.device)
        self.assertEqual(self.discovery.events.qsize(), 1)

    async def step(self):
        self.f.now += 0.02
        await self.reproduction.step(self.app)

    async def transfer_to_staged(self):
        await self.discover()
        for _ in range(18):
            await self.step()
            birth = self.f.state.snapshot()["current_birth"]
            if birth and birth["status"] == "staged":
                return birth
        self.fail("Fixture capsule did not become staged: " + repr(self.reproduction.public()))

    def fixture_child_ready(self):
        birth = self.f.state.snapshot()["current_birth"]
        birth = self.f.state.claim_start(birth["operation_id"])
        self.f.state.record_running(birth["operation_id"], birth["process_token"],
                                   {"pid": 23456, "start_identity": "EXPLICIT CHILD FIXTURE, NO PROCESS",
                                    "boot_id": self.f.state.boot_id})
        return self.f.state.record_ready(birth["operation_id"], birth["process_token"], "2" * 64)

    async def test_recorded_choice_transfers_both_platforms_but_staging_is_not_readiness(self):
        birth = await self.transfer_to_staged()
        self.assertEqual(self.planner_calls, 1)
        self.assertEqual(self.active()["phase"], "waiting")
        self.assertEqual(self.reproduction.public()["completed_children"], 0)
        destination = self.receivers[self.f.device].store / birth["release_id"]
        for name, expected in self.f.capsule_body["files"].items():
            self.assertEqual(hashlib.sha256((destination / name).read_bytes()).hexdigest(), expected["sha256"])
        self.fixture_child_ready()
        await self.step()
        self.assertIsNone(self.active())
        self.assertEqual(self.reproduction.public()["completed_children"], 1)
        self.assertFalse(self.reproduction.public()["qualification_passed"])
        outcome = list(self.reproduction.journal.operations.values())[0]
        self.assertEqual(outcome["receipt"]["ready_evidence"], "2" * 64)
        self.assertEqual(outcome["platform"], "linux-arm64")
        self.assertNotEqual(outcome["receipt"]["individual_id"], self.config.individual_id)

    async def test_deferred_model_choice_never_reserves_a_child(self):
        self.action = "defer"
        await self.discover()
        await self.step()
        self.assertEqual(self.planner_calls, 1)
        self.assertIsNone(self.f.state.snapshot()["current_birth"])
        self.assertEqual(list(self.reproduction.journal.operations.values())[0]["phase"], "deferred")
        await self.discovery.probe_one(self.f.device)
        await self.step()
        self.assertEqual(self.planner_calls, 1)

    async def test_malformed_literal_choice_is_not_repaired_or_allowed_to_deploy(self):
        await self.discover()
        original = self.planner

        async def altered(*args):
            await original(*args)
            store, operation = args[4], args[5]
            with store.db() as db:
                row = db.execute("SELECT digest,body FROM jobs WHERE request_id=?", (operation,)).fetchone()
                record = store.decode(*row)
                record["trace"]["model_streams"][-1][1]["contentBlockDelta"]["delta"]["toolUse"]["input"] = '{"action":'
                store.save(db, record)

        self.reproduction.planner = altered
        await self.step()
        self.assertEqual(list(self.reproduction.journal.operations.values())[0]["phase"], "decision_failed")
        await self.step()
        self.assertIsNone(self.f.state.snapshot()["current_birth"])

    async def test_versioned_defer_keeps_literal_none_and_never_reserves_a_child(self):
        self.choice_profile, self.action = PROFILE, "defer"
        await self.discover()
        await self.step()
        operation = list(self.reproduction.journal.operations.values())[0]
        self.assertEqual(operation["phase"], "deferred")
        self.assertIsNone(self.f.state.snapshot()["current_birth"])
        job = self.reproduction.journal.decisions.get(operation["operation_id"])
        self.assertEqual(job["record"]["trace"]["tool_events"][-1]["arguments"]["platform"], "none")
        self.assertIsNone(job["record"]["submission"]["platform"])

    async def test_versioned_establishment_still_requires_supported_inspected_platform(self):
        self.choice_profile = PROFILE
        birth = await self.transfer_to_staged()
        self.assertEqual(birth["status"], "staged")
        self.assertEqual(self.active()["platform"], "linux-arm64")

    async def test_parsed_argument_or_model_alias_substitution_fails_choice_validation(self):
        await self.discover()
        await self.step()
        job = self.reproduction.journal.decisions.get(self.active()["operation_id"])
        for field in ("arguments", "alias"):
            value = deepcopy(job)
            if field == "arguments":
                value["record"]["trace"]["model_responses"][-1]["message"]["content"][0]["toolUse"]["input"]["action"] = "defer"
            else:
                value["record"]["trace"]["http_requests"][0]["body"]["model"] = "another-model"
            from organism.capsule import canonical
            value["digest"] = hashlib.sha256(canonical(value["record"])).hexdigest()
            with self.subTest(field=field), self.assertRaises(ReproductionError):
                verify_choice(value, peer_id=self.f.device, start_token=self.config.start_token,
                              envelope=self.reproduction.envelope)

    async def test_prior_ready_flag_does_not_make_a_recovering_child_complete(self):
        await self.transfer_to_staged()
        value = deepcopy(self.active()["receipt"])
        value.update(status="running", ready_once=True, ready_evidence="2" * 64)
        self.reproduction.receipt(self.active(), value)
        self.assertEqual(self.active()["phase"], "waiting")
        self.assertEqual(self.reproduction.public()["completed_children"], 0)

    async def test_changed_child_identity_cannot_replace_same_operation_receipt(self):
        await self.transfer_to_staged()
        value = deepcopy(self.active()["receipt"])
        value["individual_id"] = "0" * 32
        with self.assertRaises(ReproductionError):
            self.reproduction.receipt(self.active(), value)

    async def test_interrupted_reasoning_is_preserved_across_application_incarnations(self):
        await self.discover()
        self.app.busy = True
        await self.step()
        intent = self.active()
        reopened = ReproductionJournal(self.work / "reproduction",
                                       replace(self.config, start_token="0" * 32), self.book)
        job = reopened.decisions.get(intent["operation_id"])
        self.assertEqual(job["record"]["status"], "interrupted")
        self.assertEqual(reopened.active()[0]["decision_token"], self.config.start_token)
        self.assertEqual(self.planner_calls, 0)

    async def test_expired_unstarted_decision_does_not_send_cancellation_or_reservation(self):
        await self.discover()
        self.app.busy = True
        await self.step()
        self.f.now = self.active()["deadline_clock"] + 1
        await self.step()
        self.assertIsNone(self.active())
        self.assertEqual(list(self.reproduction.journal.operations.values())[0]["phase"], "decision_failed")
        self.assertEqual(self.planner_calls, 0)
        self.assertEqual(self.f.state.snapshot()["birth_count"], 0)

    async def test_parent_stop_prevents_reasoning_and_remote_admission(self):
        await self.discover()
        self.parent_state.owner_control("stop", "fixture owner stop")
        await self.step()
        self.assertEqual(self.planner_calls, 0)
        self.assertIsNone(self.f.state.snapshot()["current_birth"])
        self.assertIsNone(self.active())

    async def test_foreign_local_incarnation_receipt_cannot_authorize_reproduction(self):
        await self.discover()
        original = self.client.request

        async def changed(endpoint, operation, **kwargs):
            result = await original(endpoint, operation, **kwargs)
            if operation == "self_status":
                result["incarnation_sha256"] = "0" * 64
            return result

        with patch.object(self.client, "request", side_effect=changed):
            await self.step()
        self.assertEqual(self.planner_calls, 0)
        self.assertIsNone(self.f.state.snapshot()["current_birth"])

    async def test_readiness_query_is_device_local_and_other_operations_remain_peer_only(self):
        own = await self.client.request(self.book.endpoint(self.f.parent), "self_status")
        self.assertTrue(own["ready"])
        self.assertNotIn("process_token", own)
        self.assertNotIn("operation_id", own)
        with self.assertRaises(PeerIdentityError):
            await self.client.observe(self.book.endpoint(self.f.parent))
        with self.assertRaises(PeerIdentityError):
            await self.client.request(self.book.endpoint(self.f.device), "self_status")
        other = PeerClient(self.f.policy, self.f.other_parent, self.identities[self.f.other_parent])
        with self.assertRaises(PeerProtocolError):
            await asyncio.to_thread(other.request, self.book.endpoint(self.f.parent), "self_status")

    async def test_lost_reservation_reply_reuses_same_operation_and_model_decision(self):
        await self.discover()
        await self.step()
        original = self.client.reserve

        async def lost(*args, **kwargs):
            await original(*args, **kwargs)
            raise TimeoutError("Fixture reply lost after durable reservation")

        with patch.object(self.client, "reserve", side_effect=lost):
            await self.step()
        operation = self.active()["operation_id"]
        self.assertEqual(self.f.state.snapshot()["birth_count"], 1)
        await self.step()
        self.assertEqual(self.active()["operation_id"], operation)
        self.assertEqual(self.active()["phase"], "receiving")
        self.assertEqual(self.f.state.snapshot()["birth_count"], 1)
        self.assertEqual(self.planner_calls, 1)

    async def test_reopened_journal_resumes_transfer_without_another_model_choice(self):
        await self.discover()
        await self.step()
        await self.step()
        operation = self.active()["operation_id"]
        restored = Reproduction(self.config, self.work, self.discovery, self.client, self.f.source,
                                planner=self.planner, interval=0.01)
        self.reproduction = restored
        for _ in range(15):
            await self.step()
            if self.f.state.snapshot()["current_birth"]["status"] == "staged":
                break
        self.assertEqual(self.active()["operation_id"], operation)
        self.assertEqual(self.active()["phase"], "waiting")
        self.assertEqual(self.planner_calls, 1)
        self.assertEqual(self.f.state.snapshot()["birth_count"], 1)

    async def test_foreign_child_receipt_is_rejected_without_completion(self):
        await self.transfer_to_staged()
        value = deepcopy(self.active()["receipt"])
        value.update(status="ready", ready_once=True, ready_evidence="2" * 64, device_id=self.f.other_parent)
        with self.assertRaises(ReproductionError):
            self.reproduction.receipt(self.active(), value)
        self.assertEqual(self.reproduction.public()["completed_children"], 0)

    async def test_duplicate_arrival_cannot_start_two_children_or_repeat_reasoning(self):
        await self.discover()
        await self.step()
        await self.step()
        self.discovery.pending.add(self.f.device)
        self.discovery.events.put_nowait(self.f.device)
        for _ in range(14):
            await self.step()
        self.assertEqual(self.planner_calls, 1)
        self.assertEqual(self.f.state.snapshot()["birth_count"], 1)
        self.assertEqual(len(self.reproduction.journal.operations), 1)

    async def test_mutation_budget_exhaustion_keeps_uncertain_operation_owned(self):
        await self.discover()
        await self.step()
        with patch.object(self.client, "reserve", side_effect=TimeoutError("fixture inaccessible receiver")):
            for _ in range(5):
                await self.step()
        active = self.active()
        self.assertEqual(active["phase"], "uncertain")
        self.assertEqual(active["attempts"]["reserve"], 3)
        self.assertEqual(self.planner_calls, 1)
        self.assertEqual(self.f.state.snapshot()["birth_count"], 0)
        self.assertEqual(len(self.reproduction.journal.operations), 1)

    async def test_deadline_cancels_unfinished_birth_but_never_erases_a_ready_child(self):
        await self.transfer_to_staged()
        self.f.now = self.active()["deadline_clock"] + 1
        await self.step()
        self.assertEqual(list(self.reproduction.journal.operations.values())[0]["phase"], "aborted")
        self.assertEqual(self.reproduction.public()["completed_children"], 0)

    async def test_model_invalid_platform_and_evidence_are_rejected_before_transfer(self):
        session = ReproductionSession(self.client, self.book, self.f.device, self.reproduction.envelope)
        observed = await session.invoke("inspect_candidate", {"device_id": self.f.device})
        for changes in ({"platform": "linux-x86_64"}, {"evidence_id": "0" * 64}):
            with self.assertRaises(ReproductionError):
                await session.invoke("choose_child_action", {
                    "action": "establish", "platform": "linux-arm64", "evidence_id": observed["evidence_id"], **changes})
        self.assertIsNone(session.choice)
        self.assertIsNone(self.f.state.snapshot()["current_birth"])

    async def test_owner_retirement_after_model_inspection_invalidates_actual_reservation(self):
        await self.discover()
        await self.step()
        self.f.state.owner_control("retire", "fixture recipient retirement")
        await self.step()
        self.assertIsNone(self.f.state.snapshot()["current_birth"])
        self.assertEqual(self.f.state.snapshot()["owner_mode"], "retired")
        self.assertIsNotNone(self.active())
        self.assertEqual(self.reproduction.public()["completed_children"], 0)

    async def test_changed_or_missing_carried_model_rejects_controller_construction(self):
        changed = replace(self.config, model_sha256="0" * 64)
        with self.assertRaises(ReproductionError):
            Reproduction(changed, self.work, self.discovery, self.client, self.f.source)
        self.config.model.write_bytes(b"corrupt fixture payload")
        with self.assertRaises(CapsuleError):
            Reproduction(self.config, self.work, self.discovery, self.client, self.f.source)

    async def test_changed_journal_identity_or_removed_history_cannot_reset_allowances(self):
        await self.discover()
        await self.step()
        altered = replace(self.config, individual_id="0" * 32)
        with self.assertRaises(ReproductionError):
            ReproductionJournal(self.work / "reproduction", altered, self.book)
        (self.work / "reproduction/peer-audit.sqlite3").unlink()
        with self.assertRaises(ReproductionError):
            ReproductionJournal(self.work / "reproduction", self.config, self.book)

    async def test_idle_controller_does_not_poll_local_status_or_run_inference(self):
        before = len(self.servers[self.f.parent].audit.records())
        for _ in range(10):
            await self.step()
        self.assertEqual(len(self.servers[self.f.parent].audit.records()), before)
        self.assertEqual(self.planner_calls, 0)

    async def test_stale_queued_arrival_waits_for_fresh_discovery_instead_of_disappearing(self):
        await self.discover()
        self.f.now += 31
        await self.step()
        self.assertEqual(self.reproduction.candidate, self.f.device)
        self.assertEqual(self.planner_calls, 0)
        await self.discovery.probe_one(self.f.device)
        await self.step()
        self.assertEqual(self.planner_calls, 1)

    async def test_existing_diagnostic_generation_prevents_concurrent_model_reasoning(self):
        await self.discover()
        self.app.busy = True
        await self.step()
        self.assertEqual(self.planner_calls, 0)
        self.assertEqual(self.active()["phase"], "proposed")
        self.app.busy = False
        await self.step()
        self.assertEqual(self.planner_calls, 1)

    async def test_application_starts_discovery_and_reproduction_without_a_birth_rpc(self):
        await self.app.start()
        async with asyncio.timeout(8):
            while not self.f.state.snapshot()["current_birth"] or self.f.state.snapshot()["current_birth"]["status"] != "staged":
                self.f.now += 0.02
                await asyncio.sleep(0.01)
        self.assertEqual(self.planner_calls, 1)
        response = await local_request(self.app.socket, {"operation": "offspring"})
        self.assertEqual(response["reproduction"]["completed_children"], 0)
        self.assertTrue(self.app.health()["reproduction_running"])
        await self.app.close()
        self.assertTrue(self.app.reproduction_task.done())
        self.assertTrue(self.app.discovery_task.done())

    async def test_cancelled_async_transfer_closes_socket_and_same_operation_can_resume(self):
        endpoint = self.book.endpoint(self.f.device)
        observed = await self.client.observe(endpoint)
        operation = "b" * 32
        await self.client.reserve(endpoint, operation, self.f.manifest, observed)
        name = next(iter(self.f.capsule_body["files"]))
        expected = self.f.capsule_body["files"][name]
        original = self.client.send
        entered = asyncio.Event()

        async def pause(writer, op, **fields):
            await original(writer, op, **fields)
            if op == "put":
                entered.set()
                await asyncio.Event().wait()

        with patch.object(self.client, "send", side_effect=pause):
            task = asyncio.create_task(self.client.upload(endpoint, operation, name, expected, self.f.source))
            await asyncio.wait_for(entered.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        async with asyncio.timeout(3):
            while self.servers[self.f.device]._transfer.locked():
                await asyncio.sleep(0.01)
        result = await self.client.transfer(endpoint, operation, self.f.manifest, self.f.source)
        self.assertEqual(result["status"], "staged")
        self.assertEqual(self.f.state.snapshot()["birth_count"], 1)

    def grow_async_payload_fixture(self):
        name = self.f.capsule_body["model"]["path"]
        body = (b"Explicit TLS test fixture, not model weights.\n" * BUFFER_BYTES)[:3 * BUFFER_BYTES + 123]
        (self.f.source / name).write_bytes(body)
        expected = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        self.f.capsule_body["files"][name] = expected
        self.f.manifest = sign_manifest(self.f.capsule_body, self.f.release_key)
        return name, body, expected

    @asynccontextmanager
    async def paced_async_payload(self, callback=None):
        connection = self.client.connection

        @asynccontextmanager
        async def paced(endpoint, **kwargs):
            async with connection(endpoint, **kwargs) as (reader, writer):
                class Sender:
                    payload_block = False

                    def write(self, block):
                        self.payload_block = len(block) == BUFFER_BYTES
                        writer.write(block)

                    async def drain(self):
                        if self.payload_block:
                            await asyncio.sleep(0.08)
                            if callback:
                                callback()
                        await writer.drain()

                yield reader, Sender()

        with patch.object(self.client, "connection", side_effect=paced):
            yield

    async def reserve_async_payload_fixture(self):
        endpoint = self.book.endpoint(self.f.device)
        observed = await self.client.observe(endpoint)
        await self.client.reserve(endpoint, self.f.operation, self.f.manifest, observed)
        return endpoint

    async def wait_for_async_payload_cleanup(self):
        server = self.servers[self.f.device]
        async with asyncio.timeout(3):
            while server._transfer.locked():
                await asyncio.sleep(0.01)

    async def test_async_payload_can_outlive_old_fixed_deadline(self):
        name, body, expected = self.grow_async_payload_fixture()
        endpoint = await self.reserve_async_payload_fixture()
        with (patch("organism.async_peer.TRANSFER_SECONDS", 0.15),
              patch("organism.peer_transport.TRANSFER_SECONDS", 0.15),
              patch("organism.peer_transport.NETWORK_BYTES_PER_SECOND", 256 * 1024)):
            async with self.paced_async_payload():
                result = await self.client.upload(endpoint, self.f.operation, name, expected, self.f.source)
        self.assertTrue(result["stored"])
        stored = self.servers[self.f.device].incoming / self.f.operation / name
        self.assertEqual(stored.read_bytes(), body)
        self.assertEqual(self.f.state.snapshot()["birth_count"], 1)

    async def test_async_payload_progress_does_not_reset_absolute_deadline(self):
        name, _, expected = self.grow_async_payload_fixture()
        endpoint = await self.reserve_async_payload_fixture()
        with patch("organism.async_peer.payload_seconds", return_value=0.15):
            async with self.paced_async_payload():
                with self.assertRaises(TimeoutError):
                    await self.client.upload(endpoint, self.f.operation, name, expected, self.f.source)
        await self.wait_for_async_payload_cleanup()
        self.assertFalse((self.servers[self.f.device].incoming / self.f.operation / name).exists())
        self.assertEqual(self.f.state.snapshot()["birth_count"], 1)

    async def test_async_payload_size_budget_does_not_renew_expired_birth(self):
        name, _, expected = self.grow_async_payload_fixture()
        endpoint = await self.reserve_async_payload_fixture()
        async with self.paced_async_payload(lambda: setattr(self.f, "now", 161)):
            with self.assertRaises((OSError, PeerProtocolError, asyncio.IncompleteReadError)):
                await self.client.upload(endpoint, self.f.operation, name, expected, self.f.source)
        await self.wait_for_async_payload_cleanup()
        self.assertFalse((self.servers[self.f.device].incoming / self.f.operation / name).exists())
        self.assertIsNone(self.f.state.snapshot()["current_birth"])
        self.assertEqual(self.f.state.snapshot()["birth_count"], 1)

    async def test_owner_stop_interrupts_async_payload_with_size_budget(self):
        name, _, expected = self.grow_async_payload_fixture()
        endpoint = await self.reserve_async_payload_fixture()
        async with self.paced_async_payload(
                lambda: self.f.state.owner_control("stop", "Explicit test owner stop")):
            with self.assertRaises((OSError, PeerProtocolError, asyncio.IncompleteReadError)):
                await self.client.upload(endpoint, self.f.operation, name, expected, self.f.source)
        await self.wait_for_async_payload_cleanup()
        self.assertFalse((self.servers[self.f.device].incoming / self.f.operation / name).exists())
        self.assertEqual(self.f.state.snapshot()["owner_mode"], "stopped")
