"""Local application contracts. Inference is explicit fixture data in this suite."""

import asyncio
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from organism.application import AgentApplication, local_request
from organism.application_store import ApplicationStore, JobError
from organism.diagnostic_agent import EquipmentTool, MODEL_CALL_LIMIT, RecordingModel, SPECS, TraceObserver
from organism.entrypoint import main as native_main, validate_installation
from organism.equipment import EquipmentDirectory, EquipmentError, EquipmentSession, MAX_TOOL_CALLS
from organism.inference_server import InferenceServer, RuntimeConfig
from organism.tests import test_preparation as preparation_fixture


def observation(device="device-a", **changes):
    return {"schema": "organism-temperature-v1", "device_id": device, "sensor_id": "temperature",
            "unit": "C", "normal_range_celsius": [10.0, 40.0], "maximum_observation_age_seconds": 30,
            "value": 28.0, "observed_unix_seconds": 98.0, "sequence": 1, **changes}


def configuration(root):
    model, binary = root / "model", root / "binary"
    model.write_bytes(b"Not model weights: explicit unit fixture")
    binary.write_bytes(b"Not executable code: explicit unit fixture")
    binary.chmod(0o700)
    return RuntimeConfig(model, hashlib.sha256(model.read_bytes()).hexdigest(), model.stat().st_size,
                         binary, hashlib.sha256(binary.read_bytes()).hexdigest(),
                         "a" * 32, "b" * 64, "c" * 32, "d" * 64, False, False)


class EquipmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="oa-test-", dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "device-a.json"
        self.path.write_text(json.dumps(observation()))
        self.source = EquipmentDirectory(self.root, clock=lambda: 100.0)

    def test_file_observations_and_evidence_do_not_include_expected_answers(self):
        session = EquipmentSession(self.source, "device-a")
        cap = session.inspect_device("device-a")
        reading = session.read_measurement("device-a", "temperature")
        self.assertEqual(reading["age_seconds"], 2.0)
        self.assertEqual(reading["observation_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        result = session.submit_assessment("device-a", "normal", 28, [cap["evidence_id"], reading["evidence_id"]])
        self.assertEqual(result["status"], "recorded")
        self.assertFalse(result["independently_verified"])
        self.assertNotIn("expected", json.dumps(session.events))

    def test_observation_path_and_links_cannot_escape_directory(self):
        with self.assertRaises(EquipmentError):
            self.source.read("../device-a")
        other = self.root / "device-b.json"
        other.symlink_to(self.path)
        with self.assertRaises(OSError):
            self.source.read("device-b")

    def test_extra_fields_nonfinite_values_and_future_observations_are_rejected(self):
        for change in ({"expected_status": "normal"}, {"value": float("nan")}, {"value": True},
                       {"observed_unix_seconds": 101}, {"unit": "execute"},
                       {"normal_range_celsius": [40, 10]}, {"sequence": -1}):
            with self.subTest(change=change):
                self.path.write_text(json.dumps(observation(**change)))
                with self.assertRaises(EquipmentError):
                    self.source.read("device-a")

    def test_world_writable_input_is_rejected(self):
        self.path.chmod(0o666)
        with self.assertRaises(EquipmentError):
            self.source.read("device-a")

    def test_capabilities_must_precede_measurement_and_cannot_change_silently(self):
        session = EquipmentSession(self.source, "device-a")
        self.assertEqual(session.read_measurement("device-a", "temperature")["status"], "capabilities_required")
        session.inspect_device("device-a")
        self.path.write_text(json.dumps(observation(unit="F")))
        self.assertEqual(session.read_measurement("device-a", "temperature")["status"], "capabilities_changed")

    def test_invented_or_duplicate_evidence_does_not_record_a_submission(self):
        session = EquipmentSession(self.source, "device-a")
        cap = session.inspect_device("device-a")
        reading = session.read_measurement("device-a", "temperature")
        for identifiers in ([cap["evidence_id"], "fake"], [cap["evidence_id"]] * 2,
                            [cap["evidence_id"], reading["evidence_id"], reading["evidence_id"]]):
            self.assertEqual(session.submit_assessment("device-a", "normal", 28, identifiers)["status"],
                             "missing_or_invented_evidence")
        self.assertIsNone(session.submission)

    def test_wrong_answer_is_recorded_without_being_called_correct(self):
        session = EquipmentSession(self.source, "device-a")
        cap = session.inspect_device("device-a")
        reading = session.read_measurement("device-a", "temperature")
        result = session.submit_assessment("device-a", "high", 999, [cap["evidence_id"], reading["evidence_id"]])
        self.assertEqual(result["status"], "recorded")
        self.assertFalse(result["independently_verified"])
        self.assertEqual(session.submission["temperature_celsius"], 999)

    def test_read_only_observations_do_not_change_during_tool_use(self):
        initial = self.path.read_bytes()
        session = EquipmentSession(self.source, "device-a")
        result = session.inspect_device("device-a")
        result["sensors"][0]["unit"] = "F"
        self.assertEqual(session.read_measurement("device-a", "temperature")["unit"], "C")
        self.assertEqual(self.path.read_bytes(), initial)

    def test_tool_budget_and_one_submission_are_enforced(self):
        session = EquipmentSession(self.source, "device-a")
        for _ in range(MAX_TOOL_CALLS):
            session.inspect_device("device-a")
        with self.assertRaises(RuntimeError):
            session.inspect_device("device-a")
        self.assertEqual(len(session.events), MAX_TOOL_CALLS)

    def test_literal_strands_schema_rejects_extra_arguments(self):
        session = EquipmentSession(self.source, "device-a")
        tool = EquipmentTool(session, SPECS[0])

        async def call():
            return [value async for value in tool.stream(
                {"toolUseId": "fixture-call", "input": {"device_id": "device-a", "command": "not allowed"}}, {})]

        events = asyncio.run(call())
        result = json.loads(events[0].tool_result["content"][0]["text"])
        self.assertEqual(result["status"], "invalid_observation_or_arguments")
        self.assertIsNone(session.capabilities)

    def test_model_call_hook_stops_after_the_fixed_allowance(self):
        trace = {"model_calls": [], "model_responses": []}
        observer = TraceObserver(trace, lambda: None)
        event = SimpleNamespace(cancel=None, agent=SimpleNamespace(messages=[], system_prompt="fixture",
                                tool_registry=SimpleNamespace(get_all_tool_specs=lambda: [])))
        for _ in range(MODEL_CALL_LIMIT):
            observer.before(event)
            self.assertIsNone(event.cancel)
        observer.before(event)
        self.assertIsNotNone(event.cancel)
        self.assertEqual(len(trace["model_calls"]), MODEL_CALL_LIMIT)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="oa-store-", dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ApplicationStore(self.root, "a" * 32)
        self.request = "1" * 32

    def test_retry_returns_saved_outcome_without_another_execution(self):
        value, fresh = self.store.reserve(self.request, "device-a")
        self.assertTrue(fresh)
        completed = self.store.update(self.request, trace={"fixture": True}, status="finished",
                                      submission={"status": "normal"})
        again, fresh = self.store.reserve(self.request, "device-a")
        self.assertFalse(fresh)
        self.assertEqual(completed, again)
        self.assertFalse(self.store.public(again)["independently_verified"])

    def test_request_identity_cannot_be_rebound(self):
        self.store.reserve(self.request, "device-a")
        with self.assertRaisesRegex(JobError, "another device"):
            self.store.reserve(self.request, "device-b")

    def test_restart_retains_partial_trace_as_interrupted(self):
        self.store.reserve(self.request, "device-a")
        self.store.update(self.request, trace={"partial": "observed before fixture exit"})
        reopened = ApplicationStore(self.root, "b" * 32)
        result = reopened.get(self.request)["record"]
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["trace"]["partial"], "observed before fixture exit")
        with self.assertRaises(JobError):
            self.store.update(self.request, trace={}, status="finished")

    def test_completed_trace_is_not_mutable(self):
        self.store.reserve(self.request, "device-a")
        completed = self.store.update(self.request, trace={}, status="failed", failure="fixture failure")
        with self.assertRaises(JobError):
            self.store.update(self.request, trace={"changed": True})
        self.assertEqual(completed, self.store.get(self.request))

    def test_trace_and_job_allowances_preserve_existing_records(self):
        self.store.reserve(self.request, "device-a")
        with patch("organism.application_store.MAX_TRACE_BYTES", 512):
            with self.assertRaises(JobError):
                self.store.update(self.request, trace={"large": "x" * 1024})
        self.assertEqual(self.store.get(self.request)["record"]["trace"], {})
        with patch("organism.application_store.MAX_JOBS", 1):
            with self.assertRaises(JobError):
                self.store.reserve("2" * 32, "device-a")
        self.assertIsNotNone(self.store.get(self.request))

    def test_corrupt_or_missing_journal_cannot_be_reset(self):
        self.store.reserve(self.request, "device-a")
        with sqlite3.connect(self.store.path) as db:
            db.execute("UPDATE jobs SET body='{}'")
        with self.assertRaises(JobError):
            self.store.get(self.request)
        self.store.path.unlink()
        with self.assertRaises(FileExistsError):
            ApplicationStore(self.root, "b" * 32)
        self.assertFalse(self.store.path.exists())


class StreamTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_tool_arguments_are_preserved_before_sdk_parsing(self):
        trace = {"model_streams": []}
        checkpoints = []
        event = {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
            "toolUse": {"input": '{"device_id":"fixture","evidence_ids":'}}}}

        async def original_stream(model, *args, **kwargs):
            yield event

        from strands.models import LlamaCppModel
        model = RecordingModel(trace=trace, checkpoint=lambda: checkpoints.append(True))
        try:
            with patch.object(LlamaCppModel, "stream", original_stream):
                values = [value async for value in model.stream([])]
            self.assertEqual(values, [event])
            self.assertEqual(trace["model_streams"], [[event]])
            self.assertTrue(checkpoints)
        finally:
            await model.client.aclose()


class FixtureInference:
    def __init__(self, config, work):
        self.ready, self.stopped = False, False
        self.failure = None
        self.process = SimpleNamespace(pid=12345, returncode=None)

    async def start(self):
        self.ready = True
        return {"fixture_only": True, "useful_readiness_verified": False}

    async def stop(self):
        self.ready, self.stopped, self.process.returncode = False, True, 0
        return {"owned_server_stopped": True, "fixture_only": True}


class ApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="oa-app-", dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = configuration(self.root)
        self.work = self.root / "work"
        self.work.mkdir(mode=0o700)
        self.release = asyncio.Event()
        self.calls = []

        async def assessor(server, observations, store, request_id, device_id):
            self.calls.append((request_id, device_id))
            await self.release.wait()
            store.update(request_id, trace={"fixture_only": True}, status="finished", submission={"fixture_only": True})

        self.app = AgentApplication(self.config, self.work, inference_factory=FixtureInference, assessor=assessor)
        await self.app.start()
        self.addAsyncCleanup(self.app.close)

    async def test_unix_health_is_available_without_claiming_useful_readiness(self):
        result = await local_request(self.app.socket, {"operation": "health"})
        self.assertEqual(result["status"], "serving")
        self.assertFalse(result["native"])
        self.assertFalse(result["useful_readiness_verified"])
        self.assertEqual(self.app.socket.stat().st_mode & 0o777, 0o600)

    async def test_lost_reply_and_retry_keep_one_diagnostic_request(self):
        request = {"operation": "assess", "request_id": "1" * 32, "device_id": "device-a"}
        first = await local_request(self.app.socket, request)
        second = await local_request(self.app.socket, request)
        self.assertEqual(first["request_id"], second["request_id"])
        await asyncio.sleep(0)
        self.assertEqual(len(self.calls), 1)
        self.release.set()
        await asyncio.gather(*list(self.app.job_tasks))
        third = await local_request(self.app.socket, request)
        self.assertEqual(third["status"], "finished")
        self.assertEqual(len(self.calls), 1)

    async def test_new_request_is_busy_while_existing_inference_is_active(self):
        await local_request(self.app.socket, {"operation": "assess", "request_id": "1" * 32, "device_id": "device-a"})
        second = await local_request(self.app.socket, {"operation": "assess", "request_id": "2" * 32, "device_id": "device-b"})
        self.assertEqual(second["status"], "busy_or_unavailable")
        self.assertIsNone(self.app.store.get("2" * 32))

    async def test_unknown_operations_paths_and_duplicate_identity_are_rejected(self):
        for request in ({"operation": "start"}, {"operation": "health", "extra": True},
                        {"operation": "assess", "request_id": "bad", "device_id": "../escape"}):
            result = await local_request(self.app.socket, request)
            self.assertEqual(result["status"], "rejected")

    async def test_second_application_cannot_replace_listener_or_existing_receipt(self):
        receipt = self.work / ("application-" + self.config.start_token + ".json")
        before = receipt.read_bytes()
        second = AgentApplication(self.config, self.work, inference_factory=FixtureInference)
        with self.assertRaises(BlockingIOError):
            await second.run()
        self.assertEqual(receipt.read_bytes(), before)
        self.assertEqual((await local_request(self.app.socket, {"operation": "health"}))["status"], "serving")

    async def test_shutdown_stops_owned_model_and_removes_only_owned_socket(self):
        unrelated = self.work / "owner-note"
        unrelated.write_text("keep")
        await self.app.close()
        self.assertTrue(self.app.inference.stopped)
        self.assertFalse(self.app.socket.exists())
        self.assertEqual(unrelated.read_text(), "keep")
        record = json.loads((self.work / ("application-" + self.config.start_token + ".json")).read_bytes())
        self.assertTrue(record["shutdown"]["owned_server_stopped"])

    async def test_exit_observation_makes_health_unavailable(self):
        self.app.inference.process.returncode = 7
        result = await local_request(self.app.socket, {"operation": "health"})
        self.assertFalse(result["inference_ready"])

    async def test_logging_failure_makes_health_unavailable_before_process_exit(self):
        self.app.inference.failure = "Inference log allowance exhausted"
        self.assertFalse(self.app.health()["inference_ready"])

    async def test_receipt_hard_link_is_rejected_before_truncating_other_file(self):
        other = self.work / "owner-note"
        other.write_text("preserve this content")
        pending = self.work / ".application-receipt.pending"
        os.link(other, pending)
        try:
            with self.assertRaisesRegex(ValueError, "owned regular file"):
                self.app.write_receipt()
            self.assertEqual(other.read_text(), "preserve this content")
        finally:
            pending.unlink()

    async def test_private_work_directory_required(self):
        self.work.chmod(0o755)
        with self.assertRaises(ValueError):
            AgentApplication(self.config, self.work, inference_factory=FixtureInference)
        self.work.chmod(0o700)

    async def test_model_input_digest_change_is_rejected_before_spawn(self):
        self.config.verify()
        self.config.model.write_bytes(b"changed fixture")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            await InferenceServer(self.config, self.work).start()


class BootstrapTests(unittest.TestCase):
    class ReachedInstallation(Exception):
        """Stop after bootstrap admission without executing an application."""

    def setUp(self):
        self.fixture = preparation_fixture.PreparationTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_stdlib_bootstrap_rechecks_complete_owner_installed_tree(self):
        prepared = self.fixture.prepare()
        index, manifest = validate_installation(prepared.path, owner_uid=os.geteuid())
        self.assertEqual(index["release_id"], manifest["release_id"])
        path = prepared.path / "runtime/agent/organism/entrypoint.py"
        path.chmod(0o644)
        path.write_text("# changed fixture\n")
        path.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "file differs"):
            validate_installation(prepared.path, owner_uid=os.geteuid())

    def test_native_cli_refuses_mac(self):
        with patch("organism.entrypoint.platform.system", return_value="Darwin"):
            with patch("sys.argv", ["entrypoint", "--installation", "/fixture", "--work-directory", "/fixture-work"]):
                with self.assertRaisesRegex(RuntimeError, "Linux"):
                    native_main()

    def bootstrap_group(self, parent, membership, *, supplied=None):
        root = self.fixture.fixture.root.resolve()
        token, device, individual = "a" * 32, "b" * 64, "c" * 32
        unit = "organism-" + device[:16] + "-" + individual + "-" + token + ".service"
        expected = parent + "/" + unit
        environment = {"ORGANISM_START_TOKEN": token, "ORGANISM_DEVICE_ID": device,
                       "ORGANISM_INDIVIDUAL_ID": individual,
                       "ORGANISM_SERVICE_CGROUP": expected if supplied is None else supplied}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, environment, clear=True))
            stack.enter_context(patch("organism.entrypoint.platform.system", return_value="Linux"))
            stack.enter_context(patch("organism.entrypoint.os.geteuid", return_value=1000))
            stack.enter_context(patch("organism.entrypoint.__file__", str(root / "runtime/agent/organism/entrypoint.py")))
            stack.enter_context(patch("sys.argv", ["entrypoint", "--installation", str(root),
                                                 "--work-directory", str(root / "work"),
                                                 "--node-context", str(root / "context"),
                                                 "--node-context-sha256", "d" * 64]))
            stack.enter_context(patch("organism.entrypoint.Path.read_text",
                                      return_value=membership.replace("{unit}", unit)))
            stack.enter_context(patch("organism.entrypoint.validate_installation",
                                      side_effect=self.ReachedInstallation))
            native_main()

    def test_native_cli_accepts_exact_owner_selected_service_group(self):
        for parent in ("/system.slice", "/system.slice/system-organism.slice"):
            with self.subTest(parent=parent):
                # Stop immediately after the real bootstrap admission check.
                with self.assertRaises(self.ReachedInstallation):
                    self.bootstrap_group(parent, "0::" + parent + "/{unit}\n")

    def test_native_cli_rejects_wrong_or_ambiguous_service_group(self):
        parent = "/system.slice/system-organism.slice"
        for membership in ("0::/system.slice/{unit}\n", "0::" + parent + "/{unit}/child\n",
                           "0::" + parent + "/other.service\n", "0::" + parent + "/{unit}\n0::/other\n",
                           "0::" + parent + "/{unit}\n0::" + parent + "/{unit}\n"):
            with self.subTest(membership=membership), self.assertRaisesRegex(ValueError, "service group"):
                self.bootstrap_group(parent, membership)
        for supplied in ("", "/unapproved/{unit}", "/system.slice/other.service"):
            with self.subTest(supplied=supplied), self.assertRaisesRegex(ValueError, "service group"):
                self.bootstrap_group(parent, "0::" + parent + "/{unit}\n", supplied=supplied)


class InferenceLogTests(unittest.IsolatedAsyncioTestCase):
    async def test_log_limit_stops_process_but_drains_remaining_pipe_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = configuration(root)
            server = InferenceServer(config, root)
            reader = asyncio.StreamReader()
            reader.feed_data(b"too much fixture output")
            reader.feed_eof()
            stops = []
            server.process = SimpleNamespace(stdout=reader, returncode=None, terminate=lambda: stops.append(True))
            path = root / "bounded.log"
            with patch("organism.inference_server.MAX_LOG_BYTES", 4):
                await server._drain_log(path.open("xb"))
            self.assertEqual(stops, [True])
            self.assertTrue(reader.at_eof())
            self.assertLessEqual(path.stat().st_size, 4)
            self.assertEqual(server.failure, "Inference log allowance exhausted")


if __name__ == "__main__":
    unittest.main()
