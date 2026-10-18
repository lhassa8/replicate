import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import hmac
import json
import math
import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from organism.capsule import CapsuleError, SCHEMA as CAPSULE_SCHEMA, canonical, sign_manifest, stage_capsule
from organism.enrollment import Enrollment, EnrollmentError, SCHEMA, key_id, public_bytes, sign_enrollment
from organism.receiver import LocalReceiver
from organism.receiver_state import AdmissionError, ReceiverState


def child_reservation(directory, signed, owner, device, parent, operation, manifest, ticket, barrier, queue, die=False):
    policy = Enrollment(signed, owner)
    state = ReceiverState(directory, policy, device, boot_id="fixture-boot-a", clock=lambda: 100.0)
    if barrier is not None:
        barrier.wait(timeout=10)
    try:
        result = state.reserve_birth(parent, operation, 0, manifest, admission_ticket=ticket)
        if die:
            os._exit(7)  # Committed write, deliberately lost reply; not a real worker crash.
        queue.put(("accepted", result["operation_id"]))
    except AdmissionError as exc:
        queue.put(("rejected", str(exc)))


class ReceiverStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = 100.0
        self.owner_key = Ed25519PrivateKey.generate()
        self.owner_public = public_bytes(self.owner_key.public_key())
        self.release_key = Ed25519PrivateKey.generate()
        release_public = public_bytes(self.release_key.public_key())
        self.device_keys = [Ed25519PrivateKey.generate() for _ in range(3)]
        self.identities = [key_id(public_bytes(key.public_key())) for key in self.device_keys]
        self.parent, self.device, self.other_parent = self.identities
        self.body = {
            "schema": SCHEMA, "experiment_id": "receiver-unit-fixture-not-qualification", "epoch": 1,
            "devices": {identity: {
                "public_key": base64.b64encode(public_bytes(key.public_key())).decode(),
                "platform": "linux-arm64" if index == 1 else "linux-x86_64",
            } for index, (identity, key) in enumerate(zip(self.identities, self.device_keys))},
            "release_keys": {key_id(release_public): base64.b64encode(release_public).decode()},
            "require_trained": True, "max_payload_bytes": 1024**2, "birth_seconds": 60,
            "max_births_per_device": 8, "max_process_starts": 3,
        }
        self.policy = Enrollment(sign_enrollment(self.body, self.owner_key), self.owner_public)
        self.source = self.root / "source"
        self.source.mkdir()
        self.capsule_body = {
            "schema": CAPSULE_SCHEMA, "experiment_id": self.body["experiment_id"],
            "model": {"path": "model/model.gguf", "parameters": 3_000_000_000, "trained_for_organism": True,
                      "source_model": "tiny-unit-fixture-not-model-weights", "source_revision": "test",
                      "quantization": "not-applicable"},
            "agent": "common/agent.tar", "knowledge": "common/knowledge.tar",
            "self_test": "common/check.tar", "notices": "common/notices.tar",
            "platforms": {platform: {kind: f"{platform}/{kind}.tar" for kind in ("python", "inference", "dependencies")}
                          for platform in ("linux-x86_64", "linux-arm64")}, "files": {},
        }
        paths = [self.capsule_body["model"]["path"]] + [self.capsule_body[k] for k in ("agent", "knowledge", "self_test", "notices")]
        for value in self.capsule_body["platforms"].values():
            paths.extend(value.values())
        for name in paths:
            data = ("Unit fixture, not an executable model or runtime: " + name).encode()
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.capsule_body["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.manifest = sign_manifest(self.capsule_body, self.release_key)
        self.state_path = self.root / "state"
        self.state = ReceiverState.enroll(self.state_path, self.policy, self.device,
                                         boot_id="fixture-boot-a", clock=lambda: self.now)
        self.operation = "a" * 32
        self.initial_ticket = self.state.advertisement()["admission_ticket"]

    def reopen(self, *, boot_id="fixture-boot-a", policy=None):
        return ReceiverState(self.state_path, policy or self.policy, self.device,
                             boot_id=boot_id, clock=lambda: self.now)

    def reserve(self, **changes):
        observed = self.state.advertisement()
        values = {"authenticated_peer": self.parent, "operation_id": self.operation,
                  "observed_revision": observed["revision"], "manifest": self.manifest,
                  "admission_ticket": observed["admission_ticket"]}
        values.update(changes)
        return self.state.reserve_birth(**values)

    def start(self, *, ready=False):
        birth = self.reserve()
        self.state.record_staged(self.operation, birth["manifest_sha256"])
        birth = self.state.claim_start(self.operation)
        self.state.record_running(self.operation, birth["process_token"],
                                  {"pid": 12345, "start_identity": "unit-fixture-no-live-process", "boot_id": "fixture-boot-a"})
        if ready:
            birth = self.state.record_ready(self.operation, birth["process_token"], "1" * 64)
        return birth

    def test_owner_key_and_signature_are_required(self):
        with self.assertRaises(EnrollmentError):
            Enrollment(self.policy.signed, public_bytes(Ed25519PrivateKey.generate().public_key()))
        changed = json.loads(self.policy.signed)
        changed["body"]["max_process_starts"] = 8
        with self.assertRaises(EnrollmentError):
            Enrollment(canonical(changed), self.owner_public)

    def test_more_than_six_devices_and_wrong_key_id_are_rejected(self):
        changed = deepcopy(self.body)
        for _ in range(4):
            raw = public_bytes(Ed25519PrivateKey.generate().public_key())
            changed["devices"][key_id(raw)] = {"public_key": base64.b64encode(raw).decode(), "platform": "linux-arm64"}
        with self.assertRaisesRegex(EnrollmentError, "six"):
            sign_enrollment(changed, self.owner_key)
        changed = deepcopy(self.body)
        changed["devices"][self.parent]["public_key"] = changed["devices"][self.device]["public_key"]
        with self.assertRaisesRegex(EnrollmentError, "identity"):
            sign_enrollment(changed, self.owner_key)

    def test_existing_enrollment_cannot_be_reset_or_silently_changed(self):
        self.state.owner_control("retire", "test retirement")
        with self.assertRaises(FileExistsError):
            ReceiverState.enroll(self.state_path, self.policy, self.device, boot_id="fixture-boot-a", clock=lambda: self.now)
        changed = deepcopy(self.body)
        changed["epoch"] = 2
        other_policy = Enrollment(sign_enrollment(changed, self.owner_key), self.owner_public)
        with self.assertRaisesRegex(AdmissionError, "migration"):
            self.reopen(policy=other_policy)
        self.assertEqual(self.reopen().snapshot()["owner_mode"], "retired")

    def test_missing_database_is_not_recreated(self):
        self.state_path.joinpath("state.sqlite3").unlink()
        with self.assertRaises(FileNotFoundError):
            self.reopen()
        self.assertFalse(self.state_path.joinpath("state.sqlite3").exists())

    def test_unknown_peer_self_parent_and_wrong_experiment_are_rejected(self):
        with self.assertRaises(EnrollmentError):
            self.reserve(authenticated_peer="f" * 64)
        with self.assertRaisesRegex(AdmissionError, "itself"):
            self.reserve(authenticated_peer=self.device)
        changed = deepcopy(self.capsule_body)
        changed["experiment_id"] = "unrelated"
        with self.assertRaisesRegex(AdmissionError, "experiment"):
            self.reserve(manifest=sign_manifest(changed, self.release_key))
        self.assertTrue(self.state.snapshot()["available"])

    def test_lost_reply_is_idempotent_after_reopen(self):
        first = self.reserve()
        again = self.reopen().reserve_birth(self.parent, self.operation, 0, self.manifest, admission_ticket=self.initial_ticket)
        self.assertEqual(first, again)
        self.assertEqual(self.state.snapshot()["birth_count"], 1)
        self.assertEqual(len(self.state.events()), 1)

    def test_operation_id_cannot_be_reused_by_another_parent_or_release(self):
        self.reserve()
        with self.assertRaisesRegex(AdmissionError, "different intent"):
            self.reserve(authenticated_peer=self.other_parent)
        changed = deepcopy(self.capsule_body)
        changed["model"]["source_revision"] = "different-release"
        with self.assertRaisesRegex(AdmissionError, "different intent"):
            self.reserve(manifest=sign_manifest(changed, self.release_key))

    def test_stale_observation_cannot_reserve_after_owner_stop_and_resume(self):
        self.state.owner_control("stop", "test")
        self.state.owner_control("resume", "test")
        with self.assertRaisesRegex(AdmissionError, "stale"):
            self.reserve(observed_revision=0)
        self.assertEqual(self.reserve()["status"], "receiving")

    def test_two_processes_cannot_own_one_device(self):
        context = multiprocessing.get_context("spawn")
        barrier, queue = context.Barrier(2), context.Queue()
        processes = [context.Process(target=child_reservation, args=(
            str(self.state_path), self.policy.signed, self.owner_public, self.device, parent,
            operation, self.manifest, self.initial_ticket, barrier, queue,
        )) for parent, operation in ((self.parent, "a" * 32), (self.other_parent, "b" * 32))]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=15)
                self.assertEqual(process.exitcode, 0)
            outcomes = [queue.get(timeout=2) for _ in processes]
            self.assertEqual(sorted(item[0] for item in outcomes), ["accepted", "rejected"])
            self.assertEqual(self.reopen().snapshot()["birth_count"], 1)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            queue.close()

    def test_committed_reservation_survives_process_exit_before_reply(self):
        context = multiprocessing.get_context("spawn")
        process = context.Process(target=child_reservation, args=(
            str(self.state_path), self.policy.signed, self.owner_public, self.device,
            self.parent, self.operation, self.manifest, self.initial_ticket, None, None, True,
        ))
        try:
            process.start()
            process.join(timeout=15)
            self.assertEqual(process.exitcode, 7)
            result = self.reopen().reserve_birth(self.parent, self.operation, 0, self.manifest, admission_ticket=self.initial_ticket)
            self.assertEqual(result["status"], "receiving")
            self.assertEqual(self.state.snapshot()["birth_count"], 1)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    def test_expired_reservation_does_not_silently_renew_on_retry(self):
        first = self.reserve()
        self.now = 161
        self.assertTrue(self.state.snapshot()["available"])
        retried = self.reserve(observed_revision=0)
        self.assertEqual(retried["status"], "aborted")
        self.assertEqual(retried["deadline"], first["deadline"])
        replacement = self.reserve(operation_id="b" * 32)
        self.assertNotEqual(first["individual_id"], replacement["individual_id"])

    def test_expired_started_process_keeps_slot_until_absence_is_verified(self):
        started = self.start()
        self.now = 161
        snapshot = self.state.snapshot()
        self.assertFalse(snapshot["available"])
        self.assertEqual(snapshot["current_birth"]["status"], "stop_requested")
        with self.assertRaises(AdmissionError):
            self.reserve(operation_id="b" * 32)
        self.state.record_process_gone(self.operation, started["process_token"], "2" * 64)
        self.assertTrue(self.state.snapshot()["available"])

    def test_useful_child_does_not_depend_on_parent_lease_or_liveness(self):
        ready = self.start(ready=True)
        self.now = 100_000
        observed = self.reopen().snapshot()["current_birth"]
        self.assertEqual(observed["status"], "ready")
        self.assertEqual(observed["individual_id"], ready["individual_id"])
        with self.assertRaisesRegex(AdmissionError, "independent"):
            self.state.abort_birth(self.parent, self.operation, "parent lost interest")

    def test_restart_requires_old_process_exit_and_uses_new_ownership_token(self):
        ready = self.start(ready=True)
        with self.assertRaises(AdmissionError):
            self.state.claim_start(self.operation)
        self.state.record_process_gone(self.operation, ready["process_token"], "2" * 64)
        restart = self.state.claim_start(self.operation)
        self.assertNotEqual(restart["process_token"], ready["process_token"])
        self.assertEqual(restart["individual_id"], ready["individual_id"])
        with self.assertRaisesRegex(AdmissionError, "another process"):
            self.state.record_process_gone(self.operation, ready["process_token"], "3" * 64)

    def test_receiver_restart_same_boot_does_not_relaunch_running_child(self):
        ready = self.start(ready=True)
        observed = self.reopen().snapshot()["current_birth"]
        self.assertEqual(observed["process_token"], ready["process_token"])
        self.assertEqual(observed["status"], "ready")
        with self.assertRaises(AdmissionError):
            self.reopen().claim_start(self.operation)

    def test_machine_reboot_preserves_useful_identity_but_requires_reverification(self):
        ready = self.start(ready=True)
        self.now = 1.0
        state = self.reopen(boot_id="fixture-boot-b")
        birth = state.snapshot()["current_birth"]
        self.assertEqual(birth["status"], "staged")
        self.assertTrue(birth["restart_required"])
        self.assertIsNone(birth["process_token"])
        self.assertEqual(birth["individual_id"], ready["individual_id"])

    def test_machine_reboot_aborts_uncommitted_birth(self):
        self.reserve()
        self.now = 1
        state = self.reopen(boot_id="fixture-boot-b")
        self.assertTrue(state.snapshot()["available"])
        self.assertEqual(state.peer_receipt(self.parent, self.operation)["status"], "aborted")

    def test_retirement_survives_restart_reboot_and_old_messages(self):
        ready = self.start(ready=True)
        result = self.state.owner_control("retire", "owner request")
        self.assertTrue(result["process_stop_required"])
        state = self.reopen()
        with self.assertRaises(AdmissionError):
            state.record_ready(self.operation, ready["process_token"], "1" * 64)
        with self.assertRaisesRegex(AdmissionError, "permanent"):
            state.owner_control("resume", "late retry")
        state.record_process_gone(self.operation, ready["process_token"], "2" * 64)
        self.now = 1
        state = self.reopen(boot_id="fixture-boot-b")
        self.assertEqual(state.snapshot()["owner_mode"], "retired")
        self.assertFalse(state.snapshot()["available"])
        with self.assertRaisesRegex(AdmissionError, "disabled"):
            state.reserve_birth(self.parent, "b" * 32, state.snapshot()["revision"], self.manifest,
                                admission_ticket=self.initial_ticket)

    def test_checker_evidence_deadline_is_enforced_inside_readiness_transaction(self):
        birth = self.start()
        until = self.now + 1
        self.now = until
        with self.assertRaisesRegex(AdmissionError, "evidence expired"):
            self.state.record_ready(self.operation, birth["process_token"], "1" * 64, valid_until=until)
        self.assertFalse(self.state.snapshot()["current_birth"]["ready_once"])
        for invalid in (True, float("nan"), float("inf"), -1, "later"):
            with self.subTest(invalid=invalid), self.assertRaises(AdmissionError):
                self.state.record_ready(self.operation, birth["process_token"], "1" * 64, valid_until=invalid)
        ready = self.state.record_ready(self.operation, birth["process_token"], "1" * 64, valid_until=until + 1)
        self.assertTrue(ready["ready_once"])

    def test_stop_does_not_claim_process_has_already_exited(self):
        birth = self.start()
        stopped = self.state.owner_control("stop", "operator")
        self.assertTrue(stopped["process_stop_required"])
        with self.assertRaisesRegex(AdmissionError, "old process"):
            self.state.owner_control("resume", "operator")
        self.state.record_process_gone(self.operation, birth["process_token"], "2" * 64)
        self.state.owner_control("resume", "operator")
        self.assertTrue(self.state.snapshot()["available"])

    def test_process_restart_allowance_persists(self):
        ready = self.start(ready=True)
        for _ in range(2):
            self.state.record_process_gone(self.operation, ready["process_token"], "2" * 64)
            ready = self.reopen().claim_start(self.operation)
        self.state.record_process_gone(self.operation, ready["process_token"], "2" * 64)
        with self.assertRaisesRegex(AdmissionError, "allowance"):
            self.reopen().claim_start(self.operation)
        self.assertFalse(self.state.snapshot()["available"])

    def test_public_receipts_do_not_expose_process_ownership_token(self):
        self.start(ready=True)
        self.assertNotIn("process_token", self.state.peer_receipt(self.parent, self.operation))
        self.assertNotIn("current_birth", self.state.advertisement())
        with self.assertRaisesRegex(AdmissionError, "addressed"):
            self.state.peer_receipt(self.other_parent, self.operation)

    def test_journal_preserves_hash_chain_and_detects_truncation(self):
        self.reserve()
        self.state.abort_birth(self.parent, self.operation, "fixture cancellation")
        self.assertEqual(len(self.reopen().events()), 2)
        connection = sqlite3.connect(self.state_path / "state.sqlite3")
        with connection:
            connection.execute("DELETE FROM events WHERE sequence=2")
        connection.close()
        with self.assertRaisesRegex(AdmissionError, "incomplete"):
            self.reserve(operation_id="b" * 32)
        with self.assertRaisesRegex(AdmissionError, "incomplete"):
            self.reopen().events()

    def test_clock_rollback_closes_admission_but_retirement_still_works(self):
        self.reserve()
        self.now = 50
        with self.assertRaisesRegex(AdmissionError, "backwards"):
            self.state.snapshot()
        self.state.owner_control("retire", "clock failure")
        self.assertEqual(self.reopen().snapshot()["owner_mode"], "retired")

    def test_local_receiver_recovers_copy_complete_ack_missing(self):
        receiver = LocalReceiver(self.state, self.root / "capsules")
        with patch.object(self.state, "record_staged", side_effect=OSError("simulated death after publish")):
            with self.assertRaises(OSError):
                receiver.receive(self.parent, self.operation, 0, self.manifest, self.source,
                                 admission_ticket=self.initial_ticket)
        recovered = LocalReceiver(self.reopen(), self.root / "capsules")
        receipt = recovered.receive(self.parent, self.operation, 0, self.manifest, self.source,
                                    admission_ticket=self.initial_ticket)
        self.assertEqual(receipt["status"], "staged")
        releases = [path for path in (self.root / "capsules").iterdir() if not path.name.startswith(".")]
        self.assertEqual(len(releases), 1)
        self.assertEqual(self.state.snapshot()["birth_count"], 1)

    def test_retirement_during_copy_prevents_start(self):
        receiver = LocalReceiver(self.state, self.root / "capsules")

        def retire_after_copy(*args, **kwargs):
            result = stage_capsule(*args, **kwargs)
            self.state.owner_control("retire", "operator during transfer")
            return result

        with patch("organism.receiver.stage_capsule", side_effect=retire_after_copy):
            with self.assertRaises(AdmissionError):
                receiver.receive(self.parent, self.operation, 0, self.manifest, self.source,
                                 admission_ticket=self.initial_ticket)
        with self.assertRaises(AdmissionError):
            receiver.verify_before_start(self.operation, self.manifest)
        self.assertEqual(self.reopen().snapshot()["owner_mode"], "retired")

    def test_corrupt_stored_payload_prevents_process_claim(self):
        receiver = LocalReceiver(self.state, self.root / "capsules")
        receipt = receiver.receive(self.parent, self.operation, 0, self.manifest, self.source,
                                   admission_ticket=self.initial_ticket)
        model = receiver.store / receipt["release_id"] / "model/model.gguf"
        model.write_bytes(b"corruption")
        with self.assertRaises(CapsuleError):
            receiver.verify_before_start(self.operation, self.manifest)
        self.assertIsNone(self.state.snapshot()["current_birth"]["process_token"])

    def test_receiving_store_budget_refuses_and_releases_reservation(self):
        receiver = LocalReceiver(self.state, self.root / "capsules", max_store_bytes=1)
        with self.assertRaisesRegex(AdmissionError, "allowance"):
            receiver.receive(self.parent, self.operation, 0, self.manifest, self.source,
                             admission_ticket=self.initial_ticket)
        self.assertTrue(self.state.snapshot()["available"])
        self.assertEqual(self.state.peer_receipt(self.parent, self.operation)["status"], "aborted")

    def test_unchanged_revision_does_not_make_old_observation_fresh(self):
        self.now = 131
        self.assertEqual(self.state.snapshot()["revision"], 0)
        with self.assertRaisesRegex(AdmissionError, "expired"):
            self.state.reserve_birth(self.parent, self.operation, 0, self.manifest,
                                     admission_ticket=self.initial_ticket)
        self.assertTrue(self.state.snapshot()["available"])
        self.assertEqual(self.reserve()["status"], "receiving")

    def test_observation_expiry_cannot_be_extended_by_peer(self):
        value = json.loads(base64.b64decode(self.initial_ticket))
        value["body"]["expires"] = 200
        altered = base64.b64encode(canonical(value)).decode()
        with self.assertRaisesRegex(AdmissionError, "invalid"):
            self.reserve(admission_ticket=altered)

    def test_fractional_observation_remains_valid_until_its_deadline(self):
        self.now = 100.123
        observed = self.state.advertisement()
        body = json.loads(base64.b64decode(observed["admission_ticket"]))["body"]
        self.now = math.nextafter(body["expires"], -math.inf)
        result = self.state.reserve_birth(self.parent, self.operation, observed["revision"], self.manifest,
                                          admission_ticket=observed["admission_ticket"])
        self.assertEqual(result["status"], "receiving")

    def test_fractional_observation_expires_at_its_exact_deadline(self):
        self.now = 100.123
        observed = self.state.advertisement()
        self.now = json.loads(base64.b64decode(observed["admission_ticket"]))["body"]["expires"]
        with self.assertRaisesRegex(AdmissionError, "expired"):
            self.state.reserve_birth(self.parent, self.operation, observed["revision"], self.manifest,
                                     admission_ticket=observed["admission_ticket"])
        self.assertTrue(self.state.snapshot()["available"])

    def test_authenticated_observation_rejects_one_float_step_deadline_changes(self):
        self.now = 100.123
        observed = self.state.advertisement()
        with sqlite3.connect(self.state.path) as db:
            secret = bytes.fromhex(json.loads(db.execute("SELECT body FROM state").fetchone()[0])["observation_secret"])
        for direction in (-math.inf, math.inf):
            with self.subTest(direction=direction):
                value = json.loads(base64.b64decode(observed["admission_ticket"]))
                value["body"]["expires"] = math.nextafter(value["body"]["expires"], direction)
                value["mac"] = hmac.new(secret, canonical(value["body"]), hashlib.sha256).hexdigest()
                altered = base64.b64encode(canonical(value)).decode()
                with self.assertRaisesRegex(AdmissionError, "expired"):
                    self.state.reserve_birth(self.parent, self.operation, observed["revision"], self.manifest,
                                             admission_ticket=altered)

    def test_retry_of_committed_operation_does_not_require_a_new_ticket(self):
        first = self.reserve(admission_ticket=self.initial_ticket)
        self.now = 131
        again = self.state.reserve_birth(self.parent, self.operation, 0, self.manifest,
                                        admission_ticket=self.initial_ticket)
        self.assertEqual(first["individual_id"], again["individual_id"])
        self.assertEqual(again["deadline"], 160)

    def test_process_exit_acknowledgement_is_idempotent_before_restart(self):
        ready = self.start(ready=True)
        first = self.state.record_process_gone(self.operation, ready["process_token"], "2" * 64)
        before = self.state.snapshot()["revision"]
        again = self.reopen().record_process_gone(self.operation, ready["process_token"], "2" * 64)
        self.assertEqual(first, again)
        self.assertEqual(self.state.snapshot()["revision"], before)

    def test_repeated_stop_still_reports_pending_process_exit(self):
        self.start(ready=True)
        self.state.owner_control("stop", "operator")
        self.assertTrue(self.reopen().owner_control("stop", "operator")["process_stop_required"])

    def test_cancelled_unfinished_birth_does_not_restart_after_exit(self):
        birth = self.start()
        cancelled = self.state.abort_birth(self.parent, self.operation, "parent cancels unfinished transfer")
        self.assertEqual(cancelled["status"], "stop_requested")
        exited = self.state.record_process_gone(self.operation, birth["process_token"], "2" * 64)
        self.assertEqual(exited["status"], "aborted")
        self.assertTrue(self.reopen().snapshot()["available"])
        with self.assertRaises(AdmissionError):
            self.state.claim_start(self.operation)

    def test_unlisted_file_in_published_release_prevents_start(self):
        receiver = LocalReceiver(self.state, self.root / "capsules")
        receipt = receiver.receive(self.parent, self.operation, 0, self.manifest, self.source,
                                   admission_ticket=self.initial_ticket)
        (receiver.store / receipt["release_id"] / "unlisted.py").write_text("unlisted test fixture")
        with self.assertRaisesRegex(CapsuleError, "Unlisted"):
            receiver.verify_before_start(self.operation, self.manifest)
        self.assertIsNone(self.state.snapshot()["current_birth"]["process_token"])

    def test_journal_reserve_keeps_retirement_available(self):
        with patch("organism.receiver_state.MAX_EVENTS", 10):
            self.reserve()
            self.state.abort_birth(self.parent, self.operation, "uses last ordinary event")
            self.assertFalse(self.state.advertisement()["available"])
            with self.assertRaisesRegex(AdmissionError, "journal"):
                self.reserve(operation_id="b" * 32)
            self.state.owner_control("retire", "reserved owner-control capacity")
            self.assertEqual(self.reopen().snapshot()["owner_mode"], "retired")

    def test_previously_staged_release_must_fit_current_store_allowance(self):
        receiver = LocalReceiver(self.state, self.root / "capsules")
        receiver.receive(self.parent, self.operation, 0, self.manifest, self.source,
                         admission_ticket=self.initial_ticket)
        constrained = LocalReceiver(self.reopen(), self.root / "capsules", max_store_bytes=1)
        with self.assertRaisesRegex(AdmissionError, "allowance"):
            constrained.verify_before_start(self.operation, self.manifest)
        self.assertIsNone(self.state.snapshot()["current_birth"]["process_token"])

    def test_start_rejects_replacement_manifest_with_same_signed_body(self):
        receiver = LocalReceiver(self.state, self.root / "capsules")
        receipt = receiver.receive(self.parent, self.operation, 0, self.manifest, self.source,
                                   admission_ticket=self.initial_ticket)
        replacement = json.dumps(json.loads(self.manifest), indent=2).encode()
        (receiver.store / receipt["release_id"] / ".manifest.json").write_bytes(replacement)
        with self.assertRaisesRegex(AdmissionError, "manifest differs"):
            receiver.verify_before_start(self.operation, replacement)
