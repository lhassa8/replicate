"""Scripted systemd/proc/cgroup fixtures on Mac; NOT native Linux evidence."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import threading
import unittest
from unittest.mock import patch

from organism import start_guard
from organism.device_resources import service_cgroup
from organism.receiver import LocalReceiver
from organism.receiver_state import AdmissionError
from organism.supervision import Supervisor, SupervisorBusy
from organism.supervision_records import RecordError, SupervisionRecords
from organism.systemd_supervision import (
    MEMORY_MAX, SHOW_FIELDS, TASKS_MAX, NativeLinuxHost, ObservationUncertain, ProcessSpec,
    SupervisionError, SystemdBackend, description, parse_show, process_start_ticks, render_unit, unit_name,
)
from organism.tests import test_receiver_state as fixture_module


class FixtureLinuxHost:
    """No subprocess execution: explicit fake service properties and /proc."""
    is_fixture = True

    def __init__(self, root, state):
        self.owner_uid = os.geteuid()
        self.boot_id = state.boot_id
        self.machine = "aarch64"
        self.proc_root = root / "fixture-proc"
        self.cgroup_root = root / "fixture-cgroup"
        self.unit_root = root / "fixture-systemd"
        for directory in (self.proc_root, self.cgroup_root, self.unit_root):
            directory.mkdir()
        self.state = state
        self.calls = []
        self.values = {}
        self.start_mode = "running"
        self.stop_mode = "absent"
        self.show_error = False
        self.after_start = None

    def observation(self, name, state="running", *, pid=12345, invocation="1" * 32, ticks="987"):
        birth = self.state.snapshot()["current_birth"]
        path = self.unit_root / name
        exists = path.exists()
        selected_slice = "system.slice"
        if exists:
            selected_slice = next(line.removeprefix("Slice=") for line in path.read_text().splitlines()
                                  if line.startswith("Slice="))
        cgroup = service_cgroup(name, None if selected_slice == "system.slice" else selected_slice)
        populated = state in ("running", "pending", "remaining_child")
        self.values = {key: "" for key in SHOW_FIELDS}
        self.values.update({
            "Id": name, "Description": description(birth) if exists else name,
            "LoadState": "loaded" if exists else "not-found",
            "ActiveState": "active" if state == "running" else ("activating" if state == "pending" else "inactive"),
            "SubState": "running" if state == "running" else ("start" if state == "pending" else "dead"),
            "MainPID": str(pid) if state == "running" else "0", "InvocationID": invocation,
            "ControlGroup": cgroup if exists else "", "Job": "42" if state == "pending" else "0",
            "Result": "success", "FragmentPath": str(path) if exists else "", "DropInPaths": "", "NeedDaemonReload": "no",
            "User": "organism", "Group": "organism", "Restart": "no", "KillMode": "control-group",
            "MemoryMax": str(MEMORY_MAX), "MemorySwapMax": "0", "TasksMax": str(TASKS_MAX),
            "CPUQuotaPerSecUSec": "4s", "NoNewPrivileges": "yes", "ProtectSystem": "strict", "Delegate": "no",
            "Slice": selected_slice,
        })
        group = self.cgroup_root / cgroup.lstrip("/")
        group.mkdir(parents=True, exist_ok=True)
        (group / "cgroup.events").write_text(f"populated {int(populated)}\nfrozen 0\n")
        for field, value in {"memory.max": str(MEMORY_MAX), "memory.swap.max": "0",
                             "pids.max": str(TASKS_MAX), "cpu.max": "400000 100000"}.items():
            (group / field).write_text(value + "\n")
        if state == "running":
            process = self.proc_root / str(pid)
            process.mkdir(exist_ok=True)
            (process / "stat").write_text(f"{pid} (fixture ) child) S " + "0 " * 18 + ticks + " 0\n")
            (process / "cgroup").write_text("0::" + cgroup + "\n")

    def run(self, arguments, *, timeout=15):
        self.calls.append(arguments)
        operation = arguments[1]
        if operation == "daemon-reload":
            return subprocess.CompletedProcess(arguments, 0, "", "")
        if operation == "show":
            if self.show_error:
                raise subprocess.TimeoutExpired(arguments, timeout)
            if not self.values or self.values["Id"] != arguments[2]:
                self.observation(arguments[2], "absent")
            output = "\n".join(key + "=" + value for key, value in self.values.items()) + "\n"
            return subprocess.CompletedProcess(arguments, 0, output, "")
        if operation == "start":
            mode = "running" if self.start_mode == "timeout_running" else self.start_mode
            self.observation(arguments[2], mode)
            if self.after_start:
                self.after_start()
            if self.start_mode == "timeout_running":
                raise subprocess.TimeoutExpired(arguments, timeout)
            return subprocess.CompletedProcess(arguments, 0, "", "")
        if operation == "stop":
            mode = "remaining_child" if self.stop_mode == "timeout_child" else self.stop_mode
            self.observation(arguments[2], mode)
            if self.stop_mode == "timeout_child":
                raise subprocess.TimeoutExpired(arguments, timeout)
            return subprocess.CompletedProcess(arguments, 0, "", "")
        raise AssertionError("Unexpected fixture operation")


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.ReceiverStateTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.state = f.state
        self.receiver = LocalReceiver(self.state, f.root / "store")
        observation = self.state.advertisement()
        self.receiver.receive(f.parent, f.operation, observation["revision"], f.manifest, f.source,
                              admission_ticket=observation["admission_ticket"])
        self.host = FixtureLinuxHost(f.root, self.state)
        self.guard_python = f.root / "fixture-python"
        self.guard_script = f.root / "owner-start-guard.py"
        self.program = f.root / "fixture-program"
        for path in (self.guard_python, self.program):
            path.write_text("Explicit fixture. This file is never executed.\n")
            path.chmod(0o700)
        self.guard_script.write_text("Owner-installed guard fixture. Not executed by fake systemd.\n")
        self.work = f.root / "work"
        self.work.mkdir()
        self.spec = ProcessSpec((str(self.program), "fixture-only"), self.work, (f.source,))
        self.backend = self.make_backend(self.state)
        self.supervisor = Supervisor(self.receiver, self.backend)

    def make_backend(self, state):
        return SystemdBackend(state, guard_python=self.guard_python, guard_script=self.guard_script, host=self.host)

    def start(self):
        return self.supervisor.start(self.fixture.operation, self.fixture.manifest, self.spec)

    def birth(self):
        return self.state.snapshot()["current_birth"]

    def allowed(self, birth=None, **changes):
        birth = birth or self.birth()
        arguments = {"boot_id": self.state.boot_id, "now": self.fixture.now, "owner_uid": os.geteuid()}
        arguments.update(changes)
        return start_guard.check_start_permission(
            self.state.directory, self.state.policy.policy_id, self.state.device_id,
            birth["operation_id"], birth["process_token"], **arguments)

    def test_start_records_actual_fixture_identity_but_never_useful_readiness(self):
        result = self.start()
        self.assertEqual(result["state"], "running")
        self.assertFalse(result["native_observation"])
        self.assertFalse(result["useful_readiness_verified"])
        self.assertEqual(self.birth()["status"], "running")
        self.assertFalse(self.birth()["ready_once"])
        self.assertEqual(self.birth()["process_observation"]["start_identity"], "1" * 32 + ":987")
        self.assertEqual(self.backend.records.get(self.birth(), "running")["record"]["observation"]["state"], "running")
        self.assertEqual(sum(call[1] == "start" for call in self.host.calls), 1)

    def test_timeout_does_not_discard_a_running_service_or_submit_another_start(self):
        self.host.start_mode = "timeout_running"
        first = self.start()
        birth = self.birth()
        self.assertEqual(first["state"], "running")
        self.assertIn("SubmissionUncertain", first["submission_error"])
        again = self.start()
        self.assertEqual(again["process"], first["process"])
        self.assertEqual(self.birth()["process_token"], birth["process_token"])
        self.assertEqual(sum(call[1] == "start" for call in self.host.calls), 1)

    def test_same_boot_receiver_restart_adopts_only_existing_observed_process(self):
        self.start()
        old = self.birth()
        reopened = self.fixture.reopen()
        backend = self.make_backend(reopened)
        receiver = LocalReceiver(reopened, self.receiver.store)
        restored = Supervisor(receiver, backend)
        result = restored.start(self.fixture.operation, self.fixture.manifest, self.spec)
        self.assertEqual(result["state"], "running")
        self.assertEqual(reopened.snapshot()["current_birth"]["process_token"], old["process_token"])
        self.assertEqual(sum(call[1] == "start" for call in self.host.calls), 1)

    def test_pending_service_and_queued_job_preserve_single_start_claim(self):
        self.host.start_mode = "pending"
        self.assertEqual(self.start()["state"], "pending")
        old = self.birth()
        self.host.values.update(ActiveState="inactive", SubState="dead", MainPID="0", Job="42")
        group = self.host.cgroup_root / "system.slice" / unit_name(old) / "cgroup.events"
        group.write_text("populated 0\n")
        self.assertEqual(self.supervisor.reconcile()["state"], "pending")
        self.assertEqual(self.birth()["process_token"], old["process_token"])
        self.assertEqual(self.birth()["starts"], 1)

    def test_observer_timeout_never_releases_claim(self):
        self.start()
        old = self.birth()
        self.host.show_error = True
        self.assertEqual(self.supervisor.reconcile()["state"], "uncertain")
        self.assertEqual(self.birth()["process_token"], old["process_token"])
        self.assertEqual(self.start()["state"], "uncertain")
        self.assertEqual(sum(call[1] == "start" for call in self.host.calls), 1)

    def test_verified_exit_allows_explicit_recovery_with_same_individual_new_token(self):
        self.start()
        old = self.birth()
        self.host.observation(unit_name(old), "absent")
        exited = self.supervisor.reconcile()
        self.assertEqual(exited["state"], "absent")
        self.assertTrue(exited["process_exit_verified"])
        self.assertTrue(exited["restart_required"])
        self.assertEqual(self.birth()["last_exit"]["evidence_hash"], exited["evidence_hash"])
        self.assertEqual(self.start()["state"], "running")
        self.assertEqual(self.birth()["individual_id"], old["individual_id"])
        self.assertNotEqual(self.birth()["process_token"], old["process_token"])
        self.assertEqual(self.birth()["starts"], 2)
        self.assertFalse(self.allowed(old))

    def test_confirmed_exit_receipt_survives_crash_before_state_acknowledgement(self):
        self.start()
        old = self.birth()
        self.host.observation(unit_name(old), "absent")
        with patch.object(self.state, "record_process_gone", side_effect=RuntimeError("fixture crash")):
            with self.assertRaisesRegex(RuntimeError, "fixture crash"):
                self.supervisor.reconcile()
        receipt = self.backend.records.get(old, "exit")
        result = self.supervisor.reconcile()
        self.assertEqual(result["evidence_hash"], receipt["digest"])
        self.assertIsNone(self.birth()["process_token"])

    def test_owner_retirement_stops_service_and_persists_after_reopen(self):
        self.start()
        old = self.birth()
        result = self.supervisor.owner_control("retire", "owner fixture request")
        self.assertEqual(result["control"]["owner_mode"], "retired")
        self.assertTrue(result["supervision"]["process_exit_verified"])
        self.assertIsNone(self.birth())
        self.assertEqual(self.fixture.reopen().snapshot()["owner_mode"], "retired")
        self.assertFalse(self.allowed(old))
        self.assertEqual(sum(call[1] == "stop" for call in self.host.calls), 1)
        with self.assertRaisesRegex(AdmissionError, "permanent"):
            self.supervisor.owner_control("resume", "not permitted")

    def test_stop_timeout_with_surviving_child_preserves_stop_requested(self):
        self.start()
        old = self.birth()
        self.host.stop_mode = "timeout_child"
        result = self.supervisor.owner_control("stop", "fixture surviving subprocess")
        self.assertEqual(result["supervision"]["state"], "pending")
        self.assertFalse(result["supervision"]["process_exit_verified"])
        self.assertEqual(self.birth()["status"], "stop_requested")
        self.assertEqual(self.birth()["process_token"], old["process_token"])
        with self.assertRaisesRegex(AdmissionError, "stopped before resuming"):
            self.supervisor.owner_control("resume", "too early")
        self.host.stop_mode = "absent"
        self.assertTrue(self.supervisor.reconcile()["process_exit_verified"])
        self.assertEqual(self.supervisor.owner_control("resume", "owner request")["control"]["owner_mode"], "enabled")

    def test_unavailable_observer_does_not_prevent_owner_stop_submission(self):
        self.start()
        old = self.birth()
        self.host.show_error = True
        result = self.supervisor.owner_control("retire", "observer is unavailable")
        self.assertEqual(result["supervision"]["state"], "uncertain")
        self.assertEqual(sum(call[1] == "stop" for call in self.host.calls), 1)
        self.assertEqual(self.birth()["process_token"], old["process_token"])
        self.assertEqual(self.birth()["status"], "stop_requested")
        self.host.show_error = False
        self.assertTrue(self.supervisor.reconcile()["process_exit_verified"])

    def test_bad_running_limits_do_not_prevent_retirement_of_bound_unit(self):
        self.start()
        self.host.values["MemoryMax"] = "infinity"
        result = self.supervisor.owner_control("retire", "effective limits differ")
        self.assertTrue(result["supervision"]["process_exit_verified"])
        self.assertIsNone(self.birth())

    def test_expired_birth_stops_existing_claim_before_releasing_slot(self):
        self.start()
        self.fixture.now = 161
        self.assertEqual(self.supervisor.reconcile()["state"], "absent")
        self.assertIsNone(self.birth())
        self.assertTrue(self.state.snapshot()["available"])

    def test_acknowledged_useful_identity_does_not_expire_with_parent_lease(self):
        self.start()
        old = self.birth()
        # Fixture state only: this test makes no real useful-readiness claim.
        self.state.record_ready(old["operation_id"], old["process_token"], "1" * 64)
        self.fixture.now = 500
        result = self.supervisor.reconcile()
        self.assertEqual(result["state"], "running")
        self.assertEqual(self.birth()["individual_id"], old["individual_id"])
        self.assertEqual(sum(call[1] == "stop" for call in self.host.calls), 0)

    def test_changed_pid_start_time_invocation_or_cgroup_is_not_adopted(self):
        self.start()
        old = self.birth()
        for changes in ({"pid": 23456}, {"ticks": "999"}, {"invocation": "2" * 32}):
            with self.subTest(changes=changes):
                self.host.observation(unit_name(old), **changes)
                self.assertEqual(self.supervisor.reconcile()["state"], "uncertain")
                self.assertEqual(self.birth()["process_observation"], old["process_observation"])
        self.host.observation(unit_name(old))
        (self.host.proc_root / "12345" / "cgroup").write_text("0::/unrelated.service\n")
        self.assertEqual(self.supervisor.reconcile()["state"], "uncertain")

    def test_unit_content_cannot_change_while_reusing_description(self):
        self.start()
        path = self.host.unit_root / unit_name(self.birth())
        path.write_text(path.read_text().replace("MemoryMax=6442450944", "MemoryMax=infinity"))
        self.assertEqual(self.supervisor.reconcile()["state"], "uncertain")
        self.assertIsNotNone(self.birth()["process_token"])

    def test_effective_config_dropin_account_and_loaded_file_are_checked(self):
        self.start()
        old = self.birth()
        for key, value in (("DropInPaths", "/etc/unsafe.conf"), ("NeedDaemonReload", "yes"),
                           ("User", "someone_else"), ("Group", "root"), ("MemoryMax", "infinity"),
                           ("MemorySwapMax", "1"), ("CPUQuotaPerSecUSec", "infinity"),
                           ("KillMode", "process"), ("Restart", "always"), ("Delegate", "yes"),
                           ("FragmentPath", "/unrelated.service")):
            with self.subTest(key=key):
                self.host.observation(unit_name(old))
                self.host.values[key] = value
                self.assertEqual(self.supervisor.reconcile()["state"], "uncertain")
                self.assertIsNotNone(self.birth()["process_token"])

    def test_missing_or_partial_manager_observation_cannot_prove_absence(self):
        self.start()
        self.host.observation(unit_name(self.birth()), "absent")
        del self.host.values["Job"]
        self.assertEqual(self.supervisor.reconcile()["state"], "uncertain")
        self.assertIsNotNone(self.birth()["process_token"])

    def test_manager_limits_alone_do_not_establish_kernel_limits(self):
        self.start()
        old = self.birth()
        group = self.host.cgroup_root / "system.slice" / unit_name(old)
        for field, value in (("memory.max", "max"), ("memory.swap.max", "100"),
                             ("pids.max", "max"), ("cpu.max", "max 100000"),
                             ("cpu.max", "800000 100000")):
            with self.subTest(field=field, value=value):
                self.host.observation(unit_name(old))
                (group / field).write_text(value + "\n")
                self.assertEqual(self.supervisor.reconcile()["state"], "uncertain")
                self.assertIsNotNone(self.birth()["process_token"])
        self.host.observation(unit_name(old))
        (group / "memory.max").unlink()
        self.assertEqual(self.supervisor.reconcile()["state"], "uncertain")

    def test_claim_before_unit_publication_recovers_only_after_manager_and_group_check(self):
        birth = self.receiver.verify_before_start(self.fixture.operation, self.fixture.manifest)
        self.assertFalse((self.host.unit_root / unit_name(birth)).exists())
        # A missing unit file with a remaining cgroup must not release ownership.
        self.host.observation(unit_name(birth), "remaining_child")
        self.assertEqual(self.supervisor.reconcile()["state"], "uncertain")
        self.assertIsNotNone(self.birth()["process_token"])
        self.host.observation(unit_name(birth), "absent")
        self.assertEqual(self.supervisor.reconcile()["state"], "absent")
        self.assertTrue(self.birth()["restart_required"])

    def test_missing_unit_file_does_not_hide_running_manager_state(self):
        self.start()
        (self.host.unit_root / unit_name(self.birth())).unlink()
        self.assertEqual(self.supervisor.reconcile()["state"], "uncertain")
        self.assertIsNotNone(self.birth()["process_token"])

    def test_corrupt_capsule_or_missing_runtime_is_rejected_before_start_claim(self):
        self.program.unlink()
        with self.assertRaisesRegex(SupervisionError, "missing"):
            self.start()
        self.assertIsNone(self.birth()["process_token"])
        self.program.write_text("fixture")
        self.program.chmod(0o700)
        model = self.receiver.store / self.birth()["release_id"] / "model/model.gguf"
        model.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.start()
        self.assertIsNone(self.birth()["process_token"])
        self.assertEqual(sum(call[1] == "start" for call in self.host.calls), 0)

    def test_owner_control_during_start_fences_guard_and_stops_after_observation(self):
        old = {}

        def retire():
            old.update(self.birth())
            self.state.owner_control("retire", "fixture race")
            self.assertFalse(self.allowed(old))

        self.host.after_start = retire
        result = self.start()
        self.assertEqual(result["state"], "absent")
        self.assertEqual(result["birth_status"], "stopped")
        self.assertEqual(self.state.snapshot()["owner_mode"], "retired")

    def test_supervisor_lock_prevents_parallel_launch_but_never_delays_owner_fence(self):
        entered, release = threading.Event(), threading.Event()

        def delay():
            entered.set()
            if not release.wait(timeout=5):
                raise AssertionError("Fixture gate timed out")

        self.host.after_start = delay
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.start)
            try:
                self.assertTrue(entered.wait(timeout=5))
                with self.assertRaises(SupervisorBusy):
                    self.start()
                result = self.supervisor.owner_control("retire", "concurrent owner request")
                self.assertEqual(result["supervision"]["state"], "busy")
                self.assertEqual(self.state.snapshot()["owner_mode"], "retired")
                self.assertFalse(self.allowed())
            finally:
                release.set()
            self.assertEqual(future.result(timeout=5)["state"], "absent")
        self.assertEqual(sum(call[1] == "start" for call in self.host.calls), 1)

    def test_guard_requires_exact_live_start_claim_boot_clock_and_enabled_owner(self):
        old = self.receiver.verify_before_start(self.fixture.operation, self.fixture.manifest)
        self.assertTrue(self.allowed(old))
        self.assertFalse(self.allowed(old, boot_id="another-boot"))
        self.assertFalse(self.allowed(old, now=99))
        self.assertFalse(self.allowed(old, now=float("nan")))
        wrong = {**old, "process_token": "b" * 32}
        self.assertFalse(self.allowed(wrong))
        self.fixture.now = 161
        self.assertFalse(self.allowed(old))
        self.fixture.now = 100
        self.state.owner_control("stop", "fixture")
        self.assertFalse(self.allowed(old))

    def test_guard_denies_second_execution_after_running_acknowledgement(self):
        self.start()
        self.assertFalse(self.allowed())

    def test_guard_rejects_deleted_tail_event(self):
        old = self.receiver.verify_before_start(self.fixture.operation, self.fixture.manifest)
        # Drop the last event: persisted revision still identifies the missing tail.
        with sqlite3.connect(self.state.path) as db:
            db.execute("DELETE FROM events WHERE sequence=(SELECT MAX(sequence) FROM events)")
        self.assertFalse(self.allowed(old))

    def test_guard_rejects_shared_directory_and_changed_marker(self):
        old = self.receiver.verify_before_start(self.fixture.operation, self.fixture.manifest)
        self.state.directory.chmod(0o755)
        self.assertFalse(self.allowed(old))
        self.state.directory.chmod(0o700)
        marker = self.state.directory / "enrolled.json"
        value = json.loads(marker.read_text())
        value["device_id"] = "c" * 64
        marker.write_text(json.dumps(value))
        self.assertFalse(self.allowed(old))

    def test_record_corruption_or_loss_cannot_create_a_fresh_supervisor_history(self):
        self.start()
        path = self.backend.records.path
        with sqlite3.connect(path) as db:
            db.execute("UPDATE records SET body='{}' WHERE kind='intent'")
        with self.assertRaisesRegex(RecordError, "corrupt"):
            SupervisionRecords(self.state)
        self.assertEqual(self.supervisor.reconcile()["state"], "uncertain")
        path.unlink()
        with self.assertRaises(FileExistsError):
            SupervisionRecords(self.state)
        self.assertFalse(path.exists())

    def test_running_receipt_cannot_be_overwritten_by_another_incarnation(self):
        self.start()
        old = self.birth()
        original = self.backend.records.get(old, "running")
        with self.assertRaisesRegex(RecordError, "replaced"):
            self.backend.records.put(old, "running", {"process": {"pid": 23456}})
        self.assertEqual(self.backend.records.get(old, "running"), original)

    def test_no_new_records_for_unchanged_health_polls(self):
        self.start()
        with sqlite3.connect(self.backend.records.path) as db:
            count = db.execute("SELECT count(*) FROM records").fetchone()[0]
        for _ in range(8):
            self.assertEqual(self.supervisor.reconcile()["state"], "running")
        with sqlite3.connect(self.backend.records.path) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM records").fetchone()[0], count)

    def test_profile_escaping_and_owner_guard_are_fixed_and_bounded(self):
        birth = self.receiver.verify_before_start(self.fixture.operation, self.fixture.manifest)
        spec = replace(self.spec, command=(str(self.program), "%i", "$HOME", 'a"b', "a\\b"))
        text = render_unit(birth, spec, state_directory=self.state.directory, policy_id=self.state.policy.policy_id,
                           guard_python=self.guard_python, guard_script=self.guard_script)
        for expected in ("Restart=no", "ExecCondition=+:", "ExecStart=:", "KillMode=control-group",
                         '"%%i"', '"$HOME"', '"a\\"b"', '"a\\\\b"', "MemoryMax=6442450944", "TasksMax=128",
                         "IPAddressDeny=any", "AWS_EC2_METADATA_DISABLED=true", "RuntimeMaxSec=345600"):
            self.assertIn(expected, text)
        for changes in ({"user": "root"}, {"command": (str(self.program), "line\nbreak")},
                        {"runtime_seconds": 96 * 3600 + 1}, {"lab_cidrs": ("8.8.8.0/24",)},
                        {"working_directory": Path("relative")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.spec, **changes).validate()

    def test_native_backend_refuses_mac_and_wrong_enrolled_architecture(self):
        with patch("organism.systemd_supervision.platform.system", return_value="Darwin"):
            with self.assertRaisesRegex(SupervisionError, "Linux"):
                NativeLinuxHost()
        self.host.machine = "x86_64"
        with self.assertRaisesRegex(SupervisionError, "architecture"):
            self.make_backend(self.state)

    def test_strict_process_and_manager_parsers(self):
        self.assertEqual(process_start_ticks("123 (child ) name) S " + "0 " * 18 + "987", 123), "987")
        with self.assertRaises(ObservationUncertain):
            process_start_ticks("123 (child) S 0", 123)
        with self.assertRaises(ObservationUncertain):
            process_start_ticks("124 (child) S " + "0 " * 18 + "987", 123)
        with self.assertRaises(ObservationUncertain):
            parse_show("Id=a\nId=b\n")
        with self.assertRaises(ObservationUncertain):
            parse_show("Id=a\n")


if __name__ == "__main__":
    unittest.main()
