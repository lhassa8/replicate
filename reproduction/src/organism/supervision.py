"""Trusted local lifecycle coordinator; deliberately no peer activation RPC.

ProcessSpec must come from an owner-controlled verified-package preparer.
The native package preparer and useful-model readiness verifier remain separate
work. This coordinator does not infer readiness from an alive process.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import os
import stat
import sqlite3

from .receiver_state import AdmissionError
from .readiness_supervision import ReadinessBackend
from .supervision_records import RecordError
from .systemd_supervision import ObservationUncertain, SubmissionUncertain, SupervisionError


class SupervisorBusy(SupervisionError):
    pass


class Supervisor:
    def __init__(self, receiver, backend):
        self.receiver = receiver
        self.state = receiver.state if receiver is not None else backend.state
        self.backend = backend
        if backend.state is not self.state:
            raise SupervisionError("Receiver and supervisor must share the same local ownership store")
        self.readiness = ReadinessBackend(backend)

    @contextmanager
    def _locked(self):
        path = self.state.directory / "supervisor.lock"
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            entry = os.fstat(fd)
            if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.geteuid()
                    or entry.st_nlink != 1 or stat.S_IMODE(entry.st_mode) & 0o077):
                raise SupervisionError("Supervisor lock must be private to its owner")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SupervisorBusy("Another local supervisor operation is in progress") from exc
            yield
        finally:
            os.close(fd)

    def start(self, operation_id, manifest, prepared_spec):
        if self.receiver is None:
            raise SupervisionError("This control-only supervisor cannot start applications")
        with self._locked():
            snapshot = self.state.snapshot()
            birth = snapshot["current_birth"]
            if not birth or birth["operation_id"] != operation_id:
                raise AdmissionError("No matching birth owns this device")
            if birth["process_token"]:
                # A retry or receiver restart observes this exact intent. It
                # never resubmits start because a previous reply was lost.
                return self._reconcile()
            self.backend.preflight(prepared_spec)
            birth = self.receiver.verify_before_start(operation_id, manifest)
            error = None
            try:
                self.backend.install_and_start(birth, prepared_spec)
            except (SupervisionError, RecordError, OSError, sqlite3.DatabaseError) as exc:
                error = type(exc).__name__ + ": " + str(exc)
            return self._reconcile(submission_error=error)

    def start_prepared(self, operation_id, manifest, preparer, work_directory, *,
                       lab_cidrs=("127.0.0.0/24",), user="organism", group="organism"):
        """Owner-local bridge from the verified package preparer to supervision."""
        if self.receiver is None:
            raise SupervisionError("This control-only supervisor cannot prepare applications")
        if preparer.policy.policy_id != self.state.policy.policy_id:
            raise SupervisionError("Package preparer uses another enrollment policy")
        snapshot = self.state.snapshot()
        birth = snapshot["current_birth"]
        if not birth or birth["operation_id"] != operation_id or snapshot["owner_mode"] != "enabled":
            raise AdmissionError("No authorized birth is available for preparation")
        if hashlib.sha256(manifest).hexdigest() != birth["manifest_sha256"]:
            raise AdmissionError("Preparation manifest differs from the reserved birth")
        if birth["process_token"]:
            return self.reconcile()
        prepared = preparer.prepare(self.receiver.store / birth["release_id"], manifest, self.state.platform)
        spec = prepared.process_spec(work_directory, lab_cidrs=lab_cidrs, user=user, group=group)
        return self.start(operation_id, manifest, spec)

    def reconcile(self):
        """One bounded observation; caller schedules later checks if pending."""
        with self._locked():
            return self._reconcile()

    def owner_control(self, action, reason):
        # Persist the fence even if a start operation currently holds the
        # supervisor lock. ExecCondition then denies a delayed start.
        control = self.state.owner_control(action, reason)
        try:
            observed = self.reconcile()
        except SupervisorBusy:
            observed = {"state": "busy", "process_exit_verified": False}
        return {"control": control, "supervision": observed}

    def _reconcile(self, *, submission_error=None):
        snapshot = self.state.snapshot()
        birth = snapshot["current_birth"]
        if not birth or not birth["process_token"]:
            return {"state": "idle", "owner_mode": snapshot["owner_mode"],
                    "restart_required": bool(birth and birth["restart_required"])}
        try:
            checker_stopped = None
            checker_error = None
            stop_submitted = snapshot["owner_mode"] != "enabled" or birth["status"] == "stop_requested"
            if stop_submitted:
                try:
                    checker_stopped = self.readiness.stop(birth)
                except (SupervisionError, RecordError, OSError) as exc:
                    checker_error = exc
                # Observation problems must not prevent a stop request for the
                # exact bound unit. Exit is still acknowledged only on proof.
                try:
                    observation = self.backend.stop(birth)
                except SubmissionUncertain as exc:
                    submission_error = str(exc)
                    observation = self.backend.inspect(birth)
            else:
                observation = self.backend.inspect(birth)
            latest = self.state.snapshot()
            if latest["current_birth"] is None or latest["current_birth"]["process_token"] != birth["process_token"]:
                raise ObservationUncertain("Ownership changed during process observation")
            stopping = latest["owner_mode"] != "enabled" or latest["current_birth"]["status"] == "stop_requested"
            if stopping and not stop_submitted and observation["state"] != "absent":
                try:
                    checker_stopped = self.readiness.stop(birth)
                except (SupervisionError, RecordError, OSError) as exc:
                    checker_error = exc
                try:
                    observation = self.backend.stop(birth)
                except SubmissionUncertain as exc:
                    submission_error = str(exc)
                    observation = self.backend.inspect(birth)
            if observation["state"] == "absent":
                # The application and its auxiliary check share one incarnation
                # claim. A checker error must not prevent requesting app stop,
                # but it must prevent releasing that claim.
                if checker_error is not None:
                    raise ObservationUncertain("Checker stop is not confirmed") from checker_error
                if checker_stopped is None:
                    checker_stopped = self.readiness.stop(birth)
                if checker_stopped["state"] != "not_attempted":
                    observation = {**observation, "readiness_observation": checker_stopped}
                # An identical receipt is reusable after a crash before the
                # state transition. Changed observations are never substituted
                # beneath an earlier evidence hash.
                digest = self.backend.records.put(birth, "exit", observation)
                result = self.state.record_process_gone(birth["operation_id"], birth["process_token"], digest)
                return {"state": "absent", "process_exit_verified": True, "evidence_hash": digest,
                        "birth_status": result["status"], "restart_required": result["restart_required"],
                        "native_observation": observation["native_observation"], "submission_error": submission_error}
            if observation["state"] == "running" and not stopping:
                earlier = latest["current_birth"]["process_observation"]
                if earlier is not None and earlier != observation["process"]:
                    raise ObservationUncertain("Observed process differs from the persisted incarnation")
                recorded = self.backend.records.get(birth, "running")
                if recorded is not None and recorded["record"]["observation"]["process"] != observation["process"]:
                    raise ObservationUncertain("Observed process differs from the recorded start receipt")
                if earlier is None:
                    if recorded is None:
                        self.backend.records.put(birth, "running", observation)
                    try:
                        self.state.record_running(birth["operation_id"], birth["process_token"], observation["process"])
                    except AdmissionError:
                        # Owner controls or expiry may have raced the sample.
                        # Do not report a successful running transition.
                        return {"state": "pending", "process_exit_verified": False,
                                "reason": "Ownership changed before running acknowledgement"}
                current = self.state.snapshot()["current_birth"]
                readiness = self._check_readiness(current, observation)
                return {"state": "running", "process": observation["process"],
                        "native_observation": observation["native_observation"],
                        **readiness, "submission_error": submission_error}
            return {"state": "pending", "process_exit_verified": False, "submission_error": submission_error}
        except (SupervisionError, RecordError, OSError, sqlite3.DatabaseError) as exc:
            # An observer failure never releases a process claim or starts a
            # replacement. The owner fence, if set, remains durable.
            return {"state": "uncertain", "process_exit_verified": False,
                    "reason": type(exc).__name__ + ": " + str(exc), "submission_error": submission_error}

    def _check_readiness(self, birth, observation):
        if birth is None or not birth["process_token"]:
            raise ObservationUncertain("Application ownership changed before readiness check")
        if birth["status"] not in ("running", "ready") or self.state.snapshot()["owner_mode"] != "enabled":
            return {"useful_readiness_verified": False, "readiness_state": "ownership_changed"}
        if self.readiness._intent(birth) is None:
            # A control/status invocation observes existing work; only the
            # receiving service may initiate the carried checker.
            if self.receiver is None or not self.readiness.start(birth):
                return {"useful_readiness_verified": False, "readiness_state": "not_attempted"}
            return {"useful_readiness_verified": False, "readiness_state": "pending"}
        record = self.readiness.result(birth)
        if record is None:
            return {"useful_readiness_verified": False, "readiness_state": "pending"}
        receipt = record["record"]["observation"]
        if not receipt["passed"]:
            return {"useful_readiness_verified": False, "readiness_state": "failed",
                    "readiness_failure": receipt["failure"], "readiness_result_hash": record["digest"]}
        admitted = self.backend.records.get(birth, "check_admission")
        if birth["status"] == "ready":
            if admitted is None or birth["ready_evidence"] != admitted["digest"]:
                raise ObservationUncertain("Readiness acknowledgement differs from owner evidence")
            return {"useful_readiness_verified": True, "readiness_state": "ready",
                    "readiness_evidence_hash": admitted["digest"]}
        if not self.readiness.fresh(birth):
            return {"useful_readiness_verified": False, "readiness_state": "expired"}
        final = self.backend.inspect(birth)
        if (final["state"] != "running" or final["process"] != birth["process_observation"]
                or final["process"] != observation["process"]
                or final["native_observation"] != receipt["native_observation"]):
            raise ObservationUncertain("Application changed before readiness acknowledgement")
        if admitted is None:
            self.backend.records.put(birth, "check_admission", {
                "check_result_hash": record["digest"], "application_observation": final,
                "native_observation": final["native_observation"], "qualification_passed": False,
            })
            admitted = self.backend.records.get(birth, "check_admission")
        elif (admitted["record"]["observation"]["check_result_hash"] != record["digest"]
              or admitted["record"]["observation"]["application_observation"]["process"] != final["process"]):
            raise ObservationUncertain("Persisted readiness admission differs")
        # The ownership store checks the current token, owner fence and deadline
        # atomically. A late stop or retirement cannot become a ready receipt.
        if not self.readiness.fresh(birth):
            return {"useful_readiness_verified": False, "readiness_state": "expired"}
        try:
            self.state.record_ready(birth["operation_id"], birth["process_token"], admitted["digest"],
                                    valid_until=self.readiness._intent(birth)["deadline_clock"])
        except AdmissionError:
            return {"useful_readiness_verified": False, "readiness_state": "ownership_changed"}
        return {"useful_readiness_verified": True, "readiness_state": "ready",
                "readiness_evidence_hash": admitted["digest"]}
