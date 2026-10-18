"""Evaluator-only observation clocks and a real one-call file interruption.

This component neither chooses tool arguments nor grades a diagnosis. A native
trial driver must still enforce the outer deadline and bind the actual release,
process, hardware, resource measurements and saved job. It is not deployed code.
"""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from .capsule import canonical
from .diagnostic_agent import EquipmentTool
from .equipment_interfaces import validate_document
from .qualification_inputs import materialize


CASE_SECONDS = 150
FAULT = {
    "kind": "one_configuration_read_unavailable", "operation": "configure_reader", "attempt": 1,
    "effect": "observation file absent for this call, restored before the next call",
    "charged_to_original_case_budget": True, "native_injection_verified": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def private_directory(path):
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
            and not info.st_mode & 0o077, "Evaluator requires an owned private directory")


class ObservationTrial:
    def __init__(self, case, directory, evidence, *, wall_clock=time.time, monotonic=time.monotonic):
        directory, evidence = Path(directory).absolute(), Path(evidence).absolute()
        private_directory(directory)
        private_directory(evidence.parent)
        require(not any(directory.iterdir()) and not evidence.exists() and not evidence.is_symlink()
                and not evidence.resolve().is_relative_to(directory.resolve())
                and not directory.resolve().is_relative_to(evidence.resolve()),
                "A fresh observation directory and separate new evidence directory are required")
        require(case.get("fault") is None or canonical(case["fault"]) == canonical(FAULT),
                "Unknown observation fault schedule")
        self.wall_clock, self.monotonic = wall_clock, monotonic
        wall, monotonic_start = wall_clock(), monotonic()
        require(type(wall) in (int, float) and math.isfinite(wall)
                and 1000000000 <= wall <= 4000000000
                and type(monotonic_start) in (int, float) and math.isfinite(monotonic_start),
                "Invalid evaluator clock origin")
        self.wall_start, self.monotonic_start, self.epoch = wall, monotonic_start, int(wall)
        document = materialize(case, self.epoch)
        validate_document(document, case["case_id"])
        self.raw = canonical(document)
        self.case_id, self.path = case["case_id"], directory / (case["case_id"] + ".json")
        self.evidence, self.pending = evidence, evidence / "interrupted-observation.json"
        self.fault, self.attempted, self.active, self.closed = deepcopy(case.get("fault")), False, False, False
        self.fault_outcome = None
        self.terminal_timing = None
        evidence.mkdir(mode=0o700)
        require(directory.stat().st_dev == evidence.stat().st_dev,
                "Observation interruption requires one filesystem")
        with self.path.open("xb") as stream:
            stream.write(self.raw)
        self.path.chmod(0o600)
        with (evidence / "observation.json").open("xb") as stream:
            stream.write(self.raw)
        self.record("prepared", case=digest(canonical(case)), observation=digest(self.raw),
                    clock_origin=self.epoch, wall_started=self.wall_start,
                    monotonic_started=self.monotonic_start, seconds_limit=CASE_SECONDS,
                    fault=self.fault, native_execution_verified=False, qualification_passed=False)

    def timing(self):
        wall, mono = self.wall_clock(), self.monotonic()
        require(type(wall) in (int, float) and math.isfinite(wall)
                and type(mono) in (int, float) and math.isfinite(mono), "Evaluator clock is nonfinite")
        elapsed = mono - self.monotonic_start
        require(elapsed >= 0 and abs((wall - self.wall_start) - elapsed) <= 1,
                "Evaluator wall and monotonic clocks diverged")
        return {"wall": wall, "elapsed_seconds": elapsed,
                "remaining_seconds": max(0, CASE_SECONDS - elapsed)}

    def record(self, event, **fields):
        with (self.evidence / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": event, **fields}, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def unchanged(self):
        info = self.path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.geteuid()
                and not info.st_mode & 0o077 and self.path.read_bytes() == self.raw,
                "Trial observation changed or was replaced by an unsafe file")

    def before_tool(self, name, arguments):
        require(not self.closed and self.terminal_timing is None and not self.active,
                "Trial is closed or another fault call is active")
        timing = self.timing()
        require(timing["remaining_seconds"] > 0, "Original diagnostic deadline exhausted")
        self.unchanged()
        if self.fault is None or self.attempted or name != self.fault["operation"]:
            return False
        self.attempted = True
        require(not self.pending.exists(), "Fault recovery file already exists")
        self.record("fault_armed", tool=name, arguments=deepcopy(arguments), **timing)
        os.rename(self.path, self.pending)
        self.active = True
        self.record("fault_active", observation_absent=not self.path.exists())
        return True

    def restore(self):
        require(self.active and self.pending.is_file() and not self.pending.is_symlink()
                and self.pending.read_bytes() == self.raw, "Interrupted observation differs")
        # Never overwrite a file created by another actor during the interruption.
        require(not self.path.exists() and not self.path.is_symlink(), "Observation destination was replaced")
        os.rename(self.pending, self.path)
        self.active = False
        self.unchanged()
        self.record("fault_restored", observation=digest(self.raw), **self.timing())

    def stop(self):
        """Capture completion before independent post-run grading begins."""
        require(not self.closed and not self.active and self.terminal_timing is None,
                "Invalid trial stop")
        self.unchanged()
        self.terminal_timing = self.timing()
        self.record("stopped", **self.terminal_timing)
        return deepcopy(self.terminal_timing)

    def finish(self, *, task_success):
        require(type(task_success) is bool and not self.closed and not self.active, "Invalid trial completion")
        self.unchanged()
        if self.terminal_timing is None:
            self.stop()
        timing = self.terminal_timing
        fault_verified = self.fault is None or (
            self.attempted and self.fault_outcome is not None
            and self.fault_outcome["observed_file_not_found"]
            and self.fault_outcome["configuration_unchanged"])
        success = task_success and fault_verified and timing["elapsed_seconds"] <= CASE_SECONDS
        receipt = {"task_success_reported": task_success, "fault_condition_verified": fault_verified,
                   "within_deadline": timing["elapsed_seconds"] <= CASE_SECONDS,
                   "bounded_effort_seconds": timing["elapsed_seconds"] if success else CASE_SECONDS,
                   "observation_unchanged": True, "qualification_passed": False, **timing}
        self.record("closed", **receipt)
        self.closed = True
        return receipt


class ObservationTool(EquipmentTool):
    """Delegate to the real tool, restoring its input before yielding a result."""

    def __init__(self, session, spec, trial):
        super().__init__(session, spec)
        require(session.device_id == trial.case_id
                and session.directory.directory.resolve() == trial.path.parent.resolve(),
                "Fault tool belongs to another trial observation")
        self.trial = trial

    async def stream(self, tool_use, invocation_state, **kwargs):
        before = deepcopy(self.session.configuration)
        require(not self.trial.active, "Another fault call is active")
        interrupted = False
        events = []
        try:
            interrupted = self.trial.before_tool(self.tool_name, tool_use.get("input", {}))
            async for event in super().stream(tool_use, invocation_state, **kwargs):
                require(len(events) < 1, "Unexpected multiple equipment tool results")
                events.append(event)
        finally:
            if interrupted or self.trial.active:
                self.trial.restore()
                latest = self.session.events[-1] if self.session.events else {}
                self.trial.fault_outcome = {
                    "observed_file_not_found": latest.get("tool") == self.tool_name
                    and latest.get("result") == {"status": "invalid_observation_or_arguments",
                                                "error_type": "FileNotFoundError"},
                    "configuration_unchanged": self.session.configuration == before,
                    "recorded_tool_calls": len(self.session.events),
                }
                self.trial.record("fault_result", **self.trial.fault_outcome)
        for event in events:
            yield event
