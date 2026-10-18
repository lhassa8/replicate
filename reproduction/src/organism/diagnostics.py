"""Read-only development fixtures with separately checked completion evidence.

These fixtures are development material, not blind qualification cases. They
operate on in-memory observations and have no device or network access.
"""

from copy import deepcopy
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DiagnosticCase:
    case_id: str
    unit: str
    value: float
    age_seconds: int
    expected_status: str
    expected_celsius: float | None
    sensor_id: str = "bearing-temperature"
    max_age_seconds: int = 30

    def __post_init__(self):
        if self.unit not in {"C", "F", "K"}:
            raise ValueError("Unsupported fixture unit")
        if not math.isfinite(self.value) or self.age_seconds < 0:
            raise ValueError("Invalid fixture observation")


# Expected answers are explicit evaluator data, not computed by the tool layer.
DEVELOPMENT_CASES = (
    DiagnosticCase("dev-normal-c", "C", 28.0, 2, "normal", 28.0),
    DiagnosticCase("dev-high-f", "F", 176.0, 3, "high", 80.0),
    DiagnosticCase("dev-normal-k", "K", 312.15, 1, "normal", 39.0),
    DiagnosticCase("dev-stale-c", "C", 90.0, 600, "unknown", None),
)


class DiagnosticSession:
    """Fixed observation tools; the caller cannot change acceptance criteria."""

    def __init__(self, case: DiagnosticCase, max_tool_calls: int = 8):
        self._case = case
        self._max_tool_calls = max_tool_calls
        self._tool_calls = 0
        self._inspection_id = f"{case.case_id}:capabilities"
        self._measurement_id = f"{case.case_id}:measurement"
        self._issued: set[str] = set()
        self.events: list[dict] = []
        self.submission: dict | None = None

    @property
    def device_id(self) -> str:
        return self._case.case_id

    def _record(self, tool: str, arguments: dict, result: dict) -> dict:
        self._tool_calls += 1
        if self._tool_calls > self._max_tool_calls:
            raise RuntimeError("Development fixture tool budget exhausted")
        self.events.append({
            "tool": tool, "arguments": deepcopy(arguments), "result": deepcopy(result),
        })
        return deepcopy(result)

    def inspect_device(self, device_id: str) -> dict:
        if device_id != self.device_id:
            return self._record("inspect_device", {"device_id": device_id}, {
                "status": "unknown_device",
            })
        self._issued.add(self._inspection_id)
        return self._record("inspect_device", {"device_id": device_id}, {
            "status": "ok",
            "evidence_id": self._inspection_id,
            "device_id": device_id,
            "sensors": [{
                "sensor_id": self._case.sensor_id,
                "quantity": "temperature",
                "unit": self._case.unit,
                "normal_range_celsius": [10.0, 40.0],
                "maximum_observation_age_seconds": self._case.max_age_seconds,
            }],
        })

    def read_measurement(self, device_id: str, sensor_id: str) -> dict:
        args = {"device_id": device_id, "sensor_id": sensor_id}
        if self._inspection_id not in self._issued:
            return self._record("read_measurement", args, {
                "status": "capabilities_required",
            })
        if device_id != self.device_id or sensor_id != self._case.sensor_id:
            return self._record("read_measurement", args, {
                "status": "unknown_sensor",
            })
        self._issued.add(self._measurement_id)
        return self._record("read_measurement", args, {
            "status": "ok",
            "evidence_id": self._measurement_id,
            "device_id": device_id,
            "sensor_id": sensor_id,
            "value": self._case.value,
            "unit": self._case.unit,
            "age_seconds": self._case.age_seconds,
        })

    def submit_assessment(
        self, device_id: str, status: str,
        temperature_celsius: float | None, evidence_ids: list[str],
    ) -> dict:
        args = {
            "device_id": device_id, "status": status,
            "temperature_celsius": temperature_celsius,
            "evidence_ids": deepcopy(evidence_ids),
        }
        if self.submission is not None:
            return self._record("submit_assessment", args, {
                "status": "already_submitted",
            })
        if (
            device_id != self.device_id
            or status not in {"normal", "low", "high", "unknown"}
            or not isinstance(evidence_ids, list)
            or any(not isinstance(x, str) for x in evidence_ids)
            or (
                temperature_celsius is not None
                and (
                    type(temperature_celsius) not in (int, float)
                    or not math.isfinite(temperature_celsius)
                )
            )
        ):
            return self._record("submit_assessment", args, {
                "status": "invalid_submission",
            })
        required = {self._inspection_id, self._measurement_id}
        if not required.issubset(self._issued) or set(evidence_ids) != required:
            return self._record("submit_assessment", args, {
                "status": "missing_or_invented_evidence",
            })
        result = self._record("submit_assessment", args, {
            "status": "recorded", "independently_verified": False,
        })
        self.submission = deepcopy(args)
        return result


def evaluate_submission(case: DiagnosticCase, session: DiagnosticSession) -> dict:
    """Compare the recorded result with evaluator-owned expectations."""
    submission = session.submission
    if submission is None:
        return {"passed": False, "reason": "No evidence-backed submission"}
    if case.case_id != session.device_id:
        return {"passed": False, "reason": "Evaluator/session identity mismatch"}
    actual = submission["temperature_celsius"]
    if case.expected_celsius is None:
        temperature_matches = actual is None
    else:
        temperature_matches = (
            type(actual) in (int, float)
            and math.isfinite(actual)
            and abs(actual - case.expected_celsius) <= 0.05
        )
    status_matches = submission["status"] == case.expected_status
    return {
        "passed": status_matches and temperature_matches,
        "status_matches": status_matches,
        "temperature_matches": temperature_matches,
        "expected_status": case.expected_status,
        "expected_celsius": case.expected_celsius,
        "scope": "development diagnostic task only",
    }
