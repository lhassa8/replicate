"""Bounded, in-memory device setup fixtures. Never installs on a real host.

The same public schemas can be attached to Strands tools. Admission, ownership,
resource checks, and evidence validation belong to the mechanism, not the model.
Evaluator expectations are deliberately absent from tool observations.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math

from jsonschema import Draft202012Validator


SCHEMA_VERSION = "setup-tools-v3"
PLATFORMS = ("linux-x86_64", "linux-arm64")
MAX_TOOL_CALLS = 16
MIN_MEMORY_MIB = 8192
MIN_STORAGE_MIB = 32768


def _schema(properties: dict) -> dict:
    # Copy each property separately; deepcopy of the containing dictionary
    # preserves aliases such as the reused string schema.
    properties = {name: deepcopy(definition) for name, definition in properties.items()}
    for name, definition in properties.items():
        # Pin the default text inserted by Strands 1.57.1 at registration, so
        # offline training and the actual provider receive identical schemas.
        definition.setdefault("description", f"Property {name}")
    return {
        "type": "object", "properties": properties,
        "required": list(properties), "additionalProperties": False,
    }


_ID = {"type": "string", "minLength": 1, "maxLength": 96}
_TEXT = {"type": "string", "minLength": 1, "maxLength": 160}


def _tool(name: str, description: str, **properties) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name, "description": description,
            "parameters": _schema({"device_id": _ID, **properties}),
        },
    }


TOOLS = [
    _tool("inspect_device", "Read admission, resources, and read-only equipment API capabilities."),
    _tool("prepare_runtime", "Stage a complete capsule. A lost acknowledgement requires reading progress.",
          platform_package={"type": "string", "enum": list(PLATFORMS)}),
    _tool("read_progress", "Read actual installation progress and current configuration."),
    _tool("configure_diagnostic", "Configure the organism's read-only diagnostic adapter.",
          protocol=_TEXT, selector=_TEXT, unit=_TEXT),
    _tool("check_readiness", "Check package, adapter, and fresh sample. Converts units; returns evidence."),
    _tool("refresh_sample", "Request a fresh read-only sample; invalidates earlier readiness evidence."),
    _tool("finish_peer", "Activate a simulated peer with current readiness evidence and diagnostic status.",
          readiness_evidence=_ID,
          diagnostic_status={"type": "string", "enum": ["normal", "low", "high"]}),
    _tool("submit_outcome", "Record ready, rejected, or deferred with observed evidence. Ends the task.",
          outcome={"type": "string", "enum": ["ready", "rejected", "deferred"]},
          evidence_id=_ID),
]
_TOOL_BY_NAME = {item["function"]["name"]: item["function"] for item in TOOLS}
_VALIDATORS = {
    name: Draft202012Validator(spec["parameters"])
    for name, spec in _TOOL_BY_NAME.items()
}


@dataclass(frozen=True)
class DeviceCase:
    case_id: str
    family_id: str
    architecture: str
    protocol: str
    selector: str
    unit: str
    value: float
    expected_normalized: float
    expected_diagnostic: str
    expected_outcome: str
    canonical_unit: str = "C"
    normal_low: float = 10.0
    normal_high: float = 40.0
    memory_mib: int = MIN_MEMORY_MIB
    storage_mib: int = MIN_STORAGE_MIB
    eligible: bool = True
    retired: bool = False
    occupied: bool = False
    population: int = 1
    age_seconds: int = 2
    max_age_seconds: int = 30
    acknowledgement_lost_once: bool = False
    existing_wrong_configuration: bool = False

    def __post_init__(self):
        for value in (self.value, self.expected_normalized, self.normal_low, self.normal_high):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Fixture numbers must be finite")
        if self.unit not in {"C", "F", "K", "mm/s", "in/s", "percent", "ratio", "count"}:
            raise ValueError("Unsupported fixture unit")
        if self.normal_low > self.normal_high or self.age_seconds < 0:
            raise ValueError("Invalid fixture limits")
        if self.expected_outcome not in {"ready", "rejected", "deferred"}:
            raise ValueError("Invalid evaluator outcome")
        if self.expected_diagnostic not in {"normal", "low", "high"}:
            raise ValueError("Invalid evaluator diagnostic")


def normalize(value: float, unit: str) -> float:
    """Conventional arithmetic, shared by both stock and trained agents."""
    converters = {
        "C": lambda x: x, "F": lambda x: (x - 32) * 5 / 9,
        "K": lambda x: x - 273.15, "mm/s": lambda x: x,
        "in/s": lambda x: x * 25.4, "percent": lambda x: x,
        "ratio": lambda x: x * 100, "count": lambda x: x,
    }
    result = converters[unit](value)
    if not math.isfinite(result):
        raise ValueError("Non-finite normalized measurement")
    return round(result, 6)


class SetupSession:
    """All state is local Python data. Public entry point is call()."""

    def __init__(self, case: DeviceCase, max_tool_calls: int = MAX_TOOL_CALLS):
        self.case = case
        self.max_tool_calls = max_tool_calls
        self.events: list[dict] = []
        self.submission: dict | None = None
        self._inspected = False
        self._retired = case.retired
        self._revision = 0
        self._issued: set[str] = set()
        self._capsule: set[str] = set()
        self._selected: str | None = None
        self._configuration: dict | None = None
        self._age = case.age_seconds
        self._lost_ack = False
        self._ready_ticket: str | None = None
        self._active = False
        self._diagnostic: str | None = None
        self._activation_id: str | None = None
        self._stage_count = 0
        if case.existing_wrong_configuration:
            self._capsule = set(PLATFORMS)
            self._selected = f"linux-{case.architecture}"
            self._configuration = {"protocol": "obsolete", "selector": "/old", "unit": "count"}

    def _observation(self, status: str, **fields) -> dict:
        evidence = f"{self.case.case_id}:e{len(self.events) + 1}:r{self._revision}"
        self._issued.add(evidence)
        return {"status": status, "evidence_id": evidence, **fields}

    def _invalidate(self):
        self._revision += 1
        self._ready_ticket = None

    def _refusal(self) -> str | None:
        c = self.case
        if not c.eligible:
            return "ineligible"
        if self._retired:
            return "retired"
        if c.architecture not in {"x86_64", "arm64"}:
            return "unsupported_architecture"
        if c.memory_mib < MIN_MEMORY_MIB or c.storage_mib < MIN_STORAGE_MIB:
            return "insufficient_resources"
        if c.population >= 6:
            return "population_limit"
        if c.occupied:
            return "occupied"
        return None

    def call(self, name: str, arguments: dict) -> dict:
        """Validate before effects and charge every call against the budget."""
        if len(self.events) >= self.max_tool_calls:
            raise RuntimeError("Setup fixture tool budget exhausted")
        args = deepcopy(arguments)
        if self.submission is not None:
            result = {"status": "task_ended"}
        elif name not in _VALIDATORS:
            result = {"status": "unknown_tool"}
        elif not _VALIDATORS[name].is_valid(args):
            result = {"status": "invalid_arguments"}
        elif args["device_id"] != self.case.case_id:
            result = {"status": "unknown_device"}
        elif name != "inspect_device" and not self._inspected:
            result = {"status": "inspection_required"}
        else:
            result = getattr(self, f"_{name}")(**args)
        self.events.append({"tool": name, "arguments": args, "result": deepcopy(result)})
        return deepcopy(result)

    def _inspect_device(self, device_id: str) -> dict:
        self._inspected = True
        c = self.case
        return self._observation(
            "ok", device_id=device_id, architecture=c.architecture,
            memory_mib=c.memory_mib, storage_mib=c.storage_mib,
            admission=self._refusal() or "available",
            available_platform_packages=list(PLATFORMS),
            equipment_api={"protocol": c.protocol, "selector": c.selector, "unit": c.unit},
        )

    def _prepare_runtime(self, device_id: str, platform_package: str) -> dict:
        refusal = self._refusal()
        if refusal:
            return self._observation("admission_refused", reason=refusal)
        if platform_package != f"linux-{self.case.architecture}":
            return self._observation("incompatible_package")
        if self._capsule == set(PLATFORMS) and self._selected == platform_package:
            return self._observation("already_staged")
        self._invalidate()
        self._capsule = set(PLATFORMS)
        self._selected = platform_package
        self._stage_count += 1
        if self.case.acknowledgement_lost_once and not self._lost_ack:
            self._lost_ack = True
            return self._observation("acknowledgement_lost", next_observation="read_progress")
        return self._observation("staged")

    def _read_progress(self, device_id: str) -> dict:
        return self._observation(
            "ok", selected_package=self._selected, retained_packages=sorted(self._capsule),
            configuration=deepcopy(self._configuration), active=self._active,
            admission=self._refusal() or "available",
        )

    def _configure_diagnostic(self, device_id: str, protocol: str, selector: str, unit: str) -> dict:
        refusal = self._refusal()
        if refusal:
            return self._observation("admission_refused", reason=refusal)
        if self._capsule != set(PLATFORMS):
            return self._observation("package_required")
        if self._active:
            return self._observation("already_active")
        self._invalidate()
        self._configuration = {"protocol": protocol, "selector": selector, "unit": unit}
        return self._observation("configured")

    def _check_readiness(self, device_id: str) -> dict:
        refusal = self._refusal()
        if refusal:
            return self._observation("admission_refused", reason=refusal)
        c = self.case
        if self._capsule != set(PLATFORMS) or self._selected != f"linux-{c.architecture}":
            return self._observation("incomplete_capsule")
        expected = {"protocol": c.protocol, "selector": c.selector, "unit": c.unit}
        if self._configuration != expected:
            return self._observation("adapter_mismatch")
        if self._age > c.max_age_seconds:
            return self._observation("stale_sample", age_seconds=self._age,
                                     maximum_age_seconds=c.max_age_seconds)
        result = self._observation(
            "ready", measurement={"value": normalize(c.value, c.unit), "unit": c.canonical_unit,
                                  "age_seconds": self._age},
            normal_range=[c.normal_low, c.normal_high],
        )
        self._ready_ticket = result["evidence_id"]
        return result

    def _refresh_sample(self, device_id: str) -> dict:
        refusal = self._refusal()
        if refusal:
            return self._observation("admission_refused", reason=refusal)
        if self._configuration is None:
            return self._observation("configuration_required")
        self._invalidate()
        self._age = 0
        return self._observation("refreshed")

    def _finish_peer(self, device_id: str, readiness_evidence: str, diagnostic_status: str) -> dict:
        refusal = self._refusal()
        if refusal:
            return self._observation("admission_refused", reason=refusal)
        if self._active:
            return self._observation("already_active")
        if self._capsule != set(PLATFORMS) or readiness_evidence != self._ready_ticket:
            return self._observation("current_readiness_required")
        c = self.case
        reading = normalize(c.value, c.unit)
        actual = "low" if reading < c.normal_low else "high" if reading > c.normal_high else "normal"
        if diagnostic_status != actual:
            return self._observation("diagnostic_mismatch")
        self._active = True
        self._diagnostic = diagnostic_status
        self._invalidate()
        result = self._observation("active", retained_packages=sorted(self._capsule))
        self._activation_id = result["evidence_id"]
        return result

    def _submit_outcome(self, device_id: str, outcome: str, evidence_id: str) -> dict:
        if evidence_id not in self._issued:
            return {"status": "unobserved_evidence"}
        self.submission = {"device_id": device_id, "outcome": outcome, "evidence_id": evidence_id}
        return {"status": "recorded", "independently_verified": False}

    def retire(self):
        """Evaluator fault injection, deliberately absent from the model tools."""
        self._retired = True
        self._active = False
        self._invalidate()

    def evaluator_snapshot(self) -> dict:
        return deepcopy({
            "active": self._active, "retired": self._retired,
            "selected_package": self._selected, "retained_packages": sorted(self._capsule),
            "configuration": self._configuration, "diagnostic": self._diagnostic,
            "age_seconds": self._age, "stage_count": self._stage_count,
            "activation_evidence": self._activation_id, "submission": self.submission,
        })


def evaluate_setup(case: DeviceCase, session: SetupSession) -> dict:
    """Independent checks against fixture truth; a success string is insufficient."""
    state = session.evaluator_snapshot()
    s = state["submission"]
    checks = {
        "same_fixture": asdict(case) == asdict(session.case),
        "outcome_matches": bool(s and s["outcome"] == case.expected_outcome),
        "device_matches": bool(s and s["device_id"] == case.case_id),
    }
    if case.expected_outcome == "ready":
        checks.update({
            "active": state["active"],
            "not_retired": not state["retired"],
            "admitted": case.eligible and not case.occupied and case.population < 6,
            "resource_fit": case.memory_mib >= 8192 and case.storage_mib >= 32768,
            "native_package": state["selected_package"] == f"linux-{case.architecture}",
            "complete_capsule": state["retained_packages"] == sorted(PLATFORMS),
            "correct_adapter": state["configuration"] == {
                "protocol": case.protocol, "selector": case.selector, "unit": case.unit,
            },
            "fresh_observation": state["age_seconds"] <= case.max_age_seconds,
            "correct_diagnostic": state["diagnostic"] == case.expected_diagnostic,
            "activation_evidence": bool(s and s["evidence_id"] == state["activation_evidence"]),
        })
        readiness = [
            e["result"] for e in session.events
            if e["tool"] == "check_readiness" and e["result"].get("status") == "ready"
        ]
        checks["correct_normalization"] = bool(
            readiness and abs(readiness[-1]["measurement"]["value"] - case.expected_normalized) < 1e-5
        )
    else:
        checks.update({"inactive": not state["active"], "no_new_installation": state["stage_count"] == 0})
    return {"passed": all(checks.values()), "checks": checks, "scope": "in-memory development only"}


def contract_hash() -> str:
    return hashlib.sha256(json.dumps(TOOLS, sort_keys=True).encode()).hexdigest()
