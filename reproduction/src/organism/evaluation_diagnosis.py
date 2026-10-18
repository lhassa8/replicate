"""Attach evaluator observation tools to the existing production diagnostic code.

The function's code, prompts, model settings and session types remain those of
equipment_diagnostic_agent.diagnose. A per-trial globals dictionary replaces
only its EquipmentTool factory; no application module global is patched.
"""

import asyncio
from copy import deepcopy
import hashlib
from pathlib import Path
from types import FunctionType

from . import equipment_diagnostic_agent
from .diagnostic_agent import EquipmentTool
from .evaluation_observation import ObservationTool, require


SOURCE_NAMES = (
    "equipment_diagnostic_agent.py", "diagnostic_agent.py", "equipment_interfaces.py",
    "evaluation_observation.py", "evaluation_diagnosis.py",
)


def source_identities():
    source = Path(__file__).parent
    return {name: {"bytes": (source / name).stat().st_size,
                   "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest()}
            for name in SOURCE_NAMES}


def bind_function(trial):
    original = equipment_diagnostic_agent.diagnose
    require(original.__globals__["EquipmentTool"] is EquipmentTool
            and Path(original.__code__.co_filename).resolve() == Path(equipment_diagnostic_agent.__file__).resolve(),
            "Unexpected production diagnostic implementation")
    selected = []

    def factory(session, specification):
        require(specification in equipment_diagnostic_agent.EXPERIENCE_SPECS,
                "Evaluator received an undeclared diagnostic tool")
        selected.append(deepcopy(specification))
        return ObservationTool(session, specification, trial)

    namespace = dict(original.__globals__)
    namespace["EquipmentTool"] = factory
    bound = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    bound.__kwdefaults__ = deepcopy(original.__kwdefaults__)
    bound.__annotations__ = deepcopy(original.__annotations__)
    require(bound.__code__ is original.__code__
            and set(bound.__globals__) == set(original.__globals__)
            and all(bound.__globals__[name] is value for name, value in original.__globals__.items()
                    if name != "EquipmentTool"), "Evaluator changed diagnostic code or other bindings")
    return bound, selected


class EvaluationDiagnoser:
    """One admitted application job with the observation's original deadline."""

    def __init__(self):
        self.trial, self.function, self.selected = None, None, None
        self.used, self.receipt = False, None

    def bind(self, trial):
        require(self.trial is None and not self.used and not trial.closed,
                "Evaluator diagnostic binding cannot be reused")
        trial.unchanged()
        self.function, self.selected = bind_function(trial)
        self.trial = trial
        self.sources = source_identities()
        trial.record("diagnostic_bound", sources=self.sources,
                     intervention="only the tool factory in a private function-globals dictionary",
                     production_function_code_reused=True, production_module_globals_patched=False)

    async def __call__(self, server, observations, store, request_id, device_id, *, experience_store=None):
        require(self.trial is not None and not self.used, "One bound evaluator trial is required")
        trial = self.trial
        require(device_id == trial.case_id and observations.directory.resolve() == trial.path.parent.resolve(),
                "Diagnostic request belongs to another observation")
        initial = store.get(request_id)
        require(initial is not None and initial["record"]["device_id"] == device_id
                and initial["record"].get("task_kind") == "equipment_interface_v1"
                and initial["record"]["status"] == "running" and initial["record"]["trace"] == {},
                "Evaluator requires a fresh admitted diagnostic job")
        self.used = True
        expected_specs = (equipment_diagnostic_agent.SPECS if experience_store is None
                          else equipment_diagnostic_agent.EXPERIENCE_SPECS)
        receipt = {
            "request_id": request_id, "case_id": device_id, "failure": None,
            "outer_deadline_exhausted": False, "production_function_code_reused": True,
            "production_module_globals_patched": False, "sources": self.sources,
            "qualification_passed": False,
        }
        self.receipt = receipt

        def fail_job(reason):
            current = store.get(request_id)
            if current["record"]["status"] == "running":
                store.update(request_id, trace=current["record"]["trace"], status="failed",
                             submission=current["record"]["submission"], failure=reason)

        try:
            require(self.function.__code__ is equipment_diagnostic_agent.diagnose.__code__
                    and source_identities() == self.sources, "Bound diagnostic implementation changed")
            timing = trial.timing()
            require(timing["remaining_seconds"] > 0, "Original diagnostic deadline exhausted")
            receipt["started"] = timing
            trial.record("diagnostic_started", request_id=request_id, **timing)
            await asyncio.wait_for(
                self.function(server, observations, store, request_id, device_id, experience_store=experience_store),
                timeout=timing["remaining_seconds"])
            require(self.selected == list(expected_specs), "Diagnostic tool specifications changed")
        except asyncio.TimeoutError:
            receipt.update(failure="Original diagnostic deadline exhausted", outer_deadline_exhausted=True)
            fail_job(receipt["failure"])
        except asyncio.CancelledError:
            receipt["failure"] = "Evaluator diagnostic was cancelled"
            fail_job(receipt["failure"])
            raise
        except Exception as exc:
            receipt["failure"] = type(exc).__name__ + ": " + str(exc)
            fail_job(receipt["failure"])
        finally:
            receipt["sources_unchanged"] = source_identities() == self.sources
            receipt["tool_specifications"] = deepcopy(self.selected)
            receipt["job_digest"] = store.get(request_id)["digest"]
            receipt["job_status"] = store.get(request_id)["record"]["status"]
            receipt["observation_restored"] = not trial.active
            try:
                trial.unchanged()
                receipt["observation_unchanged"] = True
                receipt["finished"] = trial.stop()
            except Exception as exc:
                receipt["observation_or_clock_failure"] = type(exc).__name__ + ": " + str(exc)
            trial.record("diagnostic_finished", **receipt)
        return store.get(request_id)
