"""Explicit deterministic control for the separate deployment comparison.

This module is carried inside each comparison capsule. It never fabricates
model responses. Original release modules and their model-only gates remain
unchanged. The control has its own profile and cannot satisfy those gates.
"""

import asyncio
from copy import deepcopy
import hashlib
import time

from organism.application import AgentApplication
from organism.capsule import canonical
from organism.equipment_baseline import solve
from organism.equipment_interfaces import InterfaceSession
from organism.reproduction import Reproduction
from organism.reproduction_agent import ReproductionSession
from organism.reproduction_protocol import PROFILE


FIXED_PROFILE = "replicate-paper-fixed-policy-v1"


class NoModel:
    """Application lifecycle adapter declaring that no inference server exists."""

    def __init__(self, config, work):
        self.config, self.work = config, work
        self.ready, self.failure, self.process = False, None, None

    async def start(self):
        self.ready = True
        return {"policy_profile": FIXED_PROFILE, "model_started": False, "pid": None}

    async def stop(self):
        self.ready = False
        return {"owned_server_stopped": True, "server_pid": None,
                "server_return_code": None, "model_started": False, "failure": None}


async def diagnose_fixed(server, directory, store, request_id, device_id, **kwargs):
    trace = {"tool_profile": "organism-canonical-temperature-tools-v1",
             "policy_profile": FIXED_PROFILE, "model_calls": [], "model_responses": [],
             "model_streams": [], "http_requests": [], "tool_events": [], "final_messages": []}
    session = InterfaceSession(directory, device_id, canonical_temperature=True)
    failure, began = None, time.monotonic()
    try:
        solve(session, device_id)
    except Exception as exc:
        failure = type(exc).__name__ + ": " + str(exc)
    trace["tool_events"] = deepcopy(session.events)
    trace["seconds"] = time.monotonic() - began
    store.update(request_id, trace=trace, submission=session.submission,
                 failure=failure, status="failed" if failure else "finished")


def verify_fixed(job, *, peer_id, envelope):
    record = job["record"]
    trace = record["trace"]
    if (hashlib.sha256(canonical(record)).hexdigest() != job["digest"]
            or record["status"] != "finished" or record["failure"] is not None
            or record["device_id"] != peer_id or trace["policy_profile"] != FIXED_PROFILE
            or any(trace[k] for k in ("model_calls", "model_responses", "model_streams", "http_requests"))):
        raise ValueError("Invalid deterministic decision identity or nonzero inference")
    events = trace["tool_events"]
    if len(events) != 2 or [e["tool"] for e in events] != ["inspect_candidate", "choose_child_action"]:
        raise ValueError("Deterministic policy must inspect and then choose once")
    observation = events[0]["result"]
    expected_hash = hashlib.sha256(canonical({k: v for k, v in observation.items() if k != "evidence_id"})).hexdigest()
    if (events[0]["arguments"] != {"device_id": peer_id} or observation["device_id"] != peer_id
            or observation["evidence_id"] != expected_hash
            or observation["release_id"] != envelope["release_id"]
            or observation["supported_platforms"] != sorted(envelope["body"]["platforms"])
            or observation["package_bytes"] != sum(e["bytes"] for e in envelope["body"]["files"].values())):
        raise ValueError("Fixed policy inspection differs from the carried release")
    admitted = (observation["owner_mode"] == "enabled" and observation["available"] is True
                and observation["platform"] in observation["supported_platforms"])
    args = {"action": "establish" if admitted else "defer",
            "platform": observation["platform"] if admitted else "none", "evidence_id": expected_hash}
    if events[1]["arguments"] != args or events[1]["result"] != {"status": "choice_recorded", "child_established": False}:
        raise ValueError("Fixed decision differs from the declared rule")
    choice = {**args, "platform": args["platform"] if admitted else None,
              "device_id": peer_id, "observed_revision": observation["revision"],
              "release_id": observation["release_id"]}
    if record["submission"] != choice:
        raise ValueError("Fixed saved choice differs from executed tools")
    return choice


class FixedReproduction(Reproduction):
    async def choose(self, app, intent):
        store, request = self.journal.decisions, intent["operation_id"]
        existing = store.get(request)
        if existing is None:
            existing, _ = store.reserve(request, intent["peer_id"])
        if existing["record"]["status"] == "running":
            if app.busy:
                return
            app.busy = True
            session = ReproductionSession(self.client, self.book, intent["peer_id"],
                                          self.envelope, tool_profile=PROFILE)
            failure, began = None, time.monotonic()
            try:
                fact = await session.invoke("inspect_candidate", {"device_id": intent["peer_id"]})
                admitted = (fact["owner_mode"] == "enabled" and fact["available"] is True
                            and fact["platform"] in fact["supported_platforms"])
                await session.invoke("choose_child_action", {
                    "action": "establish" if admitted else "defer",
                    "platform": fact["platform"] if admitted else "none", "evidence_id": fact["evidence_id"]})
            except Exception as exc:
                failure = type(exc).__name__ + ": " + str(exc)
            finally:
                app.busy = False
            trace = {"tool_profile": PROFILE, "policy_profile": FIXED_PROFILE,
                     "model_calls": [], "model_streams": [], "model_responses": [], "http_requests": [],
                     "tool_events": session.events, "seconds": time.monotonic() - began}
            store.update(request, trace=trace, submission=session.choice, failure=failure,
                         status="failed" if failure else "finished")
            existing = store.get(request)
        try:
            choice = verify_fixed(existing, peer_id=intent["peer_id"], envelope=self.envelope)
            intent.update(phase="authorized" if choice["action"] == "establish" else "deferred",
                          platform=choice["platform"], decision_hash=existing["digest"])
        except (ValueError, KeyError, TypeError) as exc:
            intent.update(phase="decision_failed", failure=str(exc)[:192], decision_hash=existing["digest"])
        self.journal.save(intent)


class FixedApplication(AgentApplication):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inference_factory=NoModel, diagnoser=diagnose_fixed, **kwargs)
        self.receipt["policy_profile"] = FIXED_PROFILE
        self.receipt["local_inference_claimed"] = False

    def health(self):
        value = super().health()
        value.update(policy_profile=FIXED_PROFILE, policy_ready=value["status"] == "serving",
                     inference_ready=False, model_started=False)
        return value

    async def dispatch(self, request):
        if not isinstance(request, dict) or request.get("operation") != "diagnose":
            return await super().dispatch(request)
        if set(request) != {"operation", "request_id", "device_id"}:
            raise ValueError("Unknown fixed-policy diagnostic request fields")
        existing = self.store.get(request["request_id"])
        if existing:
            if (existing["record"]["device_id"] != request["device_id"]
                    or existing["record"].get("task_kind") != "equipment_interface_v1"):
                raise ValueError("Request identifier belongs to another task")
            return self.store.public(existing)
        if self.busy or not self.health()["policy_ready"]:
            return {"status": "busy_or_unavailable"}
        value, fresh = self.store.reserve(request["request_id"], request["device_id"],
                                           task_kind="equipment_interface_v1")
        if fresh:
            self.busy = True
            task = asyncio.create_task(self.execute(request["request_id"], request["device_id"],
                                                   "equipment_interface_v1"))
            self.job_tasks.add(task)
            task.add_done_callback(self.job_done)
        return self.store.public(value)
