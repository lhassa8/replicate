"""Local Strands choice of a supported child-establishment procedure."""

import asyncio
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import time

import httpx
import jsonschema
from strands import Agent
from strands.tools.tools import AgentTool
from strands.types._events import ToolResultEvent

from .capsule import canonical
from .diagnostic_agent import CASE_SECONDS, MAX_OUTPUT_TOKENS, RecordingModel, TraceObserver, specification
from .discovery import public_observation
from .reproduction_journal import ReproductionError
from .reproduction_protocol import PROFILE, RAW_PROFILE, canonical_choice


SYSTEM_PROMPT = (
    "You are an organism deciding whether to establish one independent child on a discovered device. "
    "Inspect the candidate first. Use only the authenticated tool facts. "
    "If the owner enables admission, the device is available, and its platform is supported by the carried release, "
    "choose establish with that exact platform and evidence_id. Otherwise choose defer with a null platform. "
    "Choose once, then finish. The controller transfers the complete carried package after your recorded choice. "
    "You cannot invent a platform, alter permissions, deploy elsewhere, or claim a child is ready."
)
SPECS = (
    specification("inspect_candidate", "Read fresh authenticated admission facts and carried platform support.",
                  {"device_id": {"type": "string", "maxLength": 64}}),
    specification("choose_child_action", "Record establish or defer, citing the inspected evidence. Use null platform for defer.",
                  {"action": {"type": "string", "enum": ["establish", "defer"]},
                   "platform": {"type": ["string", "null"], "enum": ["linux-x86_64", "linux-arm64", None]},
                   "evidence_id": {"type": "string", "maxLength": 64}}),
)
RAW_SYSTEM_PROMPT, RAW_SPECS = SYSTEM_PROMPT, SPECS
SYSTEM_PROMPT = RAW_SYSTEM_PROMPT.replace(
    "Otherwise choose defer with a null platform.",
    'Otherwise choose defer with platform set to the string "none".')
SPECS = deepcopy(RAW_SPECS)
SPECS[1]["description"] = (
    'Record establish or defer, citing the inspected evidence. For defer, use the string "none" as platform.')
SPECS[1]["inputSchema"]["json"]["properties"]["platform"] = {
    "type": "string", "enum": ["linux-x86_64", "linux-arm64", "none"]}


class ReproductionSession:
    def __init__(self, client, book, peer_id, envelope, *, checkpoint=lambda _: None,
                 tool_profile=RAW_PROFILE):
        if tool_profile not in (RAW_PROFILE, PROFILE):
            raise ReproductionError("Unknown reproduction choice protocol")
        self.tool_profile = tool_profile
        self.client, self.book, self.peer_id, self.envelope = client, book, peer_id, envelope
        self.checkpoint, self.events = checkpoint, []
        self.inspection, self.inspected_at, self.choice = None, None, None

    def record(self, tool, arguments, result):
        if len(self.events) >= 8:
            raise RuntimeError("Reproduction tool allowance exhausted")
        self.events.append({"tool": tool, "arguments": deepcopy(arguments), "result": deepcopy(result)})
        self.checkpoint(self.events)
        return result

    async def invoke(self, tool, arguments):
        if len(self.events) >= 8 or self.choice is not None:
            raise ReproductionError("Reproduction decision is already complete or exhausted")
        if tool == "inspect_candidate":
            if arguments["device_id"] != self.peer_id:
                raise ReproductionError("Model selected a different discovered device")
            raw = await self.client.observe(self.book.endpoint(self.peer_id))
            observed = public_observation(raw, self.book.policy, self.peer_id)
            result = {"device_id": self.peer_id, "platform": observed["platform"],
                      "owner_mode": observed["owner_mode"], "available": observed["available"],
                      "revision": observed["revision"], "supported_platforms": sorted(self.envelope["body"]["platforms"]),
                      "release_id": self.envelope["release_id"],
                      "package_bytes": sum(value["bytes"] for value in self.envelope["body"]["files"].values()),
                      "transport_receipt": observed["transport_receipt"]}
            result["evidence_id"] = hashlib.sha256(canonical(result)).hexdigest()
            self.inspection, self.inspected_at = result, time.monotonic()
        elif tool == "choose_child_action":
            try:
                selected = canonical_choice(arguments, self.tool_profile)
            except ValueError as exc:
                raise ReproductionError(str(exc)) from exc
            observed = self.inspection
            if (observed is None or arguments["evidence_id"] != observed["evidence_id"]
                    or not 0 <= time.monotonic() - self.inspected_at < 30):
                raise ReproductionError("Choice lacks fresh inspected evidence")
            if selected["action"] == "establish":
                if (observed["owner_mode"] != "enabled" or not observed["available"]
                        or selected["platform"] != observed["platform"]
                        or selected["platform"] not in observed["supported_platforms"]):
                    raise ReproductionError("Chosen procedure is not admitted or supported")
            self.choice = {**selected, "device_id": self.peer_id,
                           "observed_revision": observed["revision"], "release_id": self.envelope["release_id"]}
            result = {"status": "choice_recorded", "child_established": False}
        else:
            raise ReproductionError("Unknown reproduction tool")
        return self.record(tool, arguments, result)


class ReproductionTool(AgentTool):
    def __init__(self, session, spec):
        super().__init__()
        self.session, self.spec = session, deepcopy(spec)

    @property
    def tool_name(self):
        return self.spec["name"]

    @property
    def tool_type(self):
        return "function"

    @property
    def tool_spec(self):
        return deepcopy(self.spec)

    async def stream(self, tool_use, invocation_state, **kwargs):
        arguments, status = tool_use.get("input", {}), "success"
        try:
            jsonschema.validate(arguments, self.spec["inputSchema"]["json"])
            result = await self.session.invoke(self.tool_name, arguments)
        except (ValueError, OSError, jsonschema.ValidationError, TimeoutError) as exc:
            result = self.session.record(self.tool_name, arguments, {"status": "rejected", "error_type": type(exc).__name__})
        except RuntimeError:
            result, status = {"status": "budget_exhausted"}, "error"
        yield ToolResultEvent({"toolUseId": tool_use["toolUseId"], "status": status,
                              "content": [{"text": json.dumps(result, sort_keys=True, separators=(",", ":"))}]})


async def decide(server, client, book, envelope, store, operation_id, peer_id):
    if version("strands-agents") != "1.57.1":
        raise RuntimeError("Reproduction reasoning is bound to Strands 1.57.1")
    trace = {"system_prompt": SYSTEM_PROMPT, "tool_profile": PROFILE,
             "model_calls": [], "model_responses": [], "http_requests": [],
             "tool_events": [], "final_messages": [], "model_streams": []}
    save = lambda: store.update(operation_id, trace=trace)

    def tools_changed(events):
        trace["tool_events"] = deepcopy(events)
        save()

    session = ReproductionSession(client, book, peer_id, envelope, checkpoint=tools_changed,
                                  tool_profile=PROFILE)

    async def request_hook(request):
        expected = httpx.URL(server.base_url)
        if request.url.host != expected.host or request.url.port != expected.port or request.url.scheme != "http":
            raise RuntimeError("Reproduction model request escaped its owned local server")
        trace["http_requests"].append({"method": request.method, "path": request.url.path,
                                       "body": json.loads(request.content) if request.content else None})
        save()

    model = RecordingModel(trace=trace, checkpoint=save, base_url=server.base_url, timeout=75, model_id=server.alias,
                           params={"temperature": 0, "max_tokens": MAX_OUTPUT_TOKENS})
    await model.client.aclose()
    model.client = httpx.AsyncClient(base_url=server.base_url, timeout=75, trust_env=False,
                                    headers=server.headers, event_hooks={"request": [request_hook]})
    agent = Agent(model=model, tools=[ReproductionTool(session, spec) for spec in SPECS],
                  system_prompt=SYSTEM_PROMPT, hooks=[TraceObserver(trace, save)], callback_handler=None,
                  retry_strategy=None, context_manager=False, background_tasks=False, load_tools_from_directory=False)
    failure, started = None, time.monotonic()
    try:
        await asyncio.wait_for(agent.invoke_async("Consider discovered device " + peer_id + "."), timeout=CASE_SECONDS)
    except asyncio.CancelledError:
        failure = "Application stopped during reproduction reasoning"
        raise
    except Exception as exc:
        failure = type(exc).__name__ + ": " + str(exc)[:512]
    finally:
        await model.client.aclose()
        trace["final_messages"] = deepcopy(agent.messages)
        trace["seconds"] = round(time.monotonic() - started, 3)
        if failure is None and session.choice is None:
            failure = "No complete inspected reproduction choice was submitted"
        store.update(operation_id, trace=trace, submission=session.choice, failure=failure,
                     status="failed" if failure or session.choice is None else "finished")
    return store.get(operation_id)
