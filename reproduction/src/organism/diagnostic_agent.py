"""Strands equipment reasoning with bounded calls and durable, literal traces."""

import asyncio
from copy import deepcopy
from importlib.metadata import version
import json
import time

import httpx
import jsonschema
from strands import Agent
from strands.hooks import AfterModelCallEvent, BeforeModelCallEvent, HookProvider
from strands.models import LlamaCppModel
from strands.tools.tools import AgentTool
from strands.types._events import ToolResultEvent

from .equipment import EquipmentError, EquipmentSession
from .equipment_interfaces import TemperatureValueError


MODEL_CALL_LIMIT = 4
CASE_SECONDS = 150
MAX_OUTPUT_TOKENS = 192
SYSTEM_PROMPT = 'You are a read-only equipment diagnostic agent. Inspect the device, read its temperature sensor, then submit one evidence-backed assessment. Compare Celsius temperature against the stated normal range. Convert units when necessary. If the reading is older than the allowed age, report unknown with null temperature. Use only returned evidence IDs. After submission, end with a short acknowledgement. Do not invent observations. Copy the requested device_id exactly into every tool call. Never shorten it, repeat any part of it, or construct a new identifier. Copy sensor_id directly from inspected capabilities. If a read is rejected, correct its arguments and read again. Do not submit until inspect_device and read_measurement have both returned status ok. Error strings such as unknown_sensor are not evidence IDs. Copy only evidence_id fields from successful results into the assessment. For a fresh reading, compare the Celsius value with normal_range_celsius: below the first number is low, above the second number is high, and otherwise normal. The lower and upper boundaries themselves are normal.'


def specification(name, description, properties):
    return {"name": name, "description": description, "inputSchema": {"json": {
        "type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}}}


SPECS = (
    specification("inspect_device", "Read a device's sensor capabilities and acceptable operating range.",
                  {"device_id": {"type": "string", "maxLength": 64}}),
    specification("read_measurement", "Read the specified sensor after inspecting its device capabilities.",
                  {"device_id": {"type": "string", "maxLength": 64}, "sensor_id": {"type": "string", "maxLength": 64}}),
    specification("submit_assessment", "Record one assessment with both observed capability and measurement IDs. Use null temperature when stale.",
                  {"device_id": {"type": "string", "maxLength": 64}, "status": {"type": "string", "enum": ["normal", "low", "high", "unknown"]},
                   "temperature_celsius": {"type": ["number", "null"]},
                   "evidence_ids": {"type": "array", "items": {"type": "string", "maxLength": 64}, "minItems": 2, "maxItems": 2}}),
)


def bound_specs(device_id):
    """Expose only the device already selected by this diagnostic request."""
    specs = deepcopy(SPECS)
    for spec in specs:
        spec["inputSchema"]["json"]["properties"]["device_id"]["enum"] = [device_id]
    return specs


class EquipmentTool(AgentTool):
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
        arguments = tool_use.get("input", {})
        status = "success"
        try:
            jsonschema.validate(arguments, self.spec["inputSchema"]["json"])
            result = getattr(self.session, self.tool_name)(**arguments)
        except TemperatureValueError as exc:
            result = self.session.record(self.tool_name, arguments, {
                "status": "invalid_observation_or_arguments",
                "error_type": "TemperatureValueError",
                "measurement_value_feedback": exc.feedback,
            })
        except (jsonschema.ValidationError, EquipmentError, OSError, ValueError) as exc:
            result = self.session.record(self.tool_name, arguments, {"status": "invalid_observation_or_arguments",
                                                                    "error_type": type(exc).__name__})
        except RuntimeError:
            result, status = {"status": "budget_exhausted"}, "error"
        yield ToolResultEvent({"toolUseId": tool_use["toolUseId"], "status": status,
                              "content": [{"text": json.dumps(result, sort_keys=True, separators=(",", ":"))}]})


class TraceObserver(HookProvider):
    def __init__(self, trace, save):
        self.trace, self.save = trace, save

    def register_hooks(self, registry, **kwargs):
        registry.add_callback(BeforeModelCallEvent, self.before)
        registry.add_callback(AfterModelCallEvent, self.after)

    def before(self, event):
        if len(self.trace["model_calls"]) >= MODEL_CALL_LIMIT:
            event.cancel = "Diagnostic model-call allowance exhausted"
            return
        self.trace["model_calls"].append(deepcopy({
            "messages": event.agent.messages, "system_prompt": event.agent.system_prompt,
            "tools": event.agent.tool_registry.get_all_tool_specs()}))
        self.save()

    def after(self, event):
        response = event.stop_response
        self.trace["model_responses"].append({
            "stop_reason": response.stop_reason if response else None,
            "message": deepcopy(response.message) if response else None,
            "exception": str(event.exception)[:512] if event.exception else None})
        self.save()


class RecordingModel(LlamaCppModel):
    """Retain streamed tool-argument fragments before SDK JSON parsing."""
    def __init__(self, *, trace, checkpoint, **kwargs):
        super().__init__(**kwargs)
        self.trace, self.checkpoint = trace, checkpoint

    async def stream(self, *args, **kwargs):
        events = []
        self.trace["model_streams"].append(events)
        try:
            async for event in super().stream(*args, **kwargs):
                events.append(deepcopy(event))
                if len(events) % 32 == 0:
                    self.checkpoint()
                yield event
        finally:
            self.checkpoint()


async def assess(server, observations, store, request_id, device_id):
    if version("strands-agents") != "1.57.1":
        raise RuntimeError("Application is bound to Strands 1.57.1")
    trace = {"system_prompt": SYSTEM_PROMPT, "model_calls": [], "model_responses": [],
             "http_requests": [], "tool_events": [], "final_messages": [], "model_streams": []}
    save = lambda: store.update(request_id, trace=trace)

    def tools_changed(events):
        trace["tool_events"] = deepcopy(events)
        save()

    session = EquipmentSession(observations, device_id, checkpoint=tools_changed)
    observer = TraceObserver(trace, save)

    async def request_hook(request):
        expected = httpx.URL(server.base_url)
        if request.url.host != expected.host or request.url.port != expected.port or request.url.scheme != "http":
            raise RuntimeError("Model request escaped its owned local server")
        # Authorization headers are deliberately excluded from the trace.
        trace["http_requests"].append({"method": request.method, "path": request.url.path,
                                       "body": json.loads(request.content) if request.content else None})
        save()

    model = RecordingModel(trace=trace, checkpoint=save, base_url=server.base_url, timeout=75, model_id=server.alias,
                           params={"temperature": 0, "max_tokens": MAX_OUTPUT_TOKENS})
    await model.client.aclose()
    model.client = httpx.AsyncClient(base_url=server.base_url, timeout=75, trust_env=False,
                                    headers=server.headers, event_hooks={"request": [request_hook]})
    agent = Agent(model=model, tools=[EquipmentTool(session, spec) for spec in bound_specs(device_id)],
                  system_prompt=SYSTEM_PROMPT, hooks=[observer], callback_handler=None, retry_strategy=None,
                  context_manager=False, background_tasks=False, load_tools_from_directory=False)
    failure, started = None, time.monotonic()
    try:
        await asyncio.wait_for(agent.invoke_async("Assess the device " + device_id + "."), timeout=CASE_SECONDS)
    except asyncio.CancelledError:
        failure = "Application stopped during the diagnostic"
        raise
    except Exception as exc:
        failure = type(exc).__name__ + ": " + str(exc)[:512]
    finally:
        await model.client.aclose()
        trace["final_messages"] = deepcopy(agent.messages)
        trace["seconds"] = round(time.monotonic() - started, 3)
        if failure is None and session.submission is None:
            failure = "No complete evidence-backed assessment was submitted"
        store.update(request_id, trace=trace, submission=session.submission, failure=failure,
                     status="failed" if failure or session.submission is None else "finished")
    return store.get(request_id)
