"""Four bounded live stock-model checks against the setup simulation."""

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import httpx
from strands import Agent
from strands.hooks import AfterModelCallEvent, BeforeModelCallEvent, HookProvider
from strands.models import LlamaCppModel

from .probe_local import digest_file, run, write_json
from .setup_dataset import development_cases
from .setup_reference import SYSTEM_PROMPT
from .setup_simulator import SetupSession, evaluate_setup
from .setup_tools import build_tools


SELECTION = (
    ("dev-api-b", "new", 1),
    ("dev-api-c", "stale", 1),
    ("dev-api-d", "lost_ack", 1),
    ("dev-api-a", "retired", 1),
)
MODEL_CALL_LIMIT = 12
CASE_SECONDS = 240


def selected_cases():
    inventory = {
        (case.family_id, variation, seed): case
        for split, variation, seed, case in development_cases() if split == "train"
    }
    return [inventory[item] for item in SELECTION]


class Observer(HookProvider):
    def __init__(self):
        self.calls = []
        self.responses = []

    def register_hooks(self, registry, **kwargs):
        registry.add_callback(BeforeModelCallEvent, self.before)
        registry.add_callback(AfterModelCallEvent, self.after)

    def before(self, event):
        if len(self.calls) >= MODEL_CALL_LIMIT:
            event.cancel = "Setup development model-call budget exhausted"
            return
        self.calls.append(deepcopy({
            "messages": event.agent.messages,
            "system_prompt": event.agent.system_prompt,
            "tools": event.agent.tool_registry.get_all_tool_specs(),
        }))

    def after(self, event):
        response = event.stop_response
        self.responses.append({
            "stop_reason": response.stop_reason if response else None,
            "message": deepcopy(response.message) if response else None,
            "exception": str(event.exception) if event.exception else None,
        })


class RecordingModel(LlamaCppModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stream_receipts = []

    async def stream(self, *args, **kwargs):
        events = []
        try:
            async for event in super().stream(*args, **kwargs):
                events.append(deepcopy(event))
                yield event
        finally:
            self.stream_receipts.append(events)


async def probe_cases(base_url: str, alias: str, directory: Path) -> list[dict]:
    write_json(directory / "fixture-selection.json", {
        "selection": SELECTION, "cases": [asdict(c) for c in selected_cases()],
        "scope": "public development cases", "model_call_limit": MODEL_CALL_LIMIT,
        "case_seconds": CASE_SECONDS,
        "source_hashes": {
            name: digest_file(Path(__file__).parent / name)
            for name in ("probe_setup.py", "probe_local.py", "setup_tools.py", "setup_simulator.py",
                         "setup_reference.py", "setup_dataset.py")
        },
    })
    summaries = []
    async with httpx.AsyncClient(base_url=base_url, trust_env=False, timeout=20) as formatter:
        for case in selected_cases():
            session = SetupSession(case)
            observer = Observer()
            requests = []

            async def observe_request(request):
                expected = httpx.URL(base_url)
                if request.url.host != expected.host or request.url.port != expected.port or request.url.scheme != "http":
                    raise RuntimeError("Setup preflight request escaped its owned local endpoint")
                requests.append({
                    "method": request.method, "url": str(request.url),
                    "body": json.loads(request.content),
                })

            model = RecordingModel(
                base_url=base_url, timeout=75, model_id=alias,
                params={"temperature": 0, "max_tokens": 192},
            )
            await model.client.aclose()
            model.client = httpx.AsyncClient(
                base_url=base_url, timeout=75, trust_env=False,
                event_hooks={"request": [observe_request]},
            )
            agent = Agent(
                model=model, tools=build_tools(session), system_prompt=SYSTEM_PROMPT,
                hooks=[observer], callback_handler=None, retry_strategy=None,
                context_manager=False, background_tasks=False, load_tools_from_directory=False,
            )
            started = time.monotonic()
            failure = None
            try:
                await asyncio.wait_for(
                    agent.invoke_async(f"Establish a useful peer on device {case.case_id}."),
                    timeout=CASE_SECONDS,
                )
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
            finally:
                await model.client.aclose()
            evaluation = evaluate_setup(case, session)
            evaluation["passed"] = evaluation["passed"] and failure is None
            # Record the actual server rendering and tokens, not an estimate.
            rendered = []
            for request in requests:
                item = {}
                try:
                    response = await formatter.post("/apply-template", json=request["body"])
                    response.raise_for_status()
                    item["prompt"] = response.json()["prompt"]
                    tokens = await formatter.post("/tokenize", json={
                        "content": item["prompt"], "add_special": False, "parse_special": True,
                    })
                    tokens.raise_for_status()
                    item["tokens"] = tokens.json()["tokens"]
                except Exception as exc:
                    item["failure"] = f"{type(exc).__name__}: {exc}"
                rendered.append(item)
            receipt = {
                "case_id": case.case_id, "case": asdict(case),
                "seconds": round(time.monotonic() - started, 3), "failure": failure,
                "evaluation": evaluation, "requests": requests,
                "model_calls": observer.calls, "model_responses": observer.responses,
                "model_streams": model.stream_receipts,
                "tool_events": session.events, "final_state": session.evaluator_snapshot(),
                "final_messages": agent.messages, "server_rendered_requests": rendered,
            }
            write_json(directory / f"{case.case_id}.json", receipt)
            summary = {k: receipt[k] for k in ("case_id", "seconds", "failure", "evaluation")}
            summary["model_calls"] = len(observer.calls)
            summary["tool_calls"] = len(session.events)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run(
        root, args.output.resolve(), case_runner=probe_cases, case_count=len(SELECTION),
        scope="macOS ARM64 CPU stock model; in-memory setup/recovery development only",
    )
    raise SystemExit(1 if result.get("failure") else 0)
