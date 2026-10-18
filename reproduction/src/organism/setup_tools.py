"""Strands binding for the exact setup schema and observation serialization."""

from copy import deepcopy
import json

from strands.tools.tools import AgentTool
from strands.types._events import ToolResultEvent

from .setup_simulator import SetupSession, TOOLS


class SetupTool(AgentTool):
    def __init__(self, session: SetupSession, specification: dict):
        super().__init__()
        self.session = session
        self.specification = deepcopy(specification["function"])

    @property
    def tool_name(self):
        return self.specification["name"]

    @property
    def tool_type(self):
        return "function"

    @property
    def tool_spec(self):
        return {
            "name": self.tool_name,
            "description": self.specification["description"],
            "inputSchema": {"json": deepcopy(self.specification["parameters"])},
        }

    async def stream(self, tool_use, invocation_state, **kwargs):
        # Pass original arguments directly through the shared validator. No
        # silent coercion or dropped extra fields between the model and fixture.
        try:
            result = self.session.call(self.tool_name, tool_use.get("input", {}))
            status = "success"
        except RuntimeError as exc:
            result = {"status": "budget_exhausted", "detail": str(exc)}
            status = "error"
        yield ToolResultEvent({
            "toolUseId": tool_use["toolUseId"],
            "status": status,
            "content": [{"text": json.dumps(result, sort_keys=True, separators=(",", ":"))}],
        })


def build_tools(session: SetupSession) -> list[SetupTool]:
    return [SetupTool(session, spec) for spec in TOOLS]
