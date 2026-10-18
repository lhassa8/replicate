"""Model-requested reuse of endorsed readers, with fresh capability evidence."""

from copy import deepcopy
import time

from .capsule import canonical
from .equipment_interfaces import InterfaceSession, MAX_TOOL_CALLS, require


TOOL_NAME = "prepare_reader_from_experience"
TRACE_SCHEMA = "organism-diagnostic-inheritance-v1"


class ExperiencedInterfaceSession(InterfaceSession):
    def __init__(self, directory, device_id, experience_store, *, checkpoint=lambda _: None):
        super().__init__(directory, device_id, checkpoint=checkpoint)
        self.experience_store = experience_store
        self.inheritance = {"schema": TRACE_SCHEMA, "attempts": []}

    def operation_cost(self, index, tool):
        if tool == TOOL_NAME:
            attempts = [item for item in self.inheritance["attempts"] if item["tool_event_index"] == index]
            return max([1] + [len(item["operations"]) for item in attempts])
        return 1

    def record(self, tool, arguments, result):
        cost = sum(self.operation_cost(index, event["tool"]) for index, event in enumerate(self.events))
        if cost + self.operation_cost(len(self.events), tool) > MAX_TOOL_CALLS:
            raise RuntimeError("Equipment tool-call allowance exhausted")
        return super().record(tool, arguments, result)

    def prepare_reader_from_experience(self, device_id):
        require(device_id == self.device_id, "Unknown equipment identity")
        require(len(self.inheritance["attempts"]) < MAX_TOOL_CALLS, "Experience lookup allowance exhausted")
        cost = sum(self.operation_cost(index, event["tool"]) for index, event in enumerate(self.events))
        if cost >= MAX_TOOL_CALLS:
            raise RuntimeError("Equipment tool-call allowance exhausted")
        attempt = {"tool_event_index": len(self.events), "status": "started",
                   "started_unix_seconds": self.directory.clock(), "elapsed_seconds": None,
                   "validation_unix_seconds": None, "application_unix_seconds": None,
                   "available_note_ids": [], "matching_note_ids": [], "selected_note_id": None, "operations": []}
        self.inheritance["attempts"].append(attempt)
        started = time.monotonic()
        self.checkpoint(self.events)

        def captured(events):
            attempt["operations"] = deepcopy(events)
            self.checkpoint(self.events)

        # Stage reader changes in a separate session. The live session is
        # changed only after the complete outer tool result passes its bounds.
        prepared = InterfaceSession(self.directory, device_id, checkpoint=captured)
        try:
            capabilities = prepared.inspect_equipment(device_id)
            moment = self.experience_store.now()
            require(abs(moment - self.directory.clock()) <= 2, "Experience and observation clocks disagree")
            attempt["validation_unix_seconds"] = moment
            inventory = self.experience_store.inventory()
            attempt["available_note_ids"] = inventory
            matches = {}
            for note_id in inventory:
                record = self.experience_store.get(note_id)
                if record.applicable(capabilities["descriptor"], now=moment):
                    matches[note_id] = record
            attempt["matching_note_ids"] = sorted(matches)
            readers = {canonical(record.body["reader"]) for record in matches.values()}
            selected = None
            if not matches:
                status = "no_applicable_experience"
            elif len(readers) != 1:
                status = "conflicting_experience"
            else:
                selected = min(matches)
                record = self.experience_store.get(selected)
                now = self.directory.clock()
                require(type(now) in (int, float) and abs(self.experience_store.now() - now) <= 2,
                        "Experience and observation clocks disagree")
                require(record.applicable(capabilities["descriptor"], now=int(now)),
                        "Experience expired or no longer applies")
                # Count inspection and configuration independently even though
                # the model requests them through one combined tool.
                if cost + 2 > MAX_TOOL_CALLS:
                    raise RuntimeError("Equipment tool-call allowance exhausted")
                prepared.configure_reader(device_id=device_id, **record.body["reader"])
                attempt["application_unix_seconds"] = int(now)
                status = "prepared_from_experience"
            attempt["selected_note_id"], attempt["status"] = selected, status
            result = {"status": status, "note_id": selected, "capabilities": prepared.capabilities,
                      "reader": prepared.configuration, "equipment_modified": False, "diagnosis_supplied": False}
            attempt["elapsed_seconds"] = round(time.monotonic() - started, 6)
            result = self.record(TOOL_NAME, {"device_id": device_id}, result)
            self.capabilities = deepcopy(prepared.capabilities)
            self.configuration = deepcopy(prepared.configuration)
            self.measurement = None
            return result
        except BaseException as exc:
            attempt["status"], attempt["error_type"] = "failed", type(exc).__name__
            attempt["elapsed_seconds"] = round(time.monotonic() - started, 6)
            self.checkpoint(self.events)
            raise
