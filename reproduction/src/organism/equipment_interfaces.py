"""Bounded read-only equipment interfaces, independent of diagnostic answers.

A local collector publishes a descriptor and raw JSON/register observations.
The agent selects and configures its own reader. Configuration affects only
this diagnostic session; it cannot write equipment or access arbitrary paths.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

from .capsule import canonical
from .equipment import DEVICE_ID, EquipmentError, number


SCHEMA = "organism-equipment-interface-v1"
MAX_FILE_BYTES = 65536
MAX_RESULT_BYTES = 8192
MAX_TOOL_CALLS = 8
QUANTITIES = ("temperature", "vibration", "inspection", "machine_log")
UNITS = ("C", "F", "K", "mm/s", "in/s", "ratio", "percent", "count", "records")
REGISTER = re.compile(r"(holding|input):([0-9]{1,5})\Z")
VALUE_INTEGRITY_REVISION = "canonical-value-validation-v1"


class TemperatureValueError(EquipmentError):
    """A proposed value disagrees with the issued canonical measurement."""

    def __init__(self, measurement, evidence_id):
        super().__init__("Submitted value differs from the canonical reading")
        self.feedback = {
            "schema": "organism-measurement-value-feedback-v1",
            "reading_evidence_id": evidence_id,
            "canonical_value": measurement["value"],
            "canonical_unit": measurement["unit"],
            "usable": measurement["usable"],
            "instruction": "Copy canonical_value exactly without rounding and submit again. Choose the diagnosis status from the reading. No diagnosis was recorded.",
        }


def require(condition, message):
    if not condition:
        raise EquipmentError(message)


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate observation property")
        result[key] = value
    return result


def bounded_tree(value):
    pending, visited = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        visited += 1
        require(depth <= 10 and visited <= 2048, "Observation structure exceeds bounds")
        if isinstance(item, dict):
            require(all(isinstance(key, str) and len(key) <= 160 for key in item), "Invalid observation key")
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            require(len(item) <= 64, "Observation array exceeds bounds")
            pending.extend((child, depth + 1) for child in item)
        else:
            require(item is None or type(item) is bool or (number(item) and abs(item) <= 1e15)
                    or (isinstance(item, str) and len(item) <= 512), "Invalid observation scalar")


def pointer_tokens(selector):
    require(isinstance(selector, str) and len(selector) <= 160
            and (selector == "" or selector.startswith("/"))
            and re.search(r"~(?![01])", selector) is None, "Invalid JSON pointer")
    return [] if selector == "" else [part.replace("~1", "/").replace("~0", "~")
                                      for part in selector[1:].split("/")]


def select(source, selector):
    """Resolve a declared selector; never invent a missing observation."""
    value = source["payload"]
    if source["format"] == "register_map":
        match = REGISTER.fullmatch(selector)
        if match is None or not 0 <= int(match[2]) <= 65535:
            raise EquipmentError("Invalid register selector")
        return value[match[1]][str(int(match[2]))]
    for token in pointer_tokens(selector):
        if isinstance(value, list):
            require(re.fullmatch(r"0|[1-9][0-9]*", token) is not None, "Invalid array index")
            value = value[int(token)]
        elif isinstance(value, dict):
            value = value[token]
        else:
            raise EquipmentError("Selector traverses a scalar")
    return value


def descriptor(document):
    return {key: deepcopy(document[key]) for key in ("device_id", "family", "task")} | {
        "sources": [{key: deepcopy(value) for key, value in source.items() if key != "payload"}
                    for source in document["sources"]]}


def validate_document(document, device_id):
    require(isinstance(document, dict) and set(document) == {
        "schema", "device_id", "family", "task", "sources", "sequence"} and document["schema"] == SCHEMA,
        "Unsupported equipment interface")
    require(document["device_id"] == device_id and isinstance(document["family"], str)
            and DEVICE_ID.fullmatch(document["family"]), "Equipment identity differs")
    require(type(document["sequence"]) is int and 0 <= document["sequence"] <= 2**53, "Invalid sequence")
    task = document["task"]
    require(isinstance(task, dict) and set(task) == {
        "quantity", "canonical_unit", "normal_range", "maximum_age_seconds", "description"},
        "Invalid diagnostic task descriptor")
    expected_units = {"temperature": "C", "vibration": "mm/s", "inspection": "percent", "machine_log": "count"}
    require(task["quantity"] in QUANTITIES and task["canonical_unit"] == expected_units[task["quantity"]],
            "Unsupported diagnostic quantity")
    bounds = task["normal_range"]
    require(isinstance(bounds, list) and len(bounds) == 2 and all(number(item) for item in bounds)
            and -1e6 <= bounds[0] < bounds[1] <= 1e6
            and type(task["maximum_age_seconds"]) is int and 1 <= task["maximum_age_seconds"] <= 3600,
            "Invalid diagnostic range or freshness limit")
    require(isinstance(task["description"], str) and 1 <= len(task["description"]) <= 256,
            "Invalid task description")
    sources = document["sources"]
    require(isinstance(sources, list) and 1 <= len(sources) <= 4, "Invalid source inventory")
    source_ids = set()
    for source in sources:
        require(isinstance(source, dict) and set(source) == {"source_id", "format", "fields", "payload"},
                "Invalid equipment source")
        identity = source["source_id"]
        require(isinstance(identity, str) and DEVICE_ID.fullmatch(identity) and identity not in source_ids,
                "Invalid or duplicate source identity")
        source_ids.add(identity)
        require(source["format"] in ("json_document", "register_map"), "Unsupported reader format")
        fields = source["fields"]
        require(isinstance(fields, list) and 1 <= len(fields) <= 16, "Invalid field inventory")
        selectors = set()
        for field in fields:
            require(isinstance(field, dict) and set(field) == {"selector", "kind", "meaning", "unit", "scale"},
                    "Invalid equipment field")
            selector = field["selector"]
            require(isinstance(selector, str) and selector not in selectors, "Duplicate field selector")
            selectors.add(selector)
            if source["format"] == "json_document":
                pointer_tokens(selector)
            else:
                match = REGISTER.fullmatch(selector)
                require(match is not None and str(int(match[2])) == match[2] and int(match[2]) <= 65535,
                        "Invalid register address")
            require(field["kind"] in ("number", "timestamp", "validity", "records")
                    and isinstance(field["meaning"], str) and 1 <= len(field["meaning"]) <= 160,
                    "Invalid field kind or meaning")
            require(number(field["scale"]) and 0 < field["scale"] <= 1e6, "Invalid field scale")
            if field["kind"] == "number":
                require(field["unit"] in UNITS[:-1], "Unsupported measurement unit")
            elif field["kind"] == "timestamp":
                require(field["unit"] == "unix_seconds" and field["scale"] == 1, "Invalid timestamp encoding")
            elif field["kind"] == "validity":
                require(field["unit"] == "boolean" and field["scale"] == 1, "Invalid validity encoding")
            else:
                require(field["unit"] == "records" and field["scale"] == 1, "Invalid record-list encoding")
        require(isinstance(source["payload"], (dict, list)), "Invalid raw source payload")
    require(len(canonical(descriptor(document))) <= MAX_RESULT_BYTES - 512, "Capability descriptor exceeds bounds")


class InterfaceDirectory:
    def __init__(self, directory, *, clock=time.time, observer=None):
        self.directory, self.clock, self.observer = Path(directory), clock, observer
        info = self.directory.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid in (0, os.geteuid()) and not info.st_mode & 0o022,
                "Equipment interfaces require a controlled directory")

    def read(self, device_id):
        require(isinstance(device_id, str) and DEVICE_ID.fullmatch(device_id), "Invalid equipment identity")
        directory = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(device_id + ".json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        finally:
            os.close(directory)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid in (0, os.geteuid())
                    and not info.st_mode & 0o022 and info.st_size <= MAX_FILE_BYTES, "Invalid equipment source file")
            raw = stream.read(MAX_FILE_BYTES + 1)
        require(len(raw) <= MAX_FILE_BYTES, "Equipment source exceeds bounds")
        try:
            document = json.loads(raw, object_pairs_hook=strict_object)
            bounded_tree(document)
        except (ValueError, RecursionError) as exc:
            raise EquipmentError("Invalid bounded observation JSON") from exc
        validate_document(document, device_id)
        raw_hash = hashlib.sha256(raw).hexdigest()
        if self.observer is not None:
            self.observer(device_id, raw, raw_hash)
        return document, raw_hash


class InterfaceSession:
    def __init__(self, directory, device_id, *, checkpoint=lambda _: None, canonical_temperature=False):
        from .temperature_measurement import PROFILE, RAW_PROFILE
        self.directory, self.device_id, self.checkpoint = directory, device_id, checkpoint
        self.canonical_temperature = canonical_temperature
        self.tool_profile = PROFILE if canonical_temperature else RAW_PROFILE
        self.events, self.capabilities, self.configuration, self.measurement = [], None, None, None
        self.submission = None

    def record(self, tool, arguments, result):
        if len(self.events) >= MAX_TOOL_CALLS:
            raise RuntimeError("Equipment tool-call allowance exhausted")
        require(len(canonical(result)) <= MAX_RESULT_BYTES, "Equipment tool result exceeds bounds")
        self.events.append({"tool": tool, "arguments": deepcopy(arguments), "result": deepcopy(result)})
        self.checkpoint(self.events)
        return deepcopy(result)

    def current(self, device_id):
        require(device_id == self.device_id, "Unknown equipment identity")
        document, raw_hash = self.directory.read(device_id)
        current = descriptor(document)
        require(self.capabilities is not None and current == self.capabilities["descriptor"],
                "Inspect again after equipment capabilities change")
        return document, raw_hash

    def inspect_equipment(self, device_id):
        require(device_id == self.device_id, "Unknown equipment identity")
        document, _ = self.directory.read(device_id)
        data = descriptor(document)
        result = {"status": "ok", "descriptor": data, "evidence_id": hashlib.sha256(canonical(data)).hexdigest()}
        output = self.record("inspect_equipment", {"device_id": device_id}, result)
        self.capabilities, self.configuration, self.measurement = deepcopy(result), None, None
        return output

    def configure_reader(self, device_id, source_id, value_selectors, timestamp_selector, validity_selector):
        args = {"device_id": device_id, "source_id": source_id, "value_selectors": deepcopy(value_selectors),
                "timestamp_selector": timestamp_selector, "validity_selector": validity_selector}
        document, _ = self.current(device_id)
        sources = {source["source_id"]: source for source in document["sources"]}
        require(source_id in sources, "Unknown read-only source")
        fields = {field["selector"]: field for field in sources[source_id]["fields"]}
        require(isinstance(value_selectors, list) and 1 <= len(value_selectors) <= 4
                and all(isinstance(item, str) for item in value_selectors)
                and len(set(value_selectors)) == len(value_selectors)
                and all(item in fields and fields[item]["kind"] in ("number", "records") for item in value_selectors),
                "Invalid measurement selectors")
        require(timestamp_selector in fields and fields[timestamp_selector]["kind"] == "timestamp",
                "Invalid timestamp selector")
        require(validity_selector is None or (validity_selector in fields and fields[validity_selector]["kind"] == "validity"),
                "Invalid validity selector")
        bound = {**args, "capability_evidence_id": self.capabilities["evidence_id"]}
        result = {"status": "configured", "configuration": bound,
                  "evidence_id": hashlib.sha256(canonical(bound)).hexdigest(), "equipment_modified": False}
        output = self.record("configure_reader", args, result)
        self.configuration, self.measurement = deepcopy(result), None
        return output

    def read_equipment(self, device_id):
        document, raw_hash = self.current(device_id)
        require(self.configuration is not None, "Configure the read-only reader first")
        config = self.configuration["configuration"]
        source = next(item for item in document["sources"] if item["source_id"] == config["source_id"])
        fields = {field["selector"]: field for field in source["fields"]}

        def read_field(selector):
            try:
                value = select(source, selector)
                kind = fields[selector]["kind"]
                valid = (number(value) if kind in ("number", "timestamp") else type(value) is bool
                         if kind == "validity" else isinstance(value, list) and len(value) <= 16
                         and all(isinstance(item, dict) for item in value))
                return deepcopy(value) if valid else None
            except (KeyError, IndexError, TypeError, EquipmentError):
                return None

        values = []
        for selector in config["value_selectors"]:
            value = read_field(selector)
            values.append({**deepcopy(fields[selector]), "value": value, "available": value is not None})
        observed = read_field(config["timestamp_selector"])
        now = self.directory.clock()
        require(number(now) and now >= 0, "Invalid equipment reader clock")
        time_valid = observed is not None and 0 <= observed <= now
        validity = True if config["validity_selector"] is None else read_field(config["validity_selector"])
        result = {
            "status": "observed", "device_id": device_id, "source_id": source["source_id"],
            "values": values, "observed_unix_seconds": observed, "read_unix_seconds": now,
            "age_seconds": now - observed if time_valid else None, "timestamp_usable": time_valid,
            "validity": validity, "sequence": document["sequence"], "observation_sha256": raw_hash,
            "configuration_evidence_id": self.configuration["evidence_id"],
        }
        if self.canonical_temperature and document["task"]["quantity"] == "temperature":
            from .temperature_measurement import normalize
            result["temperature_measurement"] = normalize(document["task"], result)
        result["evidence_id"] = hashlib.sha256(canonical(result)).hexdigest()
        output = self.record("read_equipment", {"device_id": device_id}, result)
        self.measurement = deepcopy(result)
        return output

    def submit_diagnosis(self, device_id, status, value, unit, evidence_id):
        args = {"device_id": device_id, "status": status, "value": value, "unit": unit,
                "evidence_id": evidence_id}
        require(self.submission is None, "A diagnosis has already been submitted")
        require(device_id == self.device_id and status in ("normal", "low", "high", "unknown")
                and (value is None or number(value)) and unit in UNITS, "Invalid diagnostic submission")
        require(isinstance(evidence_id, str), "Invalid diagnostic evidence")
        require(self.capabilities is not None and self.configuration is not None and self.measurement is not None
                and evidence_id == self.measurement["evidence_id"],
                "Missing or invented diagnostic evidence")
        if self.canonical_temperature and "temperature_measurement" in self.measurement:
            measured = self.measurement["temperature_measurement"]
            if value != measured["value"]:
                raise TemperatureValueError(measured, evidence_id)
        result = self.record("submit_diagnosis", args, {"status": "recorded", "independently_verified": False})
        self.submission = deepcopy(args)
        return result
