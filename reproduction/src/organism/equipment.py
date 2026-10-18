"""Read-only temperature observations supplied by local equipment adapters.

No evaluator answers, physical-control operations or arbitrary file paths are
available to the model. Hardware-specific collectors are separate adapters.
"""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

from .capsule import canonical


MAX_OBSERVATION_BYTES = 8192
MAX_TOOL_CALLS = 8
DEVICE_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z")


class EquipmentError(ValueError):
    pass


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


class EquipmentDirectory:
    def __init__(self, directory, *, clock=time.time):
        self.directory = Path(directory)
        entry = self.directory.lstat()
        if not stat.S_ISDIR(entry.st_mode) or entry.st_uid not in (0, os.geteuid()) or entry.st_mode & 0o022:
            raise EquipmentError("Equipment observations require a local controlled directory")
        self.clock = clock

    def read(self, device_id):
        if not isinstance(device_id, str) or not DEVICE_ID.fullmatch(device_id):
            raise EquipmentError("Invalid equipment identity")
        directory = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(device_id + ".json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        finally:
            os.close(directory)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid not in (0, os.geteuid())
                    or info.st_mode & 0o022 or info.st_size > MAX_OBSERVATION_BYTES):
                raise EquipmentError("Invalid equipment observation file")
            raw = source.read(MAX_OBSERVATION_BYTES + 1)
        if len(raw) > MAX_OBSERVATION_BYTES:
            raise EquipmentError("Equipment observation exceeds bounds")
        value = json.loads(raw)
        expected = {"schema", "device_id", "sensor_id", "unit", "normal_range_celsius",
                    "maximum_observation_age_seconds", "value", "observed_unix_seconds", "sequence"}
        if not isinstance(value, dict) or set(value) != expected or value["schema"] != "organism-temperature-v1":
            raise EquipmentError("Unsupported equipment observation schema")
        if (value["device_id"] != device_id or not isinstance(value["sensor_id"], str)
                or not DEVICE_ID.fullmatch(value["sensor_id"]) or value["unit"] not in ("C", "F", "K")):
            raise EquipmentError("Equipment identity or unit differs")
        limits = value["normal_range_celsius"]
        if (not isinstance(limits, list) or len(limits) != 2 or not all(number(x) for x in limits)
                or not -273.15 <= limits[0] < limits[1] <= 10000):
            raise EquipmentError("Invalid normal temperature range")
        if (not number(value["value"]) or not -10000 <= value["value"] <= 100000
                or type(value["maximum_observation_age_seconds"]) is not int
                or not 1 <= value["maximum_observation_age_seconds"] <= 3600
                or type(value["sequence"]) is not int or not 0 <= value["sequence"] <= 2**53
                or not number(value["observed_unix_seconds"])):
            raise EquipmentError("Invalid measurement value, age or sequence")
        now = self.clock()
        if not number(now) or not 0 <= value["observed_unix_seconds"] <= now:
            raise EquipmentError("Measurement timestamp is in the future or the clock is invalid")
        value["age_seconds"] = now - value["observed_unix_seconds"]
        value["observation_sha256"] = hashlib.sha256(raw).hexdigest()
        return value


class EquipmentSession:
    def __init__(self, directory, device_id, *, checkpoint=lambda _: None):
        self.directory, self.device_id = directory, device_id
        self.events, self.submission = [], None
        self.capabilities, self.measurement = None, None
        self.checkpoint = checkpoint

    def record(self, tool, arguments, result):
        if len(self.events) >= MAX_TOOL_CALLS:
            raise RuntimeError("Equipment tool-call allowance exhausted")
        self.events.append({"tool": tool, "arguments": deepcopy(arguments), "result": deepcopy(result)})
        self.checkpoint(self.events)
        return deepcopy(result)

    def inspect_device(self, device_id):
        args = {"device_id": device_id}
        if device_id != self.device_id:
            return self.record("inspect_device", args, {"status": "unknown_device"})
        value = self.directory.read(device_id)
        capabilities = {key: value[key] for key in (
            "device_id", "sensor_id", "unit", "normal_range_celsius", "maximum_observation_age_seconds")}
        evidence = hashlib.sha256(canonical(capabilities)).hexdigest()
        self.capabilities = {"data": capabilities, "evidence_id": evidence}
        return self.record("inspect_device", args, {
            "status": "ok", "evidence_id": evidence, "device_id": device_id,
            "sensors": [{"quantity": "temperature", **{key: capabilities[key] for key in capabilities if key != "device_id"}}],
        })

    def read_measurement(self, device_id, sensor_id):
        args = {"device_id": device_id, "sensor_id": sensor_id}
        if self.capabilities is None:
            return self.record("read_measurement", args, {"status": "capabilities_required"})
        if device_id != self.device_id or sensor_id != self.capabilities["data"]["sensor_id"]:
            return self.record("read_measurement", args, {"status": "unknown_sensor"})
        value = self.directory.read(device_id)
        if any(value[key] != expected for key, expected in self.capabilities["data"].items()):
            return self.record("read_measurement", args, {"status": "capabilities_changed"})
        result = {"status": "ok", "device_id": device_id, "sensor_id": sensor_id,
                  "value": value["value"], "unit": value["unit"], "age_seconds": value["age_seconds"],
                  "sequence": value["sequence"], "observed_unix_seconds": value["observed_unix_seconds"],
                  "observation_sha256": value["observation_sha256"]}
        result["evidence_id"] = hashlib.sha256(canonical(result)).hexdigest()
        self.measurement = deepcopy(result)
        return self.record("read_measurement", args, result)

    def submit_assessment(self, device_id, status, temperature_celsius, evidence_ids):
        args = {"device_id": device_id, "status": status, "temperature_celsius": temperature_celsius,
                "evidence_ids": deepcopy(evidence_ids)}
        if self.submission is not None:
            return self.record("submit_assessment", args, {"status": "already_submitted"})
        if (device_id != self.device_id or status not in ("normal", "low", "high", "unknown")
                or (temperature_celsius is not None and not number(temperature_celsius))
                or not isinstance(evidence_ids, list) or any(not isinstance(item, str) for item in evidence_ids)):
            return self.record("submit_assessment", args, {"status": "invalid_submission"})
        required = {item["evidence_id"] for item in (self.capabilities, self.measurement) if item is not None}
        if len(required) != 2 or len(evidence_ids) != 2 or set(evidence_ids) != required:
            return self.record("submit_assessment", args, {"status": "missing_or_invented_evidence"})
        result = self.record("submit_assessment", args, {"status": "recorded", "independently_verified": False})
        self.submission = deepcopy(args)
        return result
