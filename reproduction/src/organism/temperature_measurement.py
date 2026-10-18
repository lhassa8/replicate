"""Normalize selected temperature evidence without choosing a diagnosis."""

from decimal import Decimal
import math


PROFILE = "organism-canonical-temperature-tools-v1"
RAW_PROFILE = "organism-raw-equipment-tools-v1"
SCHEMA = "organism-temperature-measurement-v1"


def normalize(task, reading):
    """Use published units, sampling limits and the reader's actual observation."""
    if task["quantity"] != "temperature" or task["canonical_unit"] != "C":
        raise ValueError("Temperature normalization requires a Celsius task")
    reasons = []
    if reading["validity"] is not True:
        reasons.append("invalid")
    if not reading["timestamp_usable"]:
        reasons.append("unusable_timestamp")
    elif reading["age_seconds"] > task["maximum_age_seconds"]:
        reasons.append("stale")
    values = reading["values"]
    supported = (len(values) == 1 and values[0]["kind"] == "number"
                 and values[0]["unit"] in ("C", "F", "K"))
    if not supported:
        reasons.append("unsupported_measurement")
    elif not values[0]["available"]:
        reasons.append("missing")
    value = None
    if not reasons:
        field = values[0]
        scaled = Decimal(str(field["value"])) * Decimal(str(field["scale"]))
        if field["unit"] == "K":
            scaled -= Decimal("273.15")
        elif field["unit"] == "F":
            scaled = (scaled - 32) * 5 / 9
        value = float(scaled)
        if not math.isfinite(value):
            raise ValueError("Normalized temperature is not finite")
    return {
        "schema": SCHEMA, "usable": not reasons, "reason_codes": reasons,
        "value": value, "unit": "C", "normal_range": list(task["normal_range"]),
        "bounds_inclusive": True,
    }
