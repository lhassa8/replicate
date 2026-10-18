"""One-shot synthetic descriptor holdout, created after freezing the agent.

No runtime policy or reader is imported. Decimal arithmetic supplies the oracle.
The private case records are not provided to an agent; only ``document`` is.
"""

from decimal import Decimal
import hashlib
import json


CONDITIONS = (
    "midpoint", "fractional", "below", "above", "lower_boundary",
    "upper_boundary", "stale", "invalid", "missing", "future",
)
FAMILIES = (
    ("compact-celsius", "C", "1", ("-10", "55")),
    ("nested-kelvin", "K", "1", ("0", "60")),
    ("register-fahrenheit", "F", "1", ("10", "40")),
    ("escaped-array-scaled", "C", "0.01", ("-20", "80")),
)


def field(selector, kind, meaning, unit, scale=1):
    return dict(selector=selector, kind=kind, meaning=meaning, unit=unit, scale=scale)


def build_cases():
    cases = []
    for family, unit, scale, bounds in FAMILIES:
        low, high = map(Decimal, bounds)
        for condition in CONDITIONS:
            value = {
                "midpoint": (low + high) / 2, "fractional": low + Decimal("3.125"),
                "below": low - Decimal("0.125"), "above": high + Decimal("0.125"),
                "lower_boundary": low, "upper_boundary": high,
            }.get(condition, low + Decimal("7.25"))
            converted = value + Decimal("273.15") if unit == "K" else (
                value * Decimal(9) / Decimal(5) + 32 if unit == "F" else value)
            raw = float(converted / Decimal(scale))
            sampled = -600 if condition == "stale" else 600 if condition == "future" else 0
            valid = condition != "invalid"
            token = hashlib.sha256(f"replicate-paper-holdout-v1/{family}/{condition}".encode()).hexdigest()[:24]
            identity = "equipment-" + token
            format_name = "json_document"
            if family == "compact-celsius":
                selectors = ("/measurement_c", "/sample_clock", "/sample_accepted")
                payload = {"sample_clock": sampled, "sample_accepted": valid}
                value_parent, value_key = payload, "measurement_c"
            elif family == "nested-kelvin":
                selectors = (
                    "/machine/shaft/thermometry/kelvin",
                    "/machine/shaft/thermometry/capture_epoch",
                    "/machine/shaft/thermometry/quality_ok",
                )
                value_parent = {"capture_epoch": sampled, "quality_ok": valid}
                value_key = "kelvin"
                payload = {"machine": {"shaft": {"thermometry": value_parent}}}
            elif family == "register-fahrenheit":
                selectors = ("holding:5173", "input:6101", "input:6102")
                payload = {"holding": {}, "input": {"6101": sampled, "6102": valid}}
                value_parent, value_key = payload["holding"], "5173"
                format_name = "register_map"
            else:
                selectors = (
                    "/channels/0/sensor~1temperature",
                    "/channels/0/capture~0epoch",
                    "/channels/0/quality~1ok",
                )
                value_parent = {"capture~epoch": sampled, "quality/ok": valid}
                value_key = "sensor/temperature"
                payload = {"channels": [value_parent]}
            if condition != "missing":
                value_parent[value_key] = raw
            value_selector, time_selector, valid_selector = selectors
            fields = [
                field(value_selector, "number", "bearing temperature", unit, float(scale)),
                field(time_selector, "timestamp", "measurement sampling time", "unix_seconds"),
                field(valid_selector, "validity", "measurement validity", "boolean"),
            ]
            # Field order is unrelated to condition and differs across layouts.
            if family in ("nested-kelvin", "escaped-array-scaled"):
                fields = fields[1:] + fields[:1]
            source_id = "channel-" + hashlib.sha256(family.encode()).hexdigest()[:10]
            document = {
                "schema": "organism-equipment-interface-v1", "device_id": identity,
                "family": "holdout-" + family,
                "task": {
                    "quantity": "temperature", "canonical_unit": "C",
                    "normal_range": [float(low), float(high)], "maximum_age_seconds": 120,
                    "description": "Assess the bearing temperature.",
                },
                "sources": [{"source_id": source_id, "format": format_name,
                             "fields": fields, "payload": payload}], "sequence": 1,
            }
            unknown = condition in ("stale", "invalid", "missing", "future")
            cases.append({
                "case_id": identity, "profile": family, "split": "paper_synthetic_holdout",
                "variation": condition, "document": document,
                "expected_status": "unknown" if unknown else (
                    "low" if value < low else "high" if value > high else "normal"),
                "expected_value": None if unknown else float(value), "expected_unit": "C",
                "expected_reader": {
                    "device_id": identity, "source_id": source_id,
                    "value_selectors": [value_selector], "timestamp_selector": time_selector,
                    "validity_selector": valid_selector,
                },
                "provenance": {
                    "kind": "author_designed_post_freeze_synthetic_holdout",
                    "training_admitted": False, "external_benchmark": False,
                    "qualification": False, "oracle": "decimal arithmetic and declared conditions",
                },
            })
    return cases


if __name__ == "__main__":
    print(json.dumps({"schema": "replicate-paper-holdout-v1", "cases": build_cases()}, indent=2))
