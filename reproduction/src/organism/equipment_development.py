"""Public development fixtures for actual reader tools, never qualification.

Expected outcomes and reader selections stay outside the installed package and
the equipment observation files. These locally authored fixtures have no human
review or training-admission claim.
"""

from copy import deepcopy


PROFILES = {
    "a": ("development_train", "gateway-json-v1", "temperature", "C", (10, 40), 28, 28, 80, 80),
    "b": ("development_train", "gateway-register-v2", "temperature", "F", (10, 40), 95, 35, 176, 80),
    "c": ("development_train", "condition-json-v1", "vibration", "mm/s", (0, 4.5), 2, 2, 8, 8),
    "d": ("development_train", "condition-register-v3", "vibration", "in/s", (0, 4.5), 0.1, 2.54, 0.3, 7.62),
    "e": ("development_train", "inspection-status-v1", "inspection", "ratio", (0, 5), 0.01, 1, 0.12, 12),
    "f": ("development_train", "event-counter-v2", "machine_log", "count", (0, 2), 0, 0, 9, 9),
    "g": ("development_validation", "thermal-status-v4", "temperature", "K", (10, 40), 312.15, 39, 353.15, 80),
    "h": ("development_validation", "inspection-register-v5", "inspection", "percent", (0, 5), 2, 2, 15, 15),
}
VARIATIONS = ("nominal", "fault", "stale", "invalid", "missing")


def field(selector, kind, meaning, unit, scale=1):
    return {"selector": selector, "kind": kind, "meaning": meaning, "unit": unit, "scale": scale}


def case(profile, variation, now=1700000000.0):
    if profile not in PROFILES or variation not in VARIATIONS:
        raise ValueError("Only declared development profiles and variations are available")
    split, family, quantity, unit, bounds, ordinary, ordinary_truth, faulty, fault_truth = PROFILES[profile]
    fault = variation != "nominal"
    raw_value = faulty if fault else ordinary
    expected = fault_truth if fault else ordinary_truth
    age = 600 if variation == "stale" else 0
    valid = variation != "invalid"
    sample = now - age
    canonical = {"temperature": "C", "vibration": "mm/s", "inspection": "percent", "machine_log": "count"}[quantity]
    descriptions = {"temperature": "Assess the bearing temperature.",
                    "vibration": "Assess the spindle RMS vibration velocity.",
                    "inspection": "Assess the inspection reject percentage for the current sample.",
                    "machine_log": "Assess the error-event count in the completed observation window."}
    meanings = {"temperature": "bearing temperature", "vibration": "spindle RMS vibration velocity",
                "inspection": "inspection rejects in the current sample",
                "machine_log": "error-event count in the completed observation window"}
    extra = []
    if profile in ("b", "d", "h"):
        value_selector = {"b": "holding:4102", "d": "input:3025", "h": "holding:4199"}[profile]
        time_selector, validity_selector = "input:9000", "input:9001"
        bank, address = value_selector.split(":")
        payload = {"holding": {}, "input": {"9000": sample, "9001": valid}}
        payload[bank][address] = raw_value
        if variation == "missing":
            del payload[bank][address]
        format_name = "register_map"
    elif profile == "a":
        value_selector, time_selector, validity_selector = "/bearing/temp", "/sampled_at", "/valid"
        payload = {"bearing": {} if variation == "missing" else {"temp": raw_value},
                   "sampled_at": sample, "valid": valid}
        format_name = "json_document"
    elif profile == "c":
        value_selector, time_selector, validity_selector = "/spindle/rms", "/condition/time", "/condition/valid"
        payload = {"spindle": {} if variation == "missing" else {"rms": raw_value},
                   "condition": {"time": sample, "valid": valid}}
        format_name = "json_document"
    elif profile == "e":
        value_selector, time_selector, validity_selector = "/quality/reject_fraction", "/sample/time", "/sample/valid"
        payload = {"quality": {} if variation == "missing" else {"reject_fraction": raw_value},
                   "sample": {"time": sample, "valid": valid}}
        format_name = "json_document"
    elif profile == "f":
        value_selector, time_selector, validity_selector = "/window/error_count", "/window/ended_at", "/window/complete"
        payload = {"window": {"ended_at": sample, "complete": valid}}
        if variation != "missing":
            payload["window"]["error_count"] = raw_value
        format_name = "json_document"
    else:
        value_selector, time_selector, validity_selector = "/status/probe/absolute", "/status/probe/sampled_at", "/status/probe/valid"
        probe = {"sampled_at": sample, "valid": valid}
        if variation != "missing":
            probe["absolute"] = raw_value
        payload = {"status": {"probe": probe, "reported_at": now}}
        extra = [field("/status/reported_at", "timestamp", "device reporting time, not the sensor sampling time", "unix_seconds")]
        format_name = "json_document"
    identity = f"equipment-{profile}-{variation}"
    fields = [field(value_selector, "number", meanings[quantity], unit),
              field(time_selector, "timestamp", "measurement sampling time", "unix_seconds"),
              field(validity_selector, "validity", "measurement validity", "boolean"), *extra]
    document = {
        "schema": "organism-equipment-interface-v1", "device_id": identity, "family": family,
        "task": {"quantity": quantity, "canonical_unit": canonical, "normal_range": list(bounds),
                 "maximum_age_seconds": 120, "description": descriptions[quantity]},
        "sources": [{"source_id": "telemetry", "format": format_name, "fields": fields, "payload": payload}],
        "sequence": 1,
    }
    unknown = variation in ("stale", "invalid", "missing")
    return {
        "case_id": identity, "profile": "environment-" + profile, "split": split, "variation": variation,
        "document": document, "expected_status": "unknown" if unknown else "high" if fault else "normal",
        "expected_value": None if unknown else expected, "expected_unit": canonical,
        "expected_reader": {"device_id": identity, "source_id": "telemetry",
                            "value_selectors": [value_selector], "timestamp_selector": time_selector,
                            "validity_selector": validity_selector},
        "provenance": {"kind": "locally_authored_development_fixture", "human_review_claimed": False,
                       "training_admitted": False, "qualification": False},
    }


def all_cases(now=1700000000.0):
    return [case(profile, variation, now) for profile in PROFILES for variation in VARIATIONS]


def observation(case_record):
    """Return only device-published data, excluding the private test oracle."""
    return deepcopy(case_record["document"])
