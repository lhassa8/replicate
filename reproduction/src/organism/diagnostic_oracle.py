"""Evaluator-only reconstruction from raw observations and a reviewed procedure.

No runtime reader, fixed policy, model, expected answer or training code is
imported. Procedure selection is a private evaluator input, not agent assistance.
This module does not establish that a procedure matches natural-language device
documentation; that association still needs review before qualification.
"""

from decimal import Decimal, localcontext
import hashlib
import json
import math
import re


SCHEMA = "organism-diagnostic-oracle-v1"
IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
REGISTER = re.compile(r"(holding|input):(0|[1-9][0-9]{0,4})\Z")
MISSING = object()


class OracleError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise OracleError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def numeric(value):
    return type(value) in (int, float) and abs(value) <= 1e15 and math.isfinite(value)


def exact(value, keys, message):
    require(isinstance(value, dict) and set(value) == set(keys), message)


def text(value, maximum=160):
    return isinstance(value, str) and 1 <= len(value) <= maximum


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate observation property")
        result[key] = value
    return result


def bounded(value):
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
            require(item is None or type(item) is bool or numeric(item)
                    or (isinstance(item, str) and len(item) <= 512), "Invalid observation scalar")


def path(source, selector):
    require(isinstance(selector, str) and len(selector) <= 160, "Invalid observation selector")
    if source["format"] == "register_map":
        match = REGISTER.fullmatch(selector)
        require(match is not None and int(match[2]) <= 65535, "Invalid register selector")
        return [match[1], match[2]]
    require(selector == "" or selector.startswith("/"), "Invalid JSON pointer")
    require(re.search(r"~(?![01])", selector) is None, "Invalid JSON pointer escape")
    return [] if selector == "" else [part.replace("~1", "/").replace("~0", "~")
                                      for part in selector[1:].split("/")]


def read(source, selector):
    value = source["payload"]
    for part in path(source, selector):
        if isinstance(value, dict):
            value = value.get(part, MISSING)
        elif isinstance(value, list):
            require(re.fullmatch(r"0|[1-9][0-9]*", part) is not None, "Invalid array index")
            value = value[int(part)] if int(part) < len(value) else MISSING
        else:
            return MISSING
    return value


def parse(raw):
    require(isinstance(raw, bytes) and 0 < len(raw) <= 65536, "Invalid observation byte bounds")
    try:
        document = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise OracleError("Invalid observation JSON") from exc
    bounded(document)
    exact(document, ("schema", "device_id", "family", "task", "sources", "sequence"),
          "Unsupported observation document")
    require(document["schema"] == "organism-equipment-interface-v1"
            and text(document["device_id"]) and IDENTITY.fullmatch(document["device_id"])
            and text(document["family"]) and IDENTITY.fullmatch(document["family"]),
            "Unsupported observation identity or schema")
    require(type(document["sequence"]) is int and 0 <= document["sequence"] <= 2**53,
            "Invalid observation sequence")
    task = document["task"]
    exact(task, ("quantity", "canonical_unit", "normal_range", "maximum_age_seconds", "description"),
          "Invalid task description")
    units = {"temperature": "C", "vibration": "mm/s", "inspection": "percent", "machine_log": "count"}
    require(isinstance(task["quantity"], str) and task["quantity"] in units
            and task["canonical_unit"] == units[task["quantity"]], "Invalid canonical quantity or unit")
    bounds = task["normal_range"]
    require(isinstance(bounds, list) and len(bounds) == 2 and all(numeric(item) for item in bounds)
            and -1e6 <= bounds[0] < bounds[1] <= 1e6, "Invalid operating range")
    require(type(task["maximum_age_seconds"]) is int and 1 <= task["maximum_age_seconds"] <= 3600
            and text(task["description"], 256), "Invalid task freshness or description")
    sources = document["sources"]
    require(isinstance(sources, list) and 1 <= len(sources) <= 4, "Invalid source inventory")
    identities = set()
    for source in sources:
        exact(source, ("source_id", "format", "fields", "payload"), "Invalid source")
        identity = source["source_id"]
        require(text(identity) and IDENTITY.fullmatch(identity) and identity not in identities,
                "Invalid or duplicate source identity")
        identities.add(identity)
        require(source["format"] in ("json_document", "register_map"), "Invalid source format")
        require(isinstance(source["payload"], (dict, list)), "Invalid raw source payload")
        fields = source["fields"]
        require(isinstance(fields, list) and 1 <= len(fields) <= 16, "Invalid field inventory")
        selectors = set()
        for field in fields:
            exact(field, ("selector", "kind", "meaning", "unit", "scale"), "Invalid field description")
            selector = field["selector"]
            path(source, selector)
            require(selector not in selectors and text(field["meaning"]), "Duplicate or unnamed field")
            selectors.add(selector)
            require(numeric(field["scale"]) and 0 < field["scale"] <= 1e6, "Invalid field scale")
            if field["kind"] == "number":
                require(field["unit"] in ("C", "F", "K", "mm/s", "in/s", "ratio", "percent", "count"),
                        "Invalid numeric unit")
            else:
                fixed = {"timestamp": "unix_seconds", "validity": "boolean", "records": "records"}
                require(isinstance(field["kind"], str) and field["kind"] in fixed
                        and field["unit"] == fixed[field["kind"]] and field["scale"] == 1,
                        "Invalid field encoding")
    published = {key: document[key] for key in ("device_id", "family", "task")}
    published["sources"] = [{key: value for key, value in source.items() if key != "payload"}
                            for source in sources]
    require(len(encoded(published)) <= 7680, "Capability descriptor exceeds reader bounds")
    return document


def validate_plan(document, plan):
    exact(plan, ("schema", "source_id", "value_selectors", "timestamp_selector", "validity_selector", "operation"),
          "Invalid oracle plan")
    require(plan["schema"] == SCHEMA, "Unsupported oracle plan")
    require(text(plan["source_id"]), "Invalid oracle source")
    sources = [source for source in document["sources"] if source["source_id"] == plan["source_id"]]
    require(len(sources) == 1, "Oracle source is not published")
    source = sources[0]
    fields = {field["selector"]: field for field in source["fields"]}
    selections = plan["value_selectors"]
    require(isinstance(selections, list) and 1 <= len(selections) <= 4
            and all(isinstance(item, str) for item in selections)
            and len(set(selections)) == len(selections)
            and all(item in fields for item in selections), "Oracle selects an unpublished measurement")
    stamp, validity = plan["timestamp_selector"], plan["validity_selector"]
    require(isinstance(stamp, str) and stamp in fields and fields[stamp]["kind"] == "timestamp",
            "Oracle timestamp is not published")
    require(validity is None or (isinstance(validity, str) and validity in fields
                                and fields[validity]["kind"] == "validity"), "Oracle validity is not published")
    operation = plan["operation"]
    require(isinstance(operation, dict) and isinstance(operation.get("kind"), str), "Missing oracle operation")
    kind = operation["kind"]
    keys = {
        "scalar": {"kind"},
        "windowed_scalar": {"kind", "start_key", "end_key", "sequence_key", "unit_key", "scale_key"},
        "counter_pair": {"kind", "role_key", "count_key", "group_key", "accepted_role", "rejected_role"},
        "event_count": {"kind", "id_key", "class_key", "time_key", "target_class", "start_key", "end_key"},
    }
    require(kind in keys, "Unsupported oracle operation")
    exact(operation, keys[kind], "Unexpected oracle operation fields")
    require(all(text(value, 64) for key, value in operation.items() if key != "kind"),
            "Invalid oracle column or category")
    if kind in ("scalar", "windowed_scalar"):
        count = 1 if kind == "scalar" else 2
        require(len(selections) == count and fields[selections[0]]["kind"] == "number",
                "Scalar oracle requires its published numeric field")
        if kind == "windowed_scalar":
            require(fields[selections[1]]["kind"] == "records"
                    and len({operation[key] for key in keys[kind] - {"kind"}}) == 5,
                    "Windowed scalar requires unambiguous published metadata")
        permitted = {"C": ("C", "F", "K"), "mm/s": ("mm/s", "in/s"),
                     "percent": ("ratio", "percent"), "count": ("count",)}
        require(fields[selections[0]]["unit"] in permitted[document["task"]["canonical_unit"]],
                "Oracle unit cannot represent the diagnostic quantity")
    else:
        require(all(fields[item]["kind"] == "records" for item in selections),
                "Record oracle requires published record fields")
        if kind == "counter_pair":
            require(len(selections) == 1 and document["task"]["quantity"] == "inspection",
                    "Counter oracle requires one inspection record field")
            require(len({operation[key] for key in ("role_key", "count_key", "group_key")}) == 3
                    and operation["accepted_role"] != operation["rejected_role"], "Ambiguous counter roles")
        else:
            require(len(selections) == 2 and document["task"]["quantity"] == "machine_log",
                    "Event oracle requires event and window record fields")
            require(len({operation[key] for key in ("id_key", "class_key", "time_key")}) == 3
                    and operation["start_key"] != operation["end_key"], "Ambiguous event columns")
    reader = {"device_id": document["device_id"], "source_id": plan["source_id"],
              "value_selectors": list(selections), "timestamp_selector": stamp, "validity_selector": validity}
    return source, fields, reader


def records(value):
    return isinstance(value, list) and len(value) <= 16 and all(isinstance(item, dict) for item in value)


def counter_rate(rows, operation):
    if not records(rows) or len(rows) != 2:
        return None, "invalid_counter_records"
    role, count, group = (operation[key] for key in ("role_key", "count_key", "group_key"))
    accepted, rejected = operation["accepted_role"], operation["rejected_role"]
    counts, groups = {}, []
    for row in rows:
        label, amount, batch = row.get(role), row.get(count), row.get(group)
        if (not isinstance(label, str) or label not in (accepted, rejected) or label in counts
                or type(amount) is not int or not 0 <= amount <= 2**53 or not text(batch, 128)):
            return None, "invalid_counter_records"
        counts[label] = amount
        groups.append(batch)
    if len(set(groups)) != 1:
        return None, "inconsistent_counter_group"
    total = counts[accepted] + counts[rejected]
    if total == 0:
        return None, "empty_counter_total"
    return Decimal(counts[rejected]) * 100 / Decimal(total), None


def event_total(rows, windows, operation, observed):
    if not records(rows) or not records(windows) or len(windows) != 1:
        return None, "invalid_event_records"
    start, end = (windows[0].get(operation[key]) for key in ("start_key", "end_key"))
    if not numeric(start) or not numeric(end) or not 0 <= start < end or end != observed:
        return None, "inconsistent_event_window"
    identities = {}
    for row in rows:
        identity, category, moment = (row.get(operation[key]) for key in ("id_key", "class_key", "time_key"))
        if not text(identity, 128) or not text(category, 64) or not numeric(moment) or moment < 0:
            return None, "invalid_event_records"
        signature = (category, moment)
        if identity in identities and identities[identity] != signature:
            return None, "conflicting_event_identity"
        identities[identity] = signature
    count = sum(category == operation["target_class"] and start < moment <= end
                for category, moment in identities.values())
    return Decimal(count), None


def coherent_window(windows, operation, observed, sequence, field):
    if not records(windows) or len(windows) != 1:
        return False
    window = windows[0]
    start, end = (window.get(operation[key]) for key in ("start_key", "end_key"))
    serial = window.get(operation["sequence_key"])
    scale = window.get(operation["scale_key"])
    return (numeric(start) and numeric(end) and 0 <= start < end and end == observed
            and type(serial) is int and serial == sequence and numeric(scale)
            and scale == field["scale"] and window.get(operation["unit_key"]) == field["unit"])


def reconstruct(raw, plan, *, now):
    """Compute a diagnostic from raw bytes, never from a recorded submission.

    A schema/procedure defect raises OracleError. Insufficient measurement
    evidence yields unknown. Event intervals are (start, end]; identical event
    IDs must agree on class and timestamp. Counter groups must match exactly.
    """
    document = parse(raw)
    require(numeric(now) and now >= 0, "Invalid evaluator observation time")
    source, fields, reader = validate_plan(document, plan)
    task = document["task"]
    observed = read(source, plan["timestamp_selector"])
    valid = True if plan["validity_selector"] is None else read(source, plan["validity_selector"])
    values = [read(source, selector) for selector in plan["value_selectors"]]
    reason, value = None, None
    if not numeric(observed) or not 0 <= observed <= now:
        reason = "unusable_sample_time"
    elif now - observed > task["maximum_age_seconds"]:
        reason = "stale_sample"
    elif valid is not True:
        reason = "invalid_sample"
    else:
        operation = plan["operation"]
        with localcontext() as context:
            context.prec = 40
            if operation["kind"] in ("scalar", "windowed_scalar"):
                field = fields[plan["value_selectors"][0]]
                if not numeric(values[0]):
                    reason = "invalid_scalar"
                elif operation["kind"] == "windowed_scalar" and not coherent_window(
                        values[1], operation, observed, document["sequence"], field):
                    reason = "inconsistent_scalar_window"
                else:
                    value = Decimal(str(values[0])) * Decimal(str(field["scale"]))
                    unit = field["unit"]
                    if unit == "F":
                        value = (value - 32) * 5 / 9
                    elif unit == "K":
                        value -= Decimal("273.15")
                    elif unit == "in/s":
                        value *= Decimal("25.4")
                    elif unit == "ratio":
                        value *= 100
            elif operation["kind"] == "counter_pair":
                value, reason = counter_rate(values[0], operation)
            else:
                value, reason = event_total(*values, operation, observed)
    lower, upper = (Decimal(str(item)) for item in task["normal_range"])
    status = "unknown" if reason else "low" if value < lower else "high" if value > upper else "normal"
    return {"schema": "organism-diagnostic-oracle-result-v1",
            "observation_sha256": hashlib.sha256(raw).hexdigest(),
            "oracle_plan_sha256": hashlib.sha256(encoded(plan)).hexdigest(),
            "read_unix_seconds": now, "expected_reader": reader, "expected_status": status,
            "expected_value": None if reason else float(value), "expected_unit": task["canonical_unit"],
            "inconclusive_reason": reason, "procedure_semantics_review_required": True,
            "qualification_passed": False}
