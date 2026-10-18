"""Evaluator-only draft inputs. Never import into training or deployed code.

This creates cases from the existing allocation without calling a model or
baseline. It does not seal the experiment or qualify any release. A saved seed
is reused for reconstruction; failed authoring checks must not trigger sampling
replacement cases.
"""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import secrets

from .diagnostic_oracle import encoded, reconstruct, require
from .equipment_development import case as familiar_case
from .evaluation_design import trial_slots, validate


REFERENCE_TIME = 1700000000
GENERATOR = "organism-qualification-inputs-v1"


def fingerprint(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def entry(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def field(selector, kind, meaning, unit, scale=1):
    return {"selector": selector, "kind": kind, "meaning": meaning, "unit": unit, "scale": scale}


def value_number(value):
    number = Decimal(str(value))
    return int(number) if number == int(number) else float(number)


def condition(slot):
    repetition = int(slot["slot_id"].split("/")[-1]) - 1
    if slot["suite"] == "Q4_Q5":
        return ("normal", "fault", "encoding", "stale", "conflict", "configuration_interruption")[repetition]
    if slot["suite"] == "Q7":
        return ("normal", "encoding", "stale", "configuration_interruption", "fault")[repetition]
    if slot["suite"] == "protected":
        return ("normal", "fault", "stale")[repetition]
    group, cycle = repetition % 4, repetition // 4
    return (("normal", "fault", "normal", "fault")[cycle] if group == 0 else "encoding" if group == 1 else
            ("stale", "invalid", "missing", "conflict")[cycle] if group == 2 else "configuration_interruption")


def scalar_encoding(canonical, unit, scale):
    number = Decimal(str(canonical))
    if unit == "F":
        number = number * 9 / 5 + 32
    elif unit == "K":
        number += Decimal("273.15")
    elif unit == "in/s":
        number /= Decimal("25.4")
    elif unit == "ratio":
        number /= 100
    return value_number(number / Decimal(str(scale)))


def document_base(slot, identity, description, bounds, sequence):
    return {
        "schema": "organism-equipment-interface-v1", "device_id": identity, "family": slot["family"],
        "task": {"quantity": slot["domain"],
                 "canonical_unit": {"temperature": "C", "vibration": "mm/s",
                                    "inspection": "percent", "machine_log": "count"}[slot["domain"]],
                 "normal_range": bounds, "maximum_age_seconds": 300, "description": description},
        "sources": [], "sequence": sequence,
    }


def plan(source, values, timestamp, validity, operation):
    return {"schema": "organism-diagnostic-oracle-v1", "source_id": source,
            "value_selectors": values, "timestamp_selector": timestamp,
            "validity_selector": validity, "operation": operation}


def thermal(slot, identity, rng, variant):
    channel = "probe-" + str(rng.randrange(100, 999))
    selected = rng.randrange(3)
    normal = variant not in ("fault", "encoding")
    truth = Decimal(rng.randrange(2011, 3499) if normal else rng.randrange(5501, 7599)) / 100
    unit = ("C", "F", "K")[rng.randrange(3)]
    scale = (1, 0.1, 0.01)[rng.randrange(3)]
    document = document_base(slot, identity, f"Assess temperature for channel {channel}; use that channel's sample time and validity.",
                             [10, 40], rng.randrange(1, 100000))
    rows, fields = [], []
    for index in range(3):
        chosen = index == selected
        name = channel if chosen else "aux-" + str(index)
        value = scalar_encoding(truth if chosen else 18 + index, unit, scale)
        row = {"channel": name, "reading": value, "sample": REFERENCE_TIME, "usable": True}
        if chosen and variant == "stale":
            row["sample"] -= 601
        elif chosen and variant in ("invalid", "conflict"):
            row["usable"] = False
        elif chosen and variant == "missing":
            row.pop("reading")
        rows.append(row)
        stem = f"/channels/{index}"
        fields.extend([field(stem + "/reading", "number", "temperature for channel " + name, unit, scale),
                       field(stem + "/sample", "timestamp", "sensor sample time for channel " + name, "unix_seconds"),
                       field(stem + "/usable", "validity", "sensor validity for channel " + name, "boolean")])
    rng.shuffle(fields)
    document["sources"] = [{"source_id": "channels", "format": "json_document",
                            "fields": fields, "payload": {"channels": rows}}]
    stem = f"/channels/{selected}"
    return document, plan("channels", [stem + "/reading"], stem + "/sample", stem + "/usable",
                          {"kind": "scalar"}), float(truth)


def vibration(slot, identity, rng, variant):
    address = rng.randrange(100, 900)
    sequence = rng.randrange(1, 100000)
    unit = ("mm/s", "in/s")[rng.randrange(2)]
    scale = (0.001, 0.01, 0.1)[rng.randrange(3)]
    ordinary = variant not in ("fault", "encoding")
    # Inch values are chosen in exact hundredths; forward conversion is exact.
    source_velocity = Decimal(rng.randrange(4, 15) if ordinary else rng.randrange(24, 39)) / 100
    truth = source_velocity * Decimal("25.4")
    raw = value_number((source_velocity if unit == "in/s" else truth) / Decimal(str(scale)))
    sampled = REFERENCE_TIME - (601 if variant == "stale" else 0)
    metadata = {"began": sampled - rng.randrange(2, 30), "ended": sampled,
                "sequence": sequence, "unit": unit, "scale": scale}
    if variant == "conflict":
        metadata["sequence"] += 1
    payload = {"holding": {str(address): raw}, "input": {
        str(address + 1): sampled, str(address + 2): variant != "invalid", str(address + 3): [metadata]}}
    if variant == "missing":
        payload["holding"].pop(str(address))
    value, stamp, valid, window = (f"holding:{address}", f"input:{address + 1}",
                                   f"input:{address + 2}", f"input:{address + 3}")
    fields = [field(value, "number", "RMS velocity for the completed window", unit, scale),
              field(stamp, "timestamp", "completed window end time", "unix_seconds"),
              field(valid, "validity", "measurement sample validity", "boolean"),
              field(window, "records", "one window: began, ended, sequence, unit, scale; must agree with sample and descriptor", "records")]
    rng.shuffle(fields)
    document = document_base(slot, identity,
        "Assess RMS velocity. Window began must precede ended; ended equals sample time. Window sequence, unit and scale must match the observation and value descriptor; otherwise unknown.",
        [0, 4.5], sequence)
    document["sources"] = [{"source_id": "window-bank", "format": "register_map",
                            "fields": fields, "payload": payload}]
    oracle = plan("window-bank", [value, window], stamp, valid, {
        "kind": "windowed_scalar", "start_key": "began", "end_key": "ended", "sequence_key": "sequence",
        "unit_key": "unit", "scale_key": "scale"})
    return document, oracle, float(truth)


def inspection(slot, identity, rng, variant):
    percentage = rng.randrange(1, 5) if variant not in ("fault", "encoding") else rng.randrange(8, 17)
    multiple = rng.randrange(7, 83)
    group = "batch-" + str(rng.randrange(10000, 99999))
    rows = [{"outcome": "accepted", "count": (100 - percentage) * multiple, "batch": group},
            {"outcome": "rejected", "count": percentage * multiple, "batch": group}]
    if variant == "conflict":
        rows[1]["batch"] += "-other"
    rng.shuffle(rows)
    payload = {"batch_counts": rows, "sampled": REFERENCE_TIME - (601 if variant == "stale" else 0),
               "valid": variant != "invalid"}
    if variant == "missing":
        payload.pop("batch_counts")
    fields = [field("/batch_counts", "records", "two outcome/count/batch records; accepted and rejected integer counts", "records"),
              field("/sampled", "timestamp", "batch observation time", "unix_seconds"),
              field("/valid", "validity", "batch observation validity", "boolean")]
    rng.shuffle(fields)
    document = document_base(slot, identity,
        "Assess reject percentage: 100*rejected/(accepted+rejected). Both counts must be nonnegative integers for the same batch, with a positive total; otherwise unknown.",
        [0, 5], rng.randrange(1, 100000))
    document["sources"] = [{"source_id": "batch-results", "format": "json_document",
                            "fields": fields, "payload": payload}]
    oracle = plan("batch-results", ["/batch_counts"], "/sampled", "/valid", {
        "kind": "counter_pair", "role_key": "outcome", "count_key": "count", "group_key": "batch",
        "accepted_role": "accepted", "rejected_role": "rejected"})
    return document, oracle, percentage


def machine_events(slot, identity, rng, variant):
    count = rng.randrange(1, 3) if variant not in ("fault", "encoding") else rng.randrange(4, 7)
    sampled = REFERENCE_TIME - (601 if variant == "stale" else 0)
    duration = rng.randrange(30, 61)
    start = sampled - duration
    tag = str(rng.randrange(10000, 99999))
    rows = [{"event_id": f"{tag}-{index}", "category": "error", "observed": sampled - index - 1}
            for index in range(count)]
    rows.extend([deepcopy(rows[0]),
                 {"event_id": tag + "-left", "category": "error", "observed": start},
                 {"event_id": tag + "-future", "category": "error", "observed": sampled + 1},
                 {"event_id": tag + "-info", "category": "info", "observed": sampled}])
    if variant == "conflict":
        rows[count]["category"] = "warning"
    rng.shuffle(rows)
    payload = {"records": rows, "windows": [{"start": start, "end": sampled}],
               "sample": sampled, "complete": variant != "invalid"}
    if variant == "missing":
        payload.pop("records")
    fields = [field("/records", "records", "event_id, category, observed in unix seconds", "records"),
              field("/windows", "records", "one window with start and end in unix seconds", "records"),
              field("/sample", "timestamp", "observation window end time", "unix_seconds"),
              field("/complete", "validity", "window collection complete", "boolean")]
    rng.shuffle(fields)
    document = document_base(slot, identity,
        "Count unique error event_ids in (start,end]. Duplicate IDs must agree on category and observed time, even outside the window; conflicts are unknown. The sample time must equal window end.",
        [0, 2], rng.randrange(1, 100000))
    document["sources"] = [{"source_id": "event-window", "format": "json_document",
                            "fields": fields, "payload": payload}]
    oracle = plan("event-window", ["/records", "/windows"], "/sample", "/complete", {
        "kind": "event_count", "id_key": "event_id", "class_key": "category", "time_key": "observed",
        "target_class": "error", "start_key": "start", "end_key": "end"})
    return document, oracle, count


def protected(slot, identity, rng, variant):
    profile = slot["profile"][-1]
    item = familiar_case(profile, "nominal", REFERENCE_TIME)
    document = deepcopy(item["document"])
    document["device_id"] = identity
    document["sequence"] = rng.randrange(1, 100000)
    document["task"]["maximum_age_seconds"] = 300
    reader = item["expected_reader"]
    source = document["sources"][0]
    unit = next(field["unit"] for field in source["fields"] if field["selector"] == reader["value_selectors"][0])
    ranges = {"temperature": ((2111, 3399), (6111, 7499)),
              "vibration": ((111, 299), (611, 899)), "inspection": ((111, 399), (811, 1299)),
              "machine_log": ((0, 2), (7, 19))}
    low, high = ranges[slot["domain"]][variant == "fault"]
    truth = Decimal(rng.randrange(low, high + 1)) / (1 if slot["domain"] == "machine_log" else 100)

    def assign(selector, value):
        if source["format"] == "register_map":
            bank, key = selector.split(":")
            source["payload"][bank][key] = value
        else:
            keys = selector[1:].split("/")
            current = source["payload"]
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value

    assign(reader["value_selectors"][0], scalar_encoding(truth, unit, 1))
    assign(reader["timestamp_selector"], REFERENCE_TIME - (601 if variant == "stale" else 0))
    return document, plan(reader["source_id"], reader["value_selectors"], reader["timestamp_selector"],
                          reader["validity_selector"], {"kind": "scalar"}), float(truth)


def time_paths(document):
    """Enumerate generated timestamp locations once, never infer them at runtime."""
    found, pending = [], [(document, [])]
    while pending:
        value, location = pending.pop()
        if isinstance(value, dict):
            pending.extend((child, location + [key]) for key, child in value.items())
        elif isinstance(value, list):
            pending.extend((child, location + [index]) for index, child in enumerate(value))
        elif type(value) in (int, float) and REFERENCE_TIME - 3600 <= value <= REFERENCE_TIME + 3600:
            found.append({"path": location, "offset_seconds": value - REFERENCE_TIME})
    return sorted(found, key=lambda entry: encoded(entry["path"]))


def materialize(case, epoch):
    require(type(epoch) is int and 1000000000 <= epoch <= 4000000000, "Invalid trial clock origin")
    document = deepcopy(case["document"])
    for binding in case["time_binding"]["fields"]:
        current = document
        for part in binding["path"][:-1]:
            current = current[part]
        last = binding["path"][-1]
        require(current[last] == REFERENCE_TIME + binding["offset_seconds"], "Time binding no longer matches fixture")
        current[last] = epoch + binding["offset_seconds"]
    return document


def build_case(slot, seed, suite_index):
    case_seed = hashlib.sha256(bytes.fromhex(seed) + b"\0" + slot["slot_id"].encode()).hexdigest()
    rng = random.Random(int(case_seed, 16))
    identity = "qual-" + case_seed[:32]
    variant = condition(slot)
    make = protected if slot["suite"] == "protected" else {
        "environment-i": thermal, "environment-j": vibration,
        "environment-k": inspection, "environment-l": machine_events}[slot["profile"]]
    document, oracle, value = make(slot, identity, rng, variant)
    unknown = variant in ("stale", "invalid", "missing", "conflict")
    expected = {"status": "unknown" if unknown else "normal" if document["task"]["normal_range"][0] <= value <=
                document["task"]["normal_range"][1] else "high" if value > document["task"]["normal_range"][1] else "low",
                "value": None if unknown else value, "unit": document["task"]["canonical_unit"]}
    arms = slot["arms"]
    rotation = (suite_index // 2) % len(arms)
    fault = ({"kind": "one_configuration_read_unavailable", "operation": "configure_reader", "attempt": 1,
              "effect": "observation file absent for this call, restored before the next call",
              "charged_to_original_case_budget": True, "native_injection_verified": False}
             if variant == "configuration_interruption" else None)
    result = {
        "schema": "organism-qualification-input-case-v1", "status": "draft_inputs_not_sealed",
        "slot": deepcopy(slot), "case_id": identity, "case_seed": case_seed, "condition": variant,
        "document": document, "oracle_plan": oracle, "expected": expected,
        "arm_order": arms[rotation:] + arms[:rotation], "fault": fault,
        "starting_state": {"experience": [], "reader_configuration": None, "prior_tasks": [],
                           "cross_case_sharing": False, "fault_delivered": False,
                           "release_binding": None, "baseline_binding": None},
        "time_binding": {"kind": "per_arm_trial_origin_preserves_frozen_offsets",
                         "reference_unix_seconds": REFERENCE_TIME, "fields": time_paths(document),
                         "native_clock_binding_verified": False},
        "source_episode_sha256": None, "inherited_notes_sha256": None,
        "authorship": {"author": "project_agent", "independent_human_review_claimed": False,
                       "training_use": False, "candidate_model_exposed": False,
                       "baseline_exposed": False, "inaccessible_custody_claimed": False},
    }
    require(result["time_binding"]["fields"], "Case lacks explicit sample clock bindings")
    reconstructed = reconstruct(encoded(document), oracle, now=REFERENCE_TIME)
    require(reconstructed["expected_status"] == expected["status"] and reconstructed["expected_unit"] == expected["unit"]
            and (reconstructed["expected_value"] is None if expected["value"] is None else
                 reconstructed["expected_value"] is not None
                 and abs(reconstructed["expected_value"] - expected["value"]) <= 0.0000001),
            "Constructed observation differs from its independently reconstructed expectation")
    return result


def generate(design, contract, seed):
    validate(design, contract)
    require(isinstance(seed, str) and len(seed) == 64 and all(character in "0123456789abcdef" for character in seed),
            "Seed must be exactly 32 bytes of lowercase hex")
    counts, cases = Counter(), []
    for slot in trial_slots(design):
        cases.append(build_case(slot, seed, counts[slot["suite"]]))
        counts[slot["suite"]] += 1
    require(len({case["case_id"] for case in cases}) == 120
            and len({case["case_seed"] for case in cases}) == 120, "Qualification identity collision")
    return cases


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def create(output, seed=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    seed = secrets.token_hex(32) if seed is None else seed
    # Save the exact seed before constructing any case. A failure retains it.
    write(output / "seed.json", {"generator": GENERATOR, "seed": seed})
    design_raw = (root / "evaluation-design.draft.json").read_bytes()
    contract_raw = (root / "experiment-contract.draft.json").read_bytes()
    for name, raw in (("design.draft.json", design_raw), ("contract.draft.json", contract_raw)):
        (output / name).write_bytes(raw)
    (output / "sources").mkdir()
    source_names = ("qualification_inputs.py", "diagnostic_oracle.py", "equipment_development.py", "evaluation_design.py")
    source_receipts = {}
    for name in source_names:
        raw = (root / name).read_bytes()
        (output / "sources" / name).write_bytes(raw)
        source_receipts[name] = entry(raw)
    report = {"schema": "organism-qualification-input-manifest-v1",
              "created_at": datetime.now(timezone.utc).isoformat(), "generator": GENERATOR,
              "status": "draft_inputs_not_sealed", "seed_receipt": entry((output / "seed.json").read_bytes()),
              "source_receipts": source_receipts, "design_receipt": entry(design_raw), "contract_receipt": entry(contract_raw),
              "cases": [], "failure": None, "qualification_sealed": False, "ready_to_seal": False,
              "training_admitted": False, "model_calls": 0, "baseline_runs": 0,
              "native_execution_verified": False, "qualification_passed": False}
    write(output / "request.json", report)
    (output / "cases").mkdir()
    try:
        cases = generate(json.loads(design_raw), json.loads(contract_raw), seed)
        for case in cases:
            path = output / "cases" / (case["case_id"] + ".json")
            write(path, case)
            report["cases"].append({"slot_id": case["slot"]["slot_id"], "case_id": case["case_id"],
                                    "receipt": entry(path.read_bytes()), "document_sha256": fingerprint(case["document"]),
                                    "oracle_sha256": fingerprint(case["oracle_plan"]),
                                    "starting_state_sha256": fingerprint(case["starting_state"]),
                                    "time_binding_sha256": fingerprint(case["time_binding"])})
        require(all(entry((root / name).read_bytes()) == record for name, record in source_receipts.items()),
                "Input authoring source changed during generation")
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    write(output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-seed-from", type=Path)
    args = parser.parse_args()
    seed = json.loads(args.reuse_seed_from.read_bytes())["seed"] if args.reuse_seed_from else None
    result = create(args.output, seed)
    print(json.dumps({"cases": len(result["cases"]), "failure": result["failure"], "ready_to_seal": False}))
    if result["failure"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
