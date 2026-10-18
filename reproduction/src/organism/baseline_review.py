"""Review the unchanged fixed policy on public development data and boundaries.

No candidate model, qualification fixture, optimizer or network is used.
Ambiguity refusals are reported separately from successful diagnoses.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import time

from .audit_equipment_dataset import verify_fixture
from .audit_equipment_diagnostics import inspect_job, resolve
from .capsule import canonical
from .equipment_baseline import solve
from .equipment_interfaces import InterfaceDirectory, InterfaceSession
from .training_inputs import load, require, sha256


VARIANTS = ("identity_and_order", "equivalent_scaling", "freshness_boundary",
            "expired_boundary", "future_timestamp", "ambiguous_measurement")
SOURCES = ("baseline_review.py", "equipment_baseline.py", "equipment_interfaces.py",
           "equipment.py", "capsule.py", "audit_equipment_diagnostics.py",
           "audit_application.py", "audit_equipment_dataset.py", "audit_runtime_dataset.py",
           "training_inputs.py")


def records(source):
    manifest = load(source / "manifest.json")
    require(manifest["schema"] == "organism-equipment-runtime-corpus-v1"
            and manifest["training_admitted"] is False
            and manifest["qualification_data_present"] is False, "Expected unadmitted development data")
    require(set(manifest["files"]) == {"train.jsonl", "validation.jsonl", "contracts.json", "release.json"},
            "Unexpected corpus inventory")
    for name, entry in manifest["files"].items():
        path = source / name
        require(sha256(path) == entry["sha256"] and path.stat().st_size == entry["bytes"],
                "Corpus artifact changed")
    result = [json.loads(line) for split in ("train", "validation")
              for line in (source / (split + ".jsonl")).read_text().splitlines()]
    equipment = [row for row in result if row["role"] == "equipment"]
    require(len(equipment) == 240 and len({row["case_id"] for row in equipment}) == 240,
            "Unexpected equipment reference inventory")
    for row in equipment:
        verify_fixture(row["case"])
    return equipment


def assign(source, selector, value):
    if source["format"] == "register_map":
        bank, address = selector.split(":")
        source["payload"][bank][address] = value
        return
    parts = [part.replace("~1", "/").replace("~0", "~") for part in selector[1:].split("/")]
    current = source["payload"]
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    current[int(parts[-1]) if isinstance(current, list) else parts[-1]] = value


def cases(source):
    for row in records(source):
        original = row["case"]
        yield {"case": deepcopy(original["fixture"]), "clock": original["clock"],
               "source_case_id": row["case_id"], "variant": "reference",
               "expected_policy_outcome": "diagnosis"}
        if original["variation"] != "fault" or original["seed"] != 2:
            continue
        for variant in VARIANTS:
            case = deepcopy(original["fixture"])
            identity = f"baseline-{original['profile_id']}-{variant.replace('_', '-')}"
            case["case_id"] = case["document"]["device_id"] = case["expected_reader"]["device_id"] = identity
            reader = case["expected_reader"]
            document = case["document"]
            source_item = next(item for item in document["sources"] if item["source_id"] == reader["source_id"])
            field = next(item for item in source_item["fields"] if item["selector"] == reader["value_selectors"][0])
            clock = original["clock"]
            if variant == "identity_and_order":
                document["family"] = "public-review-" + original["profile_id"]
                source_item["source_id"] = reader["source_id"] = "renamed-observations"
                source_item["fields"].reverse()
                document["sources"].reverse()
            elif variant == "equivalent_scaling":
                assign(source_item, field["selector"], float(Decimal(str(resolve(source_item, field["selector"]))) * 10))
                field["scale"] = float(Decimal(str(field["scale"])) / 10)
            elif variant in ("freshness_boundary", "expired_boundary", "future_timestamp"):
                age = document["task"]["maximum_age_seconds"]
                timestamp = clock - age if variant == "freshness_boundary" else clock - age - 0.001
                if variant == "future_timestamp":
                    timestamp = clock + 1
                assign(source_item, reader["timestamp_selector"], timestamp)
                if variant != "freshness_boundary":
                    case["expected_status"], case["expected_value"] = "unknown", None
            else:
                extra = deepcopy(field)
                extra.update(selector="/review_spare" if source_item["format"] == "json_document"
                             else "holding:60000", meaning="Spare measurement from another component")
                source_item["fields"].append(extra)
                assign(source_item, extra["selector"], 0)
            yield {"case": case, "clock": clock, "source_case_id": row["case_id"], "variant": variant,
                   "expected_policy_outcome": "decline" if variant == "ambiguous_measurement" else "diagnosis"}


def verify_expected_answer(case, clock):
    """Decimal answer check, independent of the fixed policy and runtime reader."""
    document, reader = case["document"], case["expected_reader"]
    source = next(item for item in document["sources"] if item["source_id"] == reader["source_id"])
    field = next(item for item in source["fields"] if item["selector"] == reader["value_selectors"][0])
    raw, sampled = resolve(source, field["selector"]), resolve(source, reader["timestamp_selector"])
    valid = True if reader["validity_selector"] is None else resolve(source, reader["validity_selector"])
    unknown = (type(raw) not in (int, float) or type(sampled) not in (int, float) or valid is not True
               or sampled > clock or clock - sampled > document["task"]["maximum_age_seconds"])
    result, status = None, "unknown"
    if not unknown:
        result = Decimal(str(raw)) * Decimal(str(field["scale"]))
        unit = field["unit"]
        if unit == "F":
            result = (result - 32) * 5 / 9
        elif unit == "K":
            result -= Decimal("273.15")
        elif unit == "in/s":
            result *= Decimal("25.4")
        elif unit == "ratio":
            result *= 100
        else:
            require(unit in ("C", "mm/s", "percent", "count"), "Unsupported review unit")
        low, high = (Decimal(str(value)) for value in document["task"]["normal_range"])
        status = "low" if result < low else "high" if result > high else "normal"
        result = float(round(result, 6))
    require(case["expected_status"] == status and case["expected_value"] == result
            and case["expected_unit"] == document["task"]["canonical_unit"],
            "Review truth differs from independent decimal interpretation")


def execute(item):
    case, clock = item["case"], item["clock"]
    verify_expected_answer(case, clock)
    raw = canonical(case["document"])
    with tempfile.TemporaryDirectory(prefix="baseline-review-") as temporary:
        directory = Path(temporary)
        path = directory / (case["case_id"] + ".json")
        path.write_bytes(raw)
        session = InterfaceSession(InterfaceDirectory(directory, clock=lambda: clock), case["case_id"])
        failure = None
        started = time.monotonic()
        try:
            solve(session, case["case_id"])
        except Exception as exc:
            failure = type(exc).__name__ + ": " + str(exc)
        seconds = time.monotonic() - started
        require(path.read_bytes() == raw, "Fixed policy changed its observation")
    record = {"device_id": case["case_id"], "task_kind": "equipment_interface_v1",
              "status": "finished" if failure is None else "failed", "failure": failure,
              "submission": session.submission, "trace": {"tool_events": session.events, "fixed_policy": True}}
    job = {"record": record, "digest": hashlib.sha256(canonical(record)).hexdigest()}
    saved = {**item, "raw_sha256": hashlib.sha256(raw).hexdigest(), "job": job,
             "seconds": seconds, "input_unchanged": True}
    saved["evaluation"] = inspect_saved(saved)
    return saved


def inspect_saved(saved):
    case, job = saved["case"], saved["job"]
    verify_expected_answer(case, saved["clock"])
    raw = canonical(case["document"])
    require(hashlib.sha256(raw).hexdigest() == saved["raw_sha256"]
            and saved["input_unchanged"] is True, "Review input binding changed")
    checked = inspect_job(case, raw, job)
    outcome = saved["expected_policy_outcome"]
    require(outcome in ("diagnosis", "decline"), "Unknown baseline review outcome")
    declined = (saved["variant"] == "ambiguous_measurement" and job["record"]["status"] == "failed"
                and job["record"]["failure"] == "ValueError: Fixed policy cannot select an unambiguous supported reader"
                and job["record"]["submission"] is None
                and [event["tool"] for event in job["record"]["trace"]["tool_events"]] == ["inspect_equipment"])
    return {"behavior_matches_declared_scope": checked["semantic_passed"] if outcome == "diagnosis" else declined,
            "diagnostic_success": checked["semantic_passed"], "declined_ambiguous_reader": declined,
            "independent_answer_check": True, "observation_and_evidence_verified": checked["observation_and_evidence_verified"],
            "tool_calls": checked["tool_calls"], "human_review_verified": False, "qualification_passed": False}


def audit(directory, source):
    report = load(directory / "report.json")
    require(report["source_manifest_sha256"] == sha256(source / "manifest.json"), "Corpus binding changed")
    planned = {item["case"]["case_id"]: item for item in cases(source)}
    require(set(report["cases"]) == set(planned) and len(planned) == 288, "Baseline review inventory changed")
    counts = {"cases": 0, "diagnostic_successes": 0, "ambiguity_declines": 0, "scope_matches": 0}
    for identity, item in planned.items():
        path = directory / (identity + ".json")
        require(sha256(path) == report["cases"][identity]["sha256"], "Baseline case artifact changed")
        saved = load(path)
        require({key: saved[key] for key in item} == item, "Review case differs from declared development transformation")
        checked = inspect_saved(saved)
        require(checked == saved["evaluation"] == report["cases"][identity]["evaluation"], "Baseline summary changed")
        counts["cases"] += 1
        counts["diagnostic_successes"] += checked["diagnostic_success"]
        counts["ambiguity_declines"] += checked["declined_ambiguous_reader"]
        counts["scope_matches"] += checked["behavior_matches_declared_scope"]
    for name, expected in report["source_hashes"].items():
        require(sha256(directory / "sources" / name) == expected
                and sha256(Path(__file__).parent / name) == expected, "Baseline or review source changed")
    return {"audit_passed": True, **counts, "expected_behavior_passed": counts["scope_matches"] == counts["cases"],
            "scope": "Public development cases and declared transformations; ambiguity refusals are not successful diagnoses",
            "human_review_verified": False, "qualification_passed": False, "native_execution_verified": False}


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "sources"
    snapshot.mkdir()
    report = {"schema": "organism-baseline-review-v1", "recorded_at": datetime.now(timezone.utc).isoformat(),
              "source_manifest_sha256": sha256(source / "manifest.json"),
              "status": "development_evidence_for_review_not_frozen", "cases": {}, "source_hashes": {},
              "human_review_verified": False, "training_admitted": False, "qualification_passed": False,
              "model_generations": 0, "optimizer_steps": 0, "failure": None}
    for name in SOURCES:
        path = Path(__file__).parent / name
        (snapshot / name).write_bytes(path.read_bytes())
        report["source_hashes"][name] = sha256(path)
    started = time.monotonic()
    try:
        for item in cases(source):
            saved = execute(item)
            identity = item["case"]["case_id"]
            path = output / (identity + ".json")
            path.write_text(json.dumps(saved, indent=2) + "\n")
            report["cases"][identity] = {"sha256": sha256(path), "evaluation": saved["evaluation"]}
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["seconds"] = round(time.monotonic() - started, 3)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    checked = audit(output, source)
    (output / "audit.json").write_text(json.dumps(checked, indent=2) + "\n")
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checked = run(args.source, args.output)
    print(json.dumps(checked, indent=2))
    raise SystemExit(0 if checked["expected_behavior_passed"] else 1)
