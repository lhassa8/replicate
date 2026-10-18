"""Create a Markdown review packet without claiming human review or admission."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .baseline_review import audit as audit_baseline
from .inheritance_dataset import IDENTIFIER_POLICY, opaque_equipment_case
from .prepare_review import card as original_card, record_digest, validate_decisions
from .runtime_rows import inspect_record, targets
from .training_inputs import load, require, sha256, validate_recipe, verified_path


def card(record):
    if record["role"] != "equipment_inherited":
        return original_card(record)
    fixture = record["case"]["fixture"]
    selected = list(targets(record))
    state = record["case"]["knowledge_state"]
    action = ("Use the verified reader, then read a fresh measurement."
              if state == "applicable" else
              "Configure a reader from the returned capabilities, then read a fresh measurement.")
    diagnosis = (f"Report {fixture['expected_status']}: {fixture['expected_value']} {fixture['expected_unit']}."
                 if fixture["expected_value"] is not None else
                 "Report an inconclusive result because the current measurement is stale, invalid or missing.")
    return {
        "case_id": record["case_id"], "role": record["role"], "split": record["split"],
        "family": record["case"]["family_id"], "variation": record["case"]["variation"],
        "knowledge_state": state, "targets": len(selected),
        "supervised_assistants": [True] * len(selected), "context_only_errors": 0,
        "summary": action + " " + diagnosis,
        "review_evidence": {
            "expected_status": fixture["expected_status"], "expected_value": fixture["expected_value"],
            "expected_unit": fixture["expected_unit"], "expected_reader": fixture["expected_reader"],
            "device_document": fixture["document"], "knowledge_support": record["knowledge"],
        },
        "messages": record["messages"],
        "provenance": {"provenance": record["provenance"], "automated_record_evaluation": record["evaluation"]},
        "record_sha256": record_digest(record), "source_file": "records/" + record["case_id"] + ".json",
    }


def case_markdown(item, output):
    """Evaluator information and model-visible messages remain clearly separate."""
    lines = [
        "# " + item["case_id"], "",
        f"Role: {item['role']}. Split: {item['split']}. Family: {item['family']}.",
        f"Condition: {item['variation']}. Learning targets: {item['targets']}.",
        "", "**Review status: unreviewed.**", "", item["summary"], "",
        f"[Complete source record]({output / item['source_file']})",
        f"[Shared tool contracts]({output / 'contracts.json'})", "",
        "## Evaluator information", "",
        "This section is review evidence; it is not added to the model's conversation.", "",
        "```json", json.dumps(item["review_evidence"], indent=2, allow_nan=False), "```", "",
        "## Exact model-visible conversation", "",
    ]
    ordinal = 0
    for index, message in enumerate(item["messages"]):
        label = message["role"]
        if label == "assistant":
            label += " — complete learning target" if item["supervised_assistants"][ordinal] else " — context only"
            ordinal += 1
        lines += [f"### Message {index + 1}: {label}", "",
                  "```json", json.dumps(message, indent=2, allow_nan=False), "```", ""]
    lines += ["## Provenance and automated checks", "",
              "```json", json.dumps(item["provenance"], indent=2), "```", ""]
    return "\n".join(lines)


def compare_original_reference(record, originals):
    """Identify exact retention or the narrowly allowed equipment ID change."""
    identity = record["case_id"]
    if identity in originals:
        require(record_digest(record) == record_digest(originals[identity]),
                "An unchanged original reference differs")
        return identity, "unchanged"
    if record["role"] == "equipment_inherited":
        return None
    old_id = record["case"].get("original_case_id")
    require(record["role"] == "equipment" and old_id in originals,
            "A reference has no original equipment scenario")
    original = originals[old_id]
    require(record["case"] == opaque_equipment_case(original["case"]),
            "Reidentified equipment changed its original scenario")
    for field in ("schema", "role", "split", "profile_id", "tool_contract_sha256",
                  "provenance", "evaluation", "supervised_assistants"):
        require(record.get(field) == original.get(field),
                "Reidentified equipment changed " + field)

    def without_identity(arguments):
        return {key: value for key, value in arguments.items() if key not in ("device_id", "evidence_id")}

    # Independent semantic inspection checks the new evidence IDs and every
    # message. Here only the explicit identity arguments may differ.
    require([{"tool": action["tool"], "arguments": without_identity(action["arguments"])}
             for action in record["actions"]] ==
            [{"tool": action["tool"], "arguments": without_identity(action["arguments"])}
             for action in original["actions"]], "Reidentified equipment changed its actions")
    require(without_identity(record["submission"]) == without_identity(original["submission"]),
            "Reidentified equipment changed its answer")
    return old_id, "reidentified"


def build(root, output, recipe_path, baseline_path):
    root, output, recipe_path, baseline_path = map(lambda path: Path(path).resolve(),
                                                   (root, output, recipe_path, baseline_path))
    recipe = load(recipe_path)
    validate_recipe(recipe)
    require(recipe["corpus"] == "inherited_equipment_runtime" and recipe["training_admitted"] is False,
            "Expected the unadmitted inheritance pilot")
    paths = {name: verified_path(root, value) for name, value in recipe["inputs"].items()}
    source, manifest = paths["source_manifest"].parent, load(paths["source_manifest"])
    runtime_audit, row_audit = load(paths["runtime_audit"]), load(paths["row_audit"])
    require(runtime_audit["passed"] is True
            and runtime_audit["source_manifest_sha256"] == sha256(paths["source_manifest"]),
            "Missing matching inherited runtime audit")
    require(row_audit["passed"] is True and len(row_audit["requests"]) == 2112
            and row_audit["token_report_sha256"] == sha256(paths["token_report"]),
            "Missing complete inherited token-row audit")
    original_source = root / "organism/data/equipment-runtime-development-v1"
    baseline = audit_baseline(baseline_path, original_source)
    require(baseline["expected_behavior_passed"] is True, "Baseline review has unexplained failures")
    original_records = {record["case_id"]: record for split in ("train", "validation")
                        for record in [json.loads(line) for line in
                                       (original_source / (split + ".jsonl")).read_text().splitlines()]}
    contracts, release = load(paths["contracts"]), load(paths["release"])
    records, seen, retained = [], set(), set()
    retention = {"unchanged": 0, "reidentified": 0}
    for split in ("train", "validation"):
        path = paths["source_" + split]
        require(sha256(path) == manifest["files"][path.name]["sha256"], "Review source split changed")
        for line in path.read_text().splitlines():
            record = json.loads(line)
            identity = record["case_id"]
            require(record["split"] == split and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", identity)
                    and identity not in seen, "Invalid or duplicate review identity")
            seen.add(identity)
            require(inspect_record(record, contracts[record["role"]], release) == record["evaluation"],
                    "Inherited reference semantic evaluation differs")
            comparison = compare_original_reference(record, original_records)
            if comparison is not None:
                original_id, kind = comparison
                require(original_id not in retained, "Original scenario was counted twice")
                original_record = original_records[original_id]
                require(inspect_record(original_record, contracts[original_record["role"]], release)
                        == original_record["evaluation"], "Original reference semantics differ")
                retained.add(original_id)
                retention[kind] += 1
            records.append(record)
    require(len(records) == 464 and retained == set(original_records) and len(retained) == 320,
            "Review inventory does not preserve all original references")
    expected_retention = ({"unchanged": 80, "reidentified": 240}
                          if manifest.get("identifier_policy") == IDENTIFIER_POLICY
                          else {"unchanged": 320, "reidentified": 0})
    require(retention == expected_retention, "Original scenario retention differs from identifier policy")
    cards = [card(record) for record in records]
    packet = {
        "schema": "organism-reference-review-packet-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "bindings": {
            "source_manifest_sha256": sha256(paths["source_manifest"]),
            "token_report_sha256": sha256(paths["token_report"]), "recipe_sha256": sha256(recipe_path),
            "baseline_report_sha256": sha256(baseline_path / "report.json"),
            "review_builder_sha256": sha256(Path(__file__)),
        },
        "cases": cards, "contracts": contracts, "baseline": baseline,
        "original_scenario_comparison": retention,
        "baseline_scope": "Original 240 equipment cases and 48 boundary cases; not a new inherited-arm comparison.",
        "target_count": sum(item["targets"] for item in cards),
        "context_only_errors": sum(item["context_only_errors"] for item in cards),
        "human_review_verified": False, "training_admitted": False, "qualification_passed": False,
        "limitations": manifest["limitations"] + [
            "The original review bindings do not approve this revised corpus or recipe.",
            "Source counterexamples show that passing one valid observation does not establish a generally sound procedure.",
            "The baseline assessment does not establish Q5 or Q7 performance, native execution or a frozen comparator.",
        ],
    }
    require(packet["target_count"] == 2112 and packet["context_only_errors"] == 8,
            "Review learning-target inventory differs")
    worksheet = {"schema": "organism-human-review-decisions-v1", "bindings": deepcopy(packet["bindings"]),
                 "cases": [{"case_id": item["case_id"], "record_sha256": item["record_sha256"],
                            "decision": "unreviewed", "notes": ""} for item in cards],
                 "human_review_verified": False, "training_admitted": False, "qualification_passed": False}
    validation = validate_decisions(worksheet, packet)
    require(validation["decisions"] == {"unreviewed": 464}, "Review must start with no invented decisions")
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("records", "cases"):
        (output / directory).mkdir()
    for record, item in zip(records, cards):
        (output / item["source_file"]).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        (output / "cases" / (item["case_id"] + ".md")).write_text(case_markdown(item, output))
    for filename, value in (("contracts.json", contracts), ("review-data.json", packet), ("worksheet.json", worksheet)):
        (output / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Inheritance pilot review", "", "**All 464 cases are unreviewed. Training is not admitted.**", "",
        f"This candidate retains all 320 original scenarios: {retention['unchanged']} records are unchanged "
        f"and {retention['reidentified']} equipment records use opaque identifiers. It also includes 144 inherited-reader references.",
        "There are 1,584 training targets and 528 development-validation targets. Each trajectory includes its final acknowledgement.",
        "", f"[Exact draft recipe]({recipe_path})",
        f"[Review worksheet]({output / 'worksheet.json'})",
        f"[Machine-readable review packet]({output / 'review-data.json'})",
        f"[Runtime replay audit]({paths['runtime_audit']})",
        f"[Full token-row audit]({paths['row_audit']})", "",
        "## What to review", "",
        "Check that each action follows from the model-visible tool results, that the reader and unit conversions are correct, "
        "and that the final answer uses a fresh valid observation. For absent or conflicting knowledge, verify that the model "
        "configures the reader from current capabilities. The omitted-validity source is a counterexample and must never be applied.",
        "",
        "Check whether synthetic source evidence, repeated scalar task families and the narrow error coverage are acceptable "
        "for this first pilot. Inspect the model-visible conversations for shortcuts and unintended outcome hints.",
        "",
        "Record decisions in the worksheet against these exact artifact identities. Its format validator cannot establish "
        "human review, freeze the experiment, authorize compute, or admit training.", "",
        "## Limitations", "",
    ]
    lines += ["- " + limitation for limitation in packet["limitations"]]
    lines += ["", "## Case index", "", "| Case | Split | Kind | Condition | Knowledge | Targets |",
              "|---|---|---|---|---|---:|"]
    for item in cards:
        lines.append(f"| [{item['case_id']}]({output / 'cases' / (item['case_id'] + '.md')}) | "
                     f"{item['split']} | {item['role']} | {item['variation']} | "
                     f"{item.get('knowledge_state', 'not used')} | {item['targets']} |")
    (output / "CASES.md").write_text("\n".join(lines) + "\n")
    files = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    receipt = {
        "schema": "organism-inheritance-review-artifact-v1", "bindings": packet["bindings"],
        "files": files, "references": 464, "targets": 2112,
        "original_references_unchanged": retention["unchanged"],
        "original_equipment_cases_reidentified": retention["reidentified"], "original_scenarios_retained": 320,
        "new_inherited_references": 144, "human_decisions_recorded": 0,
        "worksheet_format_check": validation, "human_review_verified": False,
        "training_admitted": False, "qualification_passed": False, "format": "Markdown and JSON; no browser required",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    result = build(Path(__file__).resolve().parents[1], args.output, args.recipe, args.baseline)
    print(json.dumps({key: result[key] for key in ("references", "targets", "original_references_unchanged",
                                                  "original_equipment_cases_reidentified", "new_inherited_references",
                                                  "human_decisions_recorded")}, indent=2))
