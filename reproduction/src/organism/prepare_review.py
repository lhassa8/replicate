"""Build a local, self-contained viewer for the exact unadmitted pilot corpus."""

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .baseline_review import audit as audit_baseline
from .runtime_rows import inspect_record, targets
from .training_inputs import load, require, sha256, validate_recipe, verified_path


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def record_digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def card(record):
    case = record["case"]
    assistants = [message for message in record["messages"] if message["role"] == "assistant"]
    selected = list(targets(record))
    ordinals = {ordinal for ordinal, _, _ in selected}
    mask = [ordinal in ordinals for ordinal in range(len(assistants))]
    if record["role"] == "equipment":
        fixture = case["fixture"]
        value = fixture["expected_value"]
        summary = (f"Report {fixture['expected_status']}: {value} {fixture['expected_unit']}."
                   if value is not None else "Report an inconclusive result because the measurement is stale, invalid or missing.")
        evidence = {"expected_status": fixture["expected_status"], "expected_value": value,
                    "expected_unit": fixture["expected_unit"], "expected_reader": fixture["expected_reader"],
                    "device_document": fixture["document"]}
    else:
        submission = record["submission"]
        summary = f"Select {submission['action']} for the observed {case['owner_mode']} device on {case['platform']}."
        evidence = {"case": case, "expected_action": submission, "real_peer_used": False,
                    "descendant_established": False}
    return {
        "case_id": record["case_id"], "role": record["role"], "split": record["split"],
        "family": case.get("family_id", record["profile_id"]), "variation": case["variation"],
        "targets": len(selected), "supervised_assistants": mask, "context_only_errors": mask.count(False),
        "summary": summary, "review_evidence": evidence, "messages": record["messages"],
        "provenance": {"provenance": record["provenance"], "automated_record_evaluation": record["evaluation"]},
        "record_sha256": record_digest(record), "source_file": "records/" + record["case_id"] + ".json",
    }


def render(packet, template, script):
    require(template.count("__REVIEW_DATA__") == template.count("__REVIEW_SCRIPT__")
            == template.count("__SCRIPT_HASH__") == 1, "Review template placeholders differ")
    # The JSON is untrusted display data. It cannot terminate its data element.
    data = json.dumps(packet, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    require("</script" not in script.lower(), "Review script cannot contain an HTML script terminator")
    script_hash = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    replacements = {"__SCRIPT_HASH__": script_hash, "__REVIEW_DATA__": data, "__REVIEW_SCRIPT__": script}
    return re.sub(r"__(SCRIPT_HASH|REVIEW_DATA|REVIEW_SCRIPT)__", lambda match: replacements[match[0]], template)


def validate_decisions(review, packet):
    """Validate a user's exported record; never infer human consent or admission."""
    require(review.get("schema") == "organism-human-review-decisions-v1"
            and review.get("bindings") == packet["bindings"], "Review has different artifact bindings")
    require(all(review.get(key) is False for key in
                ("human_review_verified", "training_admitted", "qualification_passed")),
            "An exported worksheet cannot assert verified review or training admission")
    originals = {item["case_id"]: item for item in packet["cases"]}
    entries = review.get("cases")
    require(isinstance(entries, list) and len(entries) == len(originals), "Review case inventory is incomplete")
    seen = set()
    counts = Counter()
    for item in entries:
        require(isinstance(item, dict), "Invalid review record")
        identity = item.get("case_id")
        require(identity in originals and identity not in seen
                and item.get("record_sha256") == originals[identity]["record_sha256"], "Review record identity changed")
        require(item.get("decision") in ("unreviewed", "approved", "needs_changes")
                and isinstance(item.get("notes"), str) and len(item["notes"]) <= 4096,
                "Invalid review decision or notes")
        seen.add(identity)
        counts[item["decision"]] += 1
    return {"format_valid": True, "decisions": dict(counts), "human_review_verified": False,
            "training_admitted": False, "qualification_passed": False}


def build(root, output):
    root = Path(root).resolve()
    base = root / "organism"
    recipe_path = base / "training-recipe.draft.json"
    recipe = load(recipe_path)
    validate_recipe(recipe)
    require(recipe["corpus"] == "equipment_runtime" and recipe["training_admitted"] is False,
            "Review packet expects the current unadmitted equipment pilot")
    paths = {name: verified_path(root, value) for name, value in recipe["inputs"].items()}
    source = paths["source_manifest"].parent
    source_manifest = load(paths["source_manifest"])
    runtime_audit, row_audit = load(paths["runtime_audit"]), load(paths["row_audit"])
    require(runtime_audit["passed"] and runtime_audit["source_manifest_sha256"] == sha256(paths["source_manifest"]),
            "Missing matching reference audit")
    require(row_audit["passed"] and row_audit["token_report_sha256"] == sha256(paths["token_report"])
            and len(row_audit["requests"]) == 1440, "Missing complete token-row audit")
    baseline_directory = base / "evidence/baseline-review-20261014-01"
    baseline = audit_baseline(baseline_directory, source)
    require(baseline["expected_behavior_passed"], "Baseline review has unexplained failures")
    assets = Path(__file__).with_name("review_assets")
    template_path, script_path = assets / "reference_review.html", assets / "reference_review.js"
    contracts, release = load(paths["contracts"]), load(paths["release"])
    records = []
    source_lines = {}
    for split in ("train", "validation"):
        path = paths["source_" + split]
        require(sha256(path) == source_manifest["files"][path.name]["sha256"], "Source split changed")
        for line in path.read_text().splitlines():
            record = json.loads(line)
            identity = record["case_id"]
            require(record["split"] == split and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", identity)
                    and identity not in source_lines, "Invalid or duplicate review identity")
            require(inspect_record(record, contracts[record["role"]], release) == record["evaluation"],
                    "Reference semantic evaluation differs")
            records.append(record)
            source_lines[identity] = line + "\n"
    require(len(records) == 320, "Incomplete reference review")
    cards = [card(record) for record in records]
    packet = {
        "schema": "organism-reference-review-packet-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "bindings": {
            "source_manifest_sha256": sha256(paths["source_manifest"]), "token_report_sha256": sha256(paths["token_report"]),
            "recipe_sha256": sha256(recipe_path), "baseline_report_sha256": sha256(baseline_directory / "report.json"),
            "viewer_template_sha256": sha256(template_path), "viewer_script_sha256": sha256(script_path),
        },
        "cases": cards, "contracts": contracts, "baseline": baseline,
        "target_count": sum(item["targets"] for item in cards),
        "context_only_errors": sum(item["context_only_errors"] for item in cards),
        "human_review_verified": False, "training_admitted": False, "qualification_passed": False,
        "limitations": [
            "The cases and their fixed-policy demonstrations were authored by the project agent; no human review is claimed.",
            "Six equipment families train and two development families validate; four qualification families are excluded.",
            "Values and clocks are related deterministic variations, not independent physical devices.",
            "Eight injected errors cover one malformed configuration-argument type, not general recovery competence.",
            "Reproduction references use fixture peers without real package transfer, native startup or a useful descendant.",
            "The fixed policy handles one unambiguous scalar measurement and sampling time; it has no multi-counter or event-record reasoning.",
            "The policy has no repair/retry procedure after an unexpected tool or observation error.",
            "The 280 successful development/boundary diagnoses and eight ambiguity refusals are not Q5 qualification.",
            "Training and qualification remain subject to their separate frozen contracts and required compute authorization.",
        ],
    }
    require(packet["target_count"] == 1440 and packet["context_only_errors"] == 8, "Review target inventory differs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "records").mkdir()
    file_hashes = {}
    for identity, text in source_lines.items():
        path = output / "records" / (identity + ".json")
        path.write_text(text)
        file_hashes[str(path.relative_to(output))] = sha256(path)
    (output / "review-data.json").write_text(json.dumps(packet, indent=2) + "\n")
    (output / "index.html").write_text(render(packet, template_path.read_text(), script_path.read_text()))
    for name in ("review-data.json", "index.html"):
        file_hashes[name] = sha256(output / name)
    receipt = {
        "schema": "organism-reference-review-artifact-v1", "created_at": packet["created_at"],
        "bindings": packet["bindings"], "files": file_hashes,
        "source_hashes": {"prepare_review.py": sha256(Path(__file__)), "baseline_review.py": sha256(Path(__file__).with_name("baseline_review.py")),
                         "reference_review.html": sha256(template_path), "reference_review.js": sha256(script_path)},
        "references": len(records), "targets": packet["target_count"], "context_only_errors": packet["context_only_errors"],
        "human_decisions_recorded": 0, "human_review_verified": False, "training_admitted": False, "qualification_passed": False,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(Path(__file__).resolve().parents[1], args.output)
    print(json.dumps({key: result[key] for key in
                      ("references", "targets", "context_only_errors", "human_decisions_recorded", "training_admitted")}, indent=2))
