"""Check candidate-data provenance, replay, masks, and live schema agreement."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .probe_local import digest_file, write_json
from .setup_dataset import development_cases, replay_trajectory
from .setup_simulator import TOOLS, contract_hash


def audit(source: Path, tokens: Path, live: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Do not overwrite an existing alignment receipt")
    manifest = json.loads((source / "manifest.json").read_text())
    token_report = json.loads((tokens / "report.json").read_text())
    parity = json.loads((live / "template-parity.json").read_text())
    live_audit = json.loads((live / "audit.json").read_text())
    if not parity["passed"] or not live_audit["passed"]:
        raise ValueError("Live serving or receipt audit did not pass")
    if manifest["qualification_data_present"] or token_report["training_admitted"]:
        raise ValueError("Unexpected qualification material or training admission")
    if token_report["source_manifest_sha256"] != digest_file(source / "manifest.json"):
        raise ValueError("Tokenized data references a different manifest")
    if json.loads((source / "tools.json").read_text()) != TOOLS:
        raise ValueError("Dataset tools differ from current contract")
    for name, expected in manifest["files"].items():
        if digest_file(source / name) != expected["sha256"]:
            raise ValueError("Dataset payload hash changed")
    source_directory = source / "sources"
    source_directory.mkdir(exist_ok=True)
    for name, expected in manifest["sources"].items():
        current = Path(__file__).parent / name
        if digest_file(current) != expected:
            raise ValueError("Generator source no longer matches its manifest")
        frozen = source_directory / name
        if frozen.exists():
            if digest_file(frozen) != expected:
                raise ValueError("Frozen generator source changed")
        else:
            frozen.write_bytes(current.read_bytes())
    expected_cases = {"train": {}, "validation": {}}
    for split, variation, seed, case in development_cases():
        expected_cases[split][case.case_id] = case
    summary = {}
    for split in ("train", "validation"):
        records = [json.loads(line) for line in (source / f"{split}.jsonl").read_text().splitlines()]
        if len(records) != len(expected_cases[split]):
            raise ValueError("Trajectory inventory differs")
        seen = set()
        decisions = Counter()
        families = set()
        for record in records:
            case = expected_cases[split][record["case_id"]]
            if record["case_id"] in seen:
                raise ValueError("Duplicate trajectory")
            replay_trajectory(record, case)
            seen.add(case.case_id)
            families.add(case.family_id)
            decisions[case.case_id] = len(record["events"])
        token_file = tokens / f"{split}.tokens.jsonl"
        if digest_file(token_file) != token_report["splits"][split]["sha256"]:
            raise ValueError("Token payload hash changed")
        token_decisions = set()
        with token_file.open() as handle:
            for line in handle:
                row = json.loads(line)
                case = expected_cases[split][row["case_id"]]
                key = (case.case_id, row["decision_index"])
                if key in token_decisions or not 0 <= row["decision_index"] < decisions[case.case_id]:
                    raise ValueError("Duplicate or out-of-range decision")
                token_decisions.add(key)
                if row["family_id"] != case.family_id:
                    raise ValueError("Tokenized row changed API family")
                ids, labels, prefix = row["input_ids"], row["labels"], row["prompt_tokens"]
                if not (len(ids) == len(labels) <= 4096 and 0 < len(ids) - prefix <= 192):
                    raise ValueError("Token row exceeds the contract")
                if labels[:prefix] != [-100] * prefix or labels[prefix:] != ids[prefix:]:
                    raise ValueError("Prompt or target masking changed")
        if len(token_decisions) != sum(decisions.values()):
            raise ValueError("Missing tokenized decisions")
        summary[split] = {"trajectories": len(records), "decisions": len(token_decisions), "families": sorted(families)}
    if set(summary["train"]["families"]) & set(summary["validation"]["families"]):
        raise ValueError("API family leakage")
    request_count = 0
    selection = json.loads((live / "fixture-selection.json").read_text())
    for case in selection["cases"]:
        receipt = json.loads((live / f"{case['case_id']}.json").read_text())
        for request in receipt["requests"]:
            if request["body"]["tools"] != TOOLS:
                raise ValueError("Training schema differs from captured live schema")
            request_count += 1
    if request_count != len(parity["requests"]):
        raise ValueError("Serving parity inventory differs")
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True, "scope": "development candidate data and captured serving alignment",
        "candidate_dataset": str(source), "candidate_tokens": str(tokens),
        "dataset_manifest_sha256": digest_file(source / "manifest.json"),
        "token_report_sha256": digest_file(tokens / "report.json"),
        "live_audit_sha256": digest_file(live / "audit.json"),
        "live_parity_sha256": digest_file(live / "template-parity.json"),
        "audit_source_sha256": digest_file(Path(__file__)),
        "tool_contract_sha256": contract_hash(), "splits": summary,
        "live_requests_with_matching_schemas_and_native_templates": request_count,
        "training_admitted": False, "optimization_started": False,
        "qualification_data_included": False, "experiment_frozen": False,
        "rejected_representations": {
            "setup-development-v1": "Missing descriptions added by Strands to serving schemas.",
            "setup-development-v2": "Aliased property descriptions differ from actual serving schemas.",
        },
        "remaining": [
            "Verified native platform packages and complete offline capsule.",
            "Frozen admission/transport interfaces and independent qualification families.",
            "Complete bounded cloud launch, cost, cleanup plan, and spending authorization.",
            "Frozen experiment contract before optimization.",
        ],
    }
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.tokens, args.live, args.output)
    print(json.dumps({"passed": result["passed"], "splits": result["splits"], "live_requests": result["live_requests_with_matching_schemas_and_native_templates"]}, indent=2))
