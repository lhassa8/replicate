"""Development-only temperature cases for the reduced lineage milestone."""

from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from .equipment_development import case, VARIATIONS


SCHEMA = "organism-first-lineage-temperature-cases-v1"
VALUES_C = {
    "nominal": ("12.5", "24", "37.25", "40"),
    "fault": ("41", "65.5", "9", "0"),
    "stale": ("15", "32.25", "55", "4"),
    "invalid": ("15", "32.25", "55", "4"),
    "missing": ("15", "32.25", "55", "4"),
}
STALE_AGES = (121, 300, 601, 3600)


def build_cases():
    """Keep answers private; publish only the ordinary equipment document."""
    cases = []
    for variation in VARIATIONS:
        for index, value_c in enumerate(VALUES_C[variation]):
            item = case("g", variation, now=0)
            # Opaque device IDs cannot disclose the condition or expected answer.
            token = hashlib.sha256(
                f"first-lineage-20261016/{variation}/{index}".encode()
            ).hexdigest()[:20]
            device_id = "equipment-" + token
            item["case_id"] = item["document"]["device_id"] = device_id
            item["expected_reader"]["device_id"] = device_id
            probe = item["document"]["sources"][0]["payload"]["status"]["probe"]
            if variation != "missing":
                probe["absolute"] = float(Decimal(value_c) + Decimal("273.15"))
            if variation == "stale":
                probe["sampled_at"] = -STALE_AGES[index]
            unknown = variation in ("stale", "invalid", "missing")
            value = Decimal(value_c)
            item["expected_value"] = None if unknown else float(value)
            item["expected_status"] = (
                "unknown" if unknown else
                "low" if value < 10 else "high" if value > 40 else "normal"
            )
            item["provenance"].update(
                milestone="first-lineage-20261016",
                source_family="public development-validation thermal-status-v4",
                evaluation_only=True,
                training_admitted=False,
                qualification=False,
            )
            cases.append(item)
    return cases


def validate_cases(cases):
    """Reject missing conditions, changed answers, and input edits after freeze."""
    expected = build_cases()
    if (cases != expected
            or Counter(item["variation"] for item in cases)
            != Counter({variation: 4 for variation in VARIATIONS})
            or len({item["case_id"] for item in cases}) != 20):
        raise ValueError("The twenty frozen temperature cases differ")
    return cases


def load_frozen(path, expected_sha256):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular frozen case file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Frozen temperature case checksum differs")
    document = json.loads(raw)
    if (document.get("schema") != SCHEMA
            or document.get("qualification_passed") is not False
            or document.get("training_admitted") is not False):
        raise ValueError("Frozen temperature case scope differs")
    return validate_cases(document["cases"])
