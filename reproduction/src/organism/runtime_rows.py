"""Explicit corpus dispatch and supervised targets for supported runtime data."""

import json
from pathlib import Path

from .audit_runtime_dataset import inspect_record as inspect_legacy, require


EQUIPMENT_SCHEMA = "organism-equipment-runtime-trajectory-v1"
MIXED_SCHEMA = "organism-equipment-runtime-corpus-v1"
INHERITED_SCHEMA = "organism-inherited-equipment-runtime-trajectory-v1"
INHERITED_CORPUS_SCHEMA = "organism-inherited-equipment-runtime-corpus-v1"
STARTUP_CORPUS_SCHEMA = "organism-startup-complete-runtime-corpus-v1"


def inspect_record(record, contract, release):
    if record["schema"] == "organism-first-lineage-runtime-trajectory-v1":
        from .audit_first_lineage_dataset import inspect_record as inspect_first_lineage
        return inspect_first_lineage(record, contract, release)
    if record["schema"] == INHERITED_SCHEMA:
        from .audit_inheritance_dataset import inspect_record as inspect_inherited
        return inspect_inherited(record, contract, release)
    if record["schema"] == EQUIPMENT_SCHEMA:
        from .audit_equipment_dataset import inspect_record as inspect_equipment
        return inspect_equipment(record, contract, release)
    return inspect_legacy(record, contract, release)


def targets(record):
    """Yield original assistant ordinal, message index and complete target.

Only the original equipment schema can contain an untrained injected action.
Its independent record inspector must verify that exclusion before use.
    """
    require(record["schema"] in (EQUIPMENT_SCHEMA, INHERITED_SCHEMA, "organism-runtime-trajectory-v1",
                                "organism-first-lineage-runtime-trajectory-v1"),
            "Unsupported trajectory target schema")
    assistants = [(index, message) for index, message in enumerate(record["messages"])
                  if message["role"] == "assistant"]
    if record["schema"] in (EQUIPMENT_SCHEMA, "organism-first-lineage-runtime-trajectory-v1"):
        mask = record["supervised_assistants"]
        require(isinstance(mask, list) and len(mask) == len(assistants)
                and all(type(value) is bool for value in mask), "Invalid assistant supervision mask")
    else:
        mask = [True] * len(assistants)
    for ordinal, ((index, message), trainable) in enumerate(zip(assistants, mask)):
        if trainable:
            yield ordinal, index, message


async def audit(source):
    manifest = json.loads((Path(source) / "manifest.json").read_bytes())
    if manifest["schema"] == "organism-first-lineage-runtime-corpus-v1":
        from .audit_first_lineage_dataset import audit as audit_first_lineage
        return await audit_first_lineage(source)
    if manifest["schema"] == STARTUP_CORPUS_SCHEMA:
        from .audit_startup_dataset import audit as audit_startup
        return await audit_startup(source)
    if manifest["schema"] == INHERITED_CORPUS_SCHEMA:
        from .audit_inheritance_dataset import audit as audit_inherited
        return await audit_inherited(source)
    if manifest["schema"] == MIXED_SCHEMA:
        from .audit_equipment_dataset import audit as audit_equipment
        return await audit_equipment(source)
    from .audit_runtime_dataset import audit as audit_legacy
    return await audit_legacy(source)
