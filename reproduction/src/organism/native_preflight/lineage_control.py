"""Bound owner availability and observation commands; no descendant launch action."""

from datetime import datetime
import math
import time

from ..training_inputs import require
from . import lineage_network


def validate(value, args, permit, release_id, *, sequence, enabled, observed):
    require(isinstance(value, dict) and set(value) == {
        "schema", "instance_id", "slot", "prefix", "deadline", "release_id",
        "sequence", "action", "issued_at", "start_epoch", "seed_disconnection",
    } and value["schema"] == "organism-lineage-owner-control-v1",
        "Unexpected lineage owner control")
    require(value["instance_id"] == permit["instance_id"]
            and all(value[name] == getattr(args, name) for name in ("slot", "prefix", "deadline"))
            and value["release_id"] == release_id
            and type(value["sequence"]) is int and value["sequence"] == sequence,
            "Owner control changes its host, slot, release, sequence or fixed lease")
    issued = datetime.fromisoformat(value["issued_at"])
    require(issued.tzinfo is not None and 0 <= time.time() - issued.timestamp() <= 1200,
            "Owner control is stale or from the future")
    action = value["action"]
    require(action in ("enable", "finish", "endurance"), "No such owner observation action")
    require((action == "enable" and sequence == 1 and not enabled and not observed)
            or (action in ("finish", "endurance") and sequence == 2 and enabled and observed),
            "Owner control is out of sequence")
    proof = value["seed_disconnection"]
    needs_proof = (args.slot == "grandchild" and action == "enable") or action == "endurance"
    if needs_proof:
        require(isinstance(proof, dict) and set(proof) == {
            "schema", "seed_instance_id", "seed_network_interface_id", "observed_at",
            "instance_state", "interface_absent", "readback_sha256",
        } and proof["schema"] == "organism-lineage-seed-disconnection-v1"
            and proof["seed_instance_id"] == permit["instances"]["seed"]
            and proof["instance_state"] == "terminated" and proof["interface_absent"] is True,
            "Grandchild availability and endurance require the disconnected seed")
        import re
        require(re.fullmatch("eni-[0-9a-f]{17}", proof["seed_network_interface_id"])
                and re.fullmatch("[0-9a-f]{64}", proof["readback_sha256"]),
                "Seed disconnection lacks exact external evidence identities")
        at = datetime.fromisoformat(proof["observed_at"])
        require(at.tzinfo is not None and at.timestamp() <= issued.timestamp(),
                "Seed disconnection was observed after the control was issued")
    else:
        require(proof is None, "Unexpected seed-disconnection claim")
    if action == "endurance":
        require(args.slot in ("child", "grandchild")
                and type(value["start_epoch"]) in (int, float)
                and math.isfinite(value["start_epoch"])
                and 0 < value["start_epoch"] - time.time() <= 300,
                "Endurance must start together on the two descendants")
        from .profile_permit import remaining
        require(remaining(args.deadline) > value["start_epoch"] - time.time() + 7320,
                "The original lease cannot contain endurance and cleanup")
    else:
        require(value["start_epoch"] is None, "Unexpected endurance clock")
    lineage_network.slot_value(args.slot)
    return value
