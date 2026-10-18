"""Explicit wire representation for a choice that selects no child platform."""

from copy import deepcopy


RAW_PROFILE = "organism-reproduction-null-platform-v1"
PROFILE = "organism-reproduction-platform-choice-v2"
NO_PLATFORM = "none"
PLATFORMS = ("linux-x86_64", "linux-arm64")


def canonical_choice(arguments, profile):
    """Decode a declared protocol without changing the model's chosen action."""
    if profile not in (RAW_PROFILE, PROFILE):
        raise ValueError("Unknown reproduction choice protocol")
    if not isinstance(arguments, dict) or set(arguments) != {"action", "platform", "evidence_id"}:
        raise ValueError("Invalid reproduction choice fields")
    if arguments["action"] not in ("establish", "defer"):
        raise ValueError("Invalid reproduction action")
    platform = arguments["platform"]
    if arguments["action"] == "establish":
        if platform not in PLATFORMS:
            raise ValueError("Establishment requires a declared platform")
    else:
        expected = None if profile == RAW_PROFILE else NO_PLATFORM
        if platform != expected:
            raise ValueError("A deferred choice must explicitly select no platform")
        platform = None
    return {**deepcopy(arguments), "platform": platform}
