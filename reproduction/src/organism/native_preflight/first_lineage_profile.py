"""Fixed, opt-in owner test for the reduced x86-64 development milestone."""

import os
from pathlib import Path
import stat
import sys


CASES_NAME = "temperature-cases.frozen.json"
CASES_SHA256 = "aff7cf82494c5f1f194b0272245835b030990a978786c1ec51aac2d0889e9685"
CASES_BYTES = 33311
COLD_STARTS = 3
START_SECONDS = 1200
TASK_SECONDS = 3900
DRIVER_SECONDS = 7800
OBSERVER_SECONDS = 7850
MINIMUM_REMAINING_SECONDS = 8000
MAXIMUM_SAMPLES = 1600


def validate_selection(cases_sha256, architecture):
    if cases_sha256 is not None and (
            cases_sha256 != CASES_SHA256 or architecture != "x86_64"):
        raise ValueError("First-lineage observation requires the exact frozen cases and x86-64")
    return cases_sha256 is not None


def validate_cases(path):
    from ..first_lineage_cases import load_frozen
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != CASES_BYTES:
        raise ValueError("First-lineage observation requires the original frozen case file")
    return load_frozen(path, CASES_SHA256)


def restrict_owner_sources(root):
    """Keep the owner-side case generators unreadable by the application user."""
    root = Path(root)
    sys.dont_write_bytecode = True
    observed = {}
    for name in ("first_lineage_cases.py", "equipment_development.py"):
        path = root / name
        entry = path.lstat()
        if (not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1
                or entry.st_uid != os.geteuid()
                or list((root / "__pycache__").glob(path.stem + ".*.pyc"))):
            raise ValueError("Owner case source is linked, foreign-owned or already cached")
        path.chmod(0o600)
        observed[name] = {"uid": path.stat().st_uid, "mode": stat.S_IMODE(path.stat().st_mode)}
    return observed


def descriptor():
    return {
        "schema": "organism-native-first-lineage-profile-v1",
        "cases": {"bytes": CASES_BYTES, "sha256": CASES_SHA256},
        "cold_starts": COLD_STARTS,
        "start_seconds": START_SECONDS,
        "task_seconds": TASK_SECONDS,
        "driver_seconds": DRIVER_SECONDS,
        "observer_seconds": OBSERVER_SECONDS,
        "minimum_remaining_seconds": MINIMUM_REMAINING_SECONDS,
        "maximum_samples": MAXIMUM_SAMPLES,
        "qualification_passed": False,
    }
