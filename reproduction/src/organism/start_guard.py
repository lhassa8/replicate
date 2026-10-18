"""Owner-installed systemd ExecCondition; standard library only.

Exit zero permits this exact persisted start intent. Any error or mismatch
denies startup. This script does not create state, launch a model, or restart
itself. Run by the trusted receiving substrate, never from a received capsule.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import time


def check_start_permission(directory, policy_id, device_id, operation_id, process_token, *,
                           boot_id=None, now=None, owner_uid=0, purpose="application"):
    directory = Path(directory).absolute()
    if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in (policy_id, device_id)):
        return False
    if not all(re.fullmatch(r"[0-9a-f]{32}", value) for value in (operation_id, process_token)):
        return False
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        entry = os.fstat(fd)
        if entry.st_uid != owner_uid or stat.S_IMODE(entry.st_mode) & 0o077:
            return False
        for name in ("enrolled.json", "state.sqlite3"):
            child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                info = os.fstat(child)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != owner_uid or info.st_nlink != 1:
                    return False
            finally:
                os.close(child)
    finally:
        os.close(fd)
    marker = json.loads((directory / "enrolled.json").read_text())
    if marker != {"schema": "organism-receiver-state-v1", "policy_id": policy_id, "device_id": device_id}:
        return False
    if boot_id is None:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    db = sqlite3.connect((directory / "state.sqlite3").as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        row = db.execute("SELECT body FROM state WHERE singleton=1").fetchone()
        state = json.loads(row[0]) if row else {}
        # Read the transaction snapshot before sampling time. A concurrent
        # observer may advance last_clock before this read, without a clock
        # reversal or any change to the persisted start authorization.
        if now is None:
            now = time.clock_gettime(time.CLOCK_BOOTTIME)
        if not math.isfinite(now) or now < 0:
            return False
        if (state.get("policy_id"), state.get("device_id"), state.get("boot_id"), state.get("owner_mode"),
                state.get("current_birth")) != (policy_id, device_id, boot_id, "enabled", operation_id):
            return False
        if now < state["last_clock"]:
            return False
        row = db.execute("SELECT body FROM births WHERE operation_id=?", (operation_id,)).fetchone()
        birth = json.loads(row[0]) if row else {}
        status = "starting" if purpose == "application" else "running"
        if (purpose not in ("application", "readiness") or birth.get("process_token") != process_token
                or birth.get("status") != status or birth.get("cancelled")):
            return False
        if not birth.get("ready_once") and now >= birth["deadline"]:
            return False
        previous, sequence = "0" * 64, 0
        for expected, (sequence, ancestor, digest, body) in enumerate(db.execute(
                "SELECT sequence,previous_hash,event_hash,body FROM events ORDER BY sequence"), 1):
            event = json.loads(body)
            if (sequence != expected or ancestor != previous or event["sequence"] != sequence or event["revision"] != sequence
                    or hashlib.sha256(bytes.fromhex(previous) + body.encode()).hexdigest() != digest):
                return False
            previous = digest
        if sequence != state["revision"]:
            return False
        if purpose == "readiness":
            path = directory / "supervision.sqlite3"
            entry = path.lstat()
            if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != owner_uid
                    or entry.st_nlink != 1 or entry.st_mode & 0o077):
                return False
            receipts = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
            try:
                row = receipts.execute("SELECT digest,body FROM records WHERE token=? AND kind='check_intent'",
                                       (process_token,)).fetchone()
            finally:
                receipts.close()
            if row is None or len(row[1].encode()) > 65536 or hashlib.sha256(row[1].encode()).hexdigest() != row[0]:
                return False
            record = json.loads(row[1])
            expected = {"schema": "organism-supervision-record-v1", "kind": "check_intent",
                        "policy_id": policy_id, "device_id": device_id, "operation_id": operation_id,
                        "individual_id": birth["individual_id"], "process_token": process_token,
                        "manifest_sha256": birth["manifest_sha256"], "boot_id": boot_id}
            if any(record.get(key) != value for key, value in expected.items()):
                return False
            intent = record["observation"]
            # Compare the same sum used when publishing the intent. Subtracting
            # fractional boot clocks can round a valid 240-second window away
            # from 240; accepting a tolerance would also permit changed bounds.
            if (intent["application_process"] != birth["process_observation"]
                    or not intent["issued_clock"] <= now < intent["deadline_clock"]
                    or intent["deadline_clock"] != intent["issued_clock"] + 240):
                return False
        return True
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--process-token", required=True)
    parser.add_argument("--purpose", choices=("application", "readiness"), default="application")
    args = parser.parse_args()
    if os.geteuid() != 0:
        return 1
    try:
        allowed = check_start_permission(args.state, args.policy_id, args.device_id,
                                         args.operation_id, args.process_token, purpose=args.purpose)
    except Exception as exc:
        print("organism-start-guard: permission check failed: " + type(exc).__name__, file=sys.stderr)
        return 1
    if not allowed:
        print("organism-start-guard: exact persisted start permission denied", file=sys.stderr)
    return 0 if allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
