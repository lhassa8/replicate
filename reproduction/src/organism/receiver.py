"""Connect durable admission to full capsule verification and atomic staging.

This local entry point is shared with the future authenticated transport. It
performs no network discovery, process launch or qualification by itself.
"""

from pathlib import Path
import hashlib
import os
import stat

from .capsule import CapsuleError, stage_capsule, verify_capsule_directory, verify_manifest
from .receiver_state import AdmissionError


MAX_STORE_BYTES = 16 * 1024**3
MAX_RELEASES = 2


def stored_bytes(directory):
    """Account for partials as well as published payloads without following links."""
    total = 0
    for parent, directories, files in os.walk(directory, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            value = path.lstat()
            if stat.S_ISLNK(value.st_mode):
                raise CapsuleError("Receiving store contains an unexpected symbolic link")
            if stat.S_ISREG(value.st_mode):
                total += value.st_size
            elif not stat.S_ISDIR(value.st_mode):
                raise CapsuleError("Receiving store contains a special file")
    return total


class LocalReceiver:
    def __init__(self, state, store, *, max_store_bytes=MAX_STORE_BYTES):
        self.state = state
        self.store = Path(store).absolute()
        self.store.mkdir(mode=0o700, parents=False, exist_ok=True)
        entry = self.store.lstat()
        if not stat.S_ISDIR(entry.st_mode) or entry.st_uid != os.geteuid() or stat.S_IMODE(entry.st_mode) & 0o077:
            raise AdmissionError("Capsule store must be private to the receiving service")
        if type(max_store_bytes) is not int or not 1 <= max_store_bytes <= MAX_STORE_BYTES:
            raise AdmissionError("Invalid capsule-store budget")
        self.max_store_bytes = max_store_bytes

    def receive(self, authenticated_peer, operation_id, observed_revision, manifest, source, *, admission_ticket):
        policy = self.state.policy.body
        envelope = verify_manifest(manifest, self.state.policy.release_keys,
                                   require_trained=policy["require_trained"], max_payload_bytes=policy["max_payload_bytes"])
        birth = self.state.reserve_birth(authenticated_peer, operation_id, observed_revision, manifest,
                                        admission_ticket=admission_ticket)
        if birth["status"] != "receiving":
            return self.state.peer_receipt(authenticated_peer, operation_id)
        destination = self.store / envelope["release_id"]
        verification = {"require_trained": policy["require_trained"], "max_payload_bytes": policy["max_payload_bytes"]}
        exists = os.path.lexists(destination)
        releases = [path for path in self.store.iterdir() if not path.name.startswith(".")]
        needed = 0 if exists else birth["payload_bytes"] + len(manifest)
        if (len(releases) + (0 if exists else 1) > MAX_RELEASES
                or stored_bytes(self.store) + needed > self.max_store_bytes):
            self.state.abort_birth(authenticated_peer, operation_id, "Receiving store allowance exhausted")
            raise AdmissionError("Receiving store allowance exhausted")
        if not exists:
            # stage_capsule owns the cross-process installation lock. A competing
            # duplicate request may receive a busy result and retry its same ID.
            stage_capsule(Path(source), self.store, manifest, self.state.policy.release_keys, **verification)
        # A previous process may have completed the rename and died before
        # committing the database transition. Reverification recovers that gap.
        verify_capsule_directory(destination, manifest, self.state.policy.release_keys, **verification)
        self.state.record_staged(operation_id, birth["manifest_sha256"])
        return self.state.peer_receipt(authenticated_peer, operation_id)

    def verify_before_start(self, operation_id, manifest):
        """Supervisor must call this before claiming and spawning an incarnation."""
        snapshot = self.state.snapshot()
        birth = snapshot["current_birth"]
        if not birth or birth["operation_id"] != operation_id or snapshot["owner_mode"] != "enabled":
            raise AdmissionError("No authorized birth is available for activation")
        if hashlib.sha256(manifest).hexdigest() != birth["manifest_sha256"]:
            raise AdmissionError("Activation manifest differs from the reserved birth")
        if (sum(not path.name.startswith(".") for path in self.store.iterdir()) > MAX_RELEASES
                or stored_bytes(self.store) > self.max_store_bytes):
            raise AdmissionError("Receiving store allowance exhausted before activation")
        policy = self.state.policy.body
        envelope = verify_capsule_directory(
            self.store / birth["release_id"], manifest, self.state.policy.release_keys,
            require_trained=policy["require_trained"], max_payload_bytes=policy["max_payload_bytes"],
        )
        if envelope["release_id"] != birth["release_id"]:
            raise AdmissionError("Activation package differs from the recorded release")
        # claim_start rechecks ownership after disk verification, closing an
        # owner-stop race without holding a DB transaction across large reads.
        return self.state.claim_start(operation_id)
