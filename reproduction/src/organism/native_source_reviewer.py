"""Configure each enrolled individual's local, optional source reviewer.

References arrive only in its checked, immutable release. The only signing key
is this device's owner-installed identity. An empty catalog or an untrusted
checker leaves review disabled while diagnostics and note exchange can run.
"""

import json
from pathlib import Path
import time

from .capsule import canonical
from .enrollment import key_id, public_bytes
from .experience import require
from .experience_endorsement import endorse
from .experience_proposal import check_proposal, sha
from .review_inbox import ReviewInbox
from .review_reference import FILENAME, read_references


CHECKER_SOURCES = (
    "native_source_reviewer.py", "review_reference.py", "experience_endorsement.py",
    "diagnostic_oracle.py", "audit_equipment_diagnostics.py", "audit_inherited_diagnostics.py",
    "audit_application.py", "experience.py", "experience_proposal.py", "source_inventory.py",
    "experience_store.py", "review_inbox.py", "equipment_interfaces.py", "enrollment.py", "capsule.py",
)


def checker_context(references):
    sources = {name: sha(Path(__file__).with_name(name).read_bytes()) for name in CHECKER_SOURCES}
    return {"checker_sha256": sha(canonical({
                "sources": sources, "reference_catalog_sha256": references.sha256})),
            "checker_sources": sources, "reference_catalog_sha256": references.sha256}


class SourceReviewer:
    def __init__(self, references, config, policy, key, *, clock=time.time, require_live_source=True):
        require(key_id(public_bytes(key.public_key())) == config.device_id,
                "Reviewer key belongs to a different device")
        require(type(require_live_source) is bool, "Live source requirement must be explicit")
        policy.peer_key(config.device_id)
        self.references, self.config, self.policy, self.key = references, config, policy, key
        self.clock, self.require_live_source = clock, require_live_source
        context = checker_context(references)
        self._context = canonical(context)
        self.checker_id = context["checker_sha256"]

    def __call__(self, proposal, observation, job, *, source_inventory=None, inventory_checkers=None):
        source = check_proposal(proposal, self.policy, release_id=self.config.release_id)
        require(source["verifier_device_id"] == self.config.device_id,
                "Source was assigned to another reviewer")
        plan = self.references.plan(source["capability_sha256"])
        moment = self.clock()
        require(type(moment) in (int, float) and 0 <= moment <= 4000000000, "Invalid review clock")
        created = int(moment)
        return endorse(
            proposal, observation, job, plan, policy=self.policy, verifier_key=self.key,
            release_id=self.config.release_id, created=created, expires=created + 3600,
            require_live_source=self.require_live_source, source_inventory=source_inventory,
            inventory_checkers=inventory_checkers, checker_context=json.loads(self._context))


def installed_reviews(knowledge_directory, work, config, policy, key, allowed_checkers):
    """Called only after entrypoint validates every installed file and owner."""
    references = read_references(Path(knowledge_directory) / FILENAME)
    if not len(references):
        return None
    reviewer = SourceReviewer(references, config, policy, key)
    if reviewer.checker_id not in allowed_checkers:
        return None
    return ReviewInbox(
        Path(work) / "source-review", config, policy, checker_id=reviewer.checker_id,
        reviewer=reviewer, inventory_key=key, inventory_checkers=allowed_checkers,
        require_live_source=True)
