"""Bounded source procedures carried in the immutable knowledge component.

The signed release is the authority for these review records. Schema checks
cannot establish that a human reviewed a procedure or that its interpretation
matches an equipment manual. Empty references are valid and disable review.
No measurement, expected diagnosis, qualification case, or device key belongs
in this catalog. Peers cannot add entries through the experience transport.
"""

import json
import os
from pathlib import Path
import stat
from types import MappingProxyType

from .capsule import canonical
from .diagnostic_oracle import parse, validate_plan
from .experience import exact, hash_value, integer, require, unique_object
from .experience_proposal import sha


SCHEMA = "organism-source-review-references-v1"
FILENAME = "SOURCE-REVIEW-REFERENCES.json"
MAX_BYTES = 1024 * 1024
MAX_PROCEDURES = 128


class ReviewReferences:
    def __init__(self, raw):
        require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_BYTES, "Invalid reference catalog size")
        try:
            value = json.loads(raw, object_pairs_hook=unique_object)
            exact(value, ("schema", "procedures"))
            require(value["schema"] == SCHEMA and isinstance(value["procedures"], dict)
                    and len(value["procedures"]) <= MAX_PROCEDURES, "Invalid reference catalog schema")
            plans = {}
            for identity, entry in value["procedures"].items():
                require(hash_value(identity), "Invalid reference capability identity")
                exact(entry, ("capabilities", "plan", "review"))
                capabilities = entry["capabilities"]
                exact(capabilities, ("family", "task", "sources"))
                require(sha(canonical(capabilities)) == identity, "Reference capabilities differ from their identity")
                require(isinstance(capabilities["sources"], list), "Invalid reference sources")
                sources = []
                for source in capabilities["sources"]:
                    exact(source, ("source_id", "format", "fields"))
                    sources.append({**source, "payload": {}})
                # Check descriptions and selectors without inventing an
                # observation, expected result, or source task.
                document = parse(canonical({
                    "schema": "organism-equipment-interface-v1", "device_id": "reference-check",
                    "sequence": 0, **capabilities, "sources": sources}))
                validate_plan(document, entry["plan"])
                review = entry["review"]
                exact(review, ("decision", "review_id", "reviewer", "reviewed_unix_seconds", "reference_sha256"))
                require(review["decision"] == "approved" and hash_value(review["reference_sha256"])
                        and integer(review["reviewed_unix_seconds"])
                        and all(isinstance(review[name], str) and 1 <= len(review[name]) <= 128
                                and all(32 <= ord(character) < 127 for character in review[name])
                                for name in ("review_id", "reviewer")),
                        "Reference has no explicit review record")
                plans[identity] = canonical(entry["plan"])
        except (TypeError, KeyError, AttributeError, RecursionError, UnicodeError) as exc:
            raise ValueError("Malformed source-review references") from exc
        self.sha256 = sha(raw)
        # Independent copies prevent caller mutations from changing an active
        # reviewer's approved procedure while keeping its identity unchanged.
        self._plans = MappingProxyType(plans)

    def __len__(self):
        return len(self._plans)

    def plan(self, capability_sha256):
        require(capability_sha256 in self._plans, "No locally reviewed source procedure")
        return json.loads(self._plans[capability_sha256])


def read_references(path):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and 0 < info.st_size <= MAX_BYTES,
                "References must be a bounded regular file")
        raw = source.read(MAX_BYTES + 1)
        require(len(raw) == info.st_size, "Reference catalog changed while reading")
    return ReviewReferences(raw)
