"""Inside-individual discovery, local reasoning and bounded peer establishment."""

import asyncio
import hashlib
import re
from pathlib import Path

from .capsule import MAX_MANIFEST_BYTES, verify_capsule_directory
from .discovery import public_observation
from .reproduction_agent import decide
from .reproduction_evidence import verify_choice
from .reproduction_journal import ReproductionError, ReproductionJournal, TERMINAL


STEP_SECONDS = 5
UNCERTAIN_SECONDS = 60
RECEIPT_FIELDS = {"operation_id", "parent_device_id", "device_id", "individual_id", "release_id",
                  "status", "ready_once", "ready_evidence", "preparation_failures",
                  "preparation_allowance_exhausted", "transport_receipt"}


class Reproduction:
    def __init__(self, config, work, discovery, client, capsule, *, planner=decide, interval=STEP_SECONDS):
        self.config, self.discovery, self.client = config, discovery, client
        self.book, self.clock, self.boot_id = discovery.book, discovery.clock, discovery.boot_id
        if (client.identity != config.device_id or client.policy.policy_id != self.book.policy.policy_id
                or discovery.local_identity != config.device_id or discovery.start_token != config.start_token):
            raise ReproductionError("Reproduction context belongs to another local individual")
        if type(interval) not in (int, float) or not 0.01 <= interval <= STEP_SECONDS:
            raise ReproductionError("Invalid reproduction observation interval")
        self.interval, self.planner = interval, planner
        self.capsule = Path(capsule).absolute()
        path = self.capsule / ".manifest.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ReproductionError("Complete carried capsule manifest is required")
        self.manifest = path.read_bytes()
        policy = self.book.policy.body
        self.envelope = verify_capsule_directory(
            self.capsule, self.manifest, self.book.policy.release_keys,
            require_trained=policy["require_trained"], max_payload_bytes=policy["max_payload_bytes"])
        model = self.envelope["body"]["model"]
        declared = self.envelope["body"]["files"][model["path"]]
        if (self.envelope["release_id"] != config.release_id or declared["sha256"] != config.model_sha256
                or declared["bytes"] != config.model_bytes
                or model["trained_for_organism"] != config.trained_for_organism
                or config.model.resolve() != (self.capsule / model["path"]).resolve()):
            raise ReproductionError("Carried reproductive model differs from the locally configured model")
        self.files = tuple(sorted(self.envelope["body"]["files"]))
        self.journal = ReproductionJournal(Path(work) / "reproduction", config, self.book)
        self.closed, self.candidate = False, None

    async def parent_ready(self):
        result = await self.client.request(self.book.endpoint(self.config.device_id), "self_status")
        expected = {"device_id", "policy_id", "owner_mode", "status", "individual_id",
                    "release_id", "incarnation_sha256", "ready", "ready_evidence", "transport_receipt"}
        if not isinstance(result, dict) or set(result) != expected or type(result["ready"]) is not bool:
            raise ReproductionError("Unexpected local admission receipt")
        self.client.check_anchor(result)
        if result["device_id"] != self.config.device_id or result["policy_id"] != self.book.policy.policy_id:
            raise ReproductionError("Local admission receipt identity differs")
        if result["ready"] is not True:
            return False
        if (result["owner_mode"] != "enabled" or result["status"] != "ready"
                or result["individual_id"] != self.config.individual_id or result["release_id"] != self.config.release_id
                or result["incarnation_sha256"] != hashlib.sha256(self.config.start_token.encode()).hexdigest()
                or not isinstance(result["ready_evidence"], str)
                or not re.fullmatch("[0-9a-f]{64}", result["ready_evidence"])):
            raise ReproductionError("Local readiness belongs to another incarnation or owner state")
        return True

    def receipt(self, intent, result):
        if not isinstance(result, dict) or set(result) != RECEIPT_FIELDS:
            raise ReproductionError("Unexpected child receipt fields")
        self.client.check_anchor(result)
        expected = {"operation_id": intent["operation_id"], "parent_device_id": self.config.device_id,
                    "device_id": intent["peer_id"], "release_id": self.config.release_id}
        if (any(result[key] != value for key, value in expected.items())
                or not isinstance(result["individual_id"], str)
                or not re.fullmatch("[0-9a-f]{32}", result["individual_id"])
                or result["individual_id"] == self.config.individual_id
                or type(result["ready_once"]) is not bool
                or result["status"] not in {"receiving", "staged", "starting", "running", "ready",
                                             "stop_requested", "aborted", "failed", "stopped"}):
            raise ReproductionError("Child receipt differs from the recorded reproduction intent")
        previous = intent["receipt"]
        if previous and previous["individual_id"] != result["individual_id"]:
            raise ReproductionError("Child identity changed during the same operation")
        if result["status"] == "ready":
            if (result["ready_once"] is not True or not isinstance(result["ready_evidence"], str)
                    or not re.fullmatch("[0-9a-f]{64}", result["ready_evidence"])):
                raise ReproductionError("Child readiness lacks its owner-side evidence")
            phase = "complete"
        elif result["status"] in ("aborted", "failed", "stopped"):
            phase = result["status"]
        else:
            phase = "receiving" if result["status"] == "receiving" else "waiting"
        intent.update(phase=phase, receipt=result, failure=None, retry_clock=self.clock() + self.interval)
        return self.journal.save(intent)

    def attempt(self, intent, action):
        attempts = intent["attempts"]
        if attempts.get(action, 0) >= 3:
            intent.update(phase="uncertain", failure="Mutation allowance exhausted: " + action,
                          retry_clock=self.clock() + UNCERTAIN_SECONDS)
            self.journal.save(intent)
            return False
        attempts[action] = attempts.get(action, 0) + 1
        self.journal.save(intent)  # The attempt exists before network side effects.
        return True

    async def choose(self, app, intent):
        store, request = self.journal.decisions, intent["operation_id"]
        existing = store.get(request)
        if existing is None:
            existing, _ = store.reserve(request, intent["peer_id"])
        if existing["record"]["status"] == "running":
            if app.busy:
                return
            app.busy = True
            try:
                await self.planner(app.inference, self.client, self.book, self.envelope, store, request, intent["peer_id"])
            finally:
                app.busy = False
            existing = store.get(request)
        record, choice = existing["record"], existing["record"]["submission"]
        if record["status"] == "finished":
            try:
                choice = verify_choice(existing, peer_id=intent["peer_id"], start_token=intent["decision_token"],
                                       envelope=self.envelope)
            except (ValueError, KeyError, TypeError) as exc:
                intent.update(phase="decision_failed", failure=("Invalid reproduction trace: " + str(exc))[:192],
                              decision_hash=existing["digest"])
                self.journal.save(intent)
                return
        if record["status"] != "finished" or not isinstance(choice, dict):
            intent.update(phase="decision_failed", failure=(record.get("failure") or "No complete model choice")[:192])
        elif (choice.get("device_id") != intent["peer_id"] or choice.get("release_id") != self.config.release_id
              or choice.get("action") not in ("establish", "defer")):
            intent.update(phase="decision_failed", failure="Persisted model choice belongs to another target or release")
        elif choice["action"] == "defer":
            intent.update(phase="deferred")
        elif (choice.get("platform") != self.book.policy.platform(intent["peer_id"])
              or choice["platform"] not in self.envelope["body"]["platforms"]):
            intent.update(phase="decision_failed", failure="Persisted model choice selected an unsupported procedure")
        else:
            intent.update(phase="authorized", platform=choice["platform"])
        intent["decision_hash"] = existing["digest"]
        self.journal.save(intent)

    async def advance(self, intent):
        endpoint = self.book.endpoint(intent["peer_id"])
        expired = intent["boot_id"] != self.boot_id or self.clock() >= intent["deadline_clock"]
        if expired and not intent["attempts"].get("reserve"):
            intent.update(phase="failed", failure="Original lifetime ended before any reservation was attempted")
            return self.journal.save(intent)
        if intent["phase"] == "uncertain":
            # A timeout is not evidence of absence. Continue observing this
            # exact operation; never free the parent slot on an unknown result.
            result = await self.client.request(endpoint, "status", operation_id=intent["operation_id"])
            return self.receipt(intent, result)
        if expired:
            intent["phase"] = "cancelling"
        if intent["phase"] == "cancelling":
            if self.attempt(intent, "cancel"):
                result = await self.client.request(endpoint, "cancel", operation_id=intent["operation_id"],
                                                   reason="Original reproduction deadline or incarnation ended")
                return self.receipt(intent, result)
            return
        if intent["phase"] in ("authorized", "reserving"):
            intent["phase"] = "reserving"
            if self.attempt(intent, "reserve"):
                observed = await self.client.observe(endpoint)
                public_observation(observed, self.book.policy, intent["peer_id"])
                result = await self.client.reserve(endpoint, intent["operation_id"], self.manifest, observed)
                return self.receipt(intent, result)
            return
        if intent["phase"] == "receiving":
            index = intent["payload_index"]
            if index < len(self.files):
                if not self.attempt(intent, "file-" + str(index)):
                    return
                name = self.files[index]
                await self.client.upload(endpoint, intent["operation_id"], name,
                                         self.envelope["body"]["files"][name], self.capsule)
                intent.update(payload_index=index + 1, failure=None, retry_clock=self.clock())
                return self.journal.save(intent)
            if self.attempt(intent, "commit"):
                result = await self.client.request(endpoint, "commit", operation_id=intent["operation_id"])
                return self.receipt(intent, result)
            return
        if intent["phase"] == "waiting":
            result = await self.client.request(endpoint, "status", operation_id=intent["operation_id"])
            return self.receipt(intent, result)
        raise ReproductionError("Unknown active reproduction transition")

    async def step(self, app):
        if self.closed or not app.health()["inference_ready"]:
            return
        active = self.journal.active()
        intent = active[0] if active else None
        if intent is None and self.candidate is None and self.discovery.events.empty():
            return
        if intent and intent["boot_id"] == self.boot_id and self.clock() < intent["retry_clock"]:
            return
        try:
            if not await self.parent_ready():
                return
            if intent is None:
                if self.candidate is None:
                    try:
                        identity = self.discovery.events.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    self.discovery.pending.discard(identity)
                    self.candidate = identity
                candidate = next(item for item in self.discovery.snapshot() if item["device_id"] == self.candidate)
                if candidate["state"] == "stale":
                    # A queued arrival can age while another child is being
                    # established. Keep it until the next background probe.
                    return
                self.candidate = None
                if candidate["state"] != "current" or not candidate["available_observed"]:
                    return
                intent = self.journal.propose(candidate, boot_id=self.boot_id, clock=self.clock())
                if intent is None:
                    return
            if intent["phase"] == "proposed":
                if intent["boot_id"] != self.boot_id or self.clock() >= intent["deadline_clock"]:
                    intent.update(phase="decision_failed", failure="Original decision lifetime ended before any remote mutation")
                    self.journal.save(intent)
                else:
                    await self.choose(app, intent)
            else:
                await self.advance(intent)
        except (OSError, ValueError, TimeoutError, asyncio.IncompleteReadError) as exc:
            # The durable attempt counter is reread, because an operation can
            # fail after its intent was saved but before a reply was received.
            if intent is not None:
                latest = self.journal.operations[intent["operation_id"]]
                if latest["phase"] not in TERMINAL:
                    value = dict(latest)
                    value.update(failure=(type(exc).__name__ + ": " + str(exc))[:192],
                                 retry_clock=self.clock() + (UNCERTAIN_SECONDS if latest["phase"] == "uncertain"
                                                            else self.interval))
                    self.journal.save(value)

    async def run(self, app):
        while not self.closed:
            await self.step(app)
            await asyncio.sleep(self.interval)

    def public(self):
        return self.journal.public()
