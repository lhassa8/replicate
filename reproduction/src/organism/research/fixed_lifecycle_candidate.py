"""Unadmitted lifecycle correction for a future, separately frozen comparison.

The October 17 native experiment did not carry this module. Its frozen
control_policy.FixedApplication and control_entry remain unchanged.
"""

import asyncio

from organism.reproduction import UNCERTAIN_SECONDS
from organism.reproduction_journal import TERMINAL
from organism.research.control_policy import FixedApplication, FixedReproduction


class FixedLifecycleCandidate(FixedApplication):
    """Supervise policy availability while continuing to deny model inference."""

    async def run(self):
        try:
            await self.start()
            while not self.stop_event.is_set():
                if not self.health()["policy_ready"]:
                    raise RuntimeError("Fixed policy application became unavailable")
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass
            if self.failure:
                raise RuntimeError(self.failure)
        except BaseException as exc:
            self.receipt["failure"] = type(exc).__name__ + ": " + str(exc)[:512]
            raise
        finally:
            await self.close()


class FixedReproductionLifecycleCandidate(FixedReproduction):
    """Use the same policy-liveness boundary for the reproduction controller.

    The frozen Reproduction.step also requires inference_ready. This explicit
    candidate retains its original state transitions and error handling, with
    policy_ready as the initial liveness condition. It never reports inference.
    """

    async def step(self, app):
        if self.closed or not app.health()["policy_ready"]:
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
                candidate = next(item for item in self.discovery.snapshot()
                                 if item["device_id"] == self.candidate)
                if candidate["state"] == "stale":
                    return
                self.candidate = None
                if candidate["state"] != "current" or not candidate["available_observed"]:
                    return
                intent = self.journal.propose(candidate, boot_id=self.boot_id, clock=self.clock())
                if intent is None:
                    return
            if intent["phase"] == "proposed":
                if intent["boot_id"] != self.boot_id or self.clock() >= intent["deadline_clock"]:
                    intent.update(phase="decision_failed",
                                  failure="Original decision lifetime ended before any remote mutation")
                    self.journal.save(intent)
                else:
                    await self.choose(app, intent)
            else:
                await self.advance(intent)
        except (OSError, ValueError, TimeoutError, asyncio.IncompleteReadError) as exc:
            if intent is not None:
                latest = self.journal.operations[intent["operation_id"]]
                if latest["phase"] not in TERMINAL:
                    value = dict(latest)
                    value.update(failure=(type(exc).__name__ + ": " + str(exc))[:192],
                                 retry_clock=self.clock() + (
                                     UNCERTAIN_SECONDS if latest["phase"] == "uncertain" else self.interval))
                    self.journal.save(value)
