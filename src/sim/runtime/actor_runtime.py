from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

ActorId = Literal["world", "knowledge", "governance"] | str


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """Append-only event envelope used for actor-to-actor communication."""

    event_type: str
    from_actor: ActorId
    to_actor: ActorId
    payload: dict[str, Any]
    step: int
    event_index: int = -1
    version_vector: dict[str, int] = field(default_factory=dict)
    attempt: int = 1


class Sequencer:
    """Centralized deterministic ordering service for all emitted events."""

    def __init__(self) -> None:
        self._counter = 0
        self._vector: dict[str, int] = {}

    def stamp(self, envelope: EventEnvelope) -> EventEnvelope:
        self._counter += 1
        producer = envelope.from_actor
        self._vector[producer] = self._vector.get(producer, 0) + 1
        vector = dict(self._vector)
        return EventEnvelope(
            event_type=envelope.event_type,
            from_actor=envelope.from_actor,
            to_actor=envelope.to_actor,
            payload=dict(envelope.payload),
            step=envelope.step,
            event_index=self._counter,
            version_vector=vector,
            attempt=envelope.attempt,
        )


@dataclass(slots=True)
class MailboxDelivery:
    envelope: EventEnvelope
    delivery_id: str
    delivered_at: float


class Mailbox:
    """Mailbox with explicit ACK and retry semantics."""

    def __init__(self, owner: ActorId, *, retry_timeout_s: float = 0.05) -> None:
        self.owner = owner
        self._retry_timeout_s = retry_timeout_s
        self._ready: deque[EventEnvelope] = deque()
        self._inflight: dict[str, MailboxDelivery] = {}

    def push(self, envelope: EventEnvelope) -> None:
        self._ready.append(envelope)

    def receive(self) -> MailboxDelivery | None:
        self._requeue_expired()
        if not self._ready:
            return None
        envelope = self._ready.popleft()
        delivery_id = f"{envelope.event_index}:{envelope.attempt}:{len(self._inflight)}"
        delivery = MailboxDelivery(
            envelope=envelope,
            delivery_id=delivery_id,
            delivered_at=time.monotonic(),
        )
        self._inflight[delivery_id] = delivery
        return delivery

    def ack(self, delivery_id: str) -> None:
        self._inflight.pop(delivery_id, None)

    def _requeue_expired(self) -> None:
        now = time.monotonic()
        expired = [
            delivery_id
            for delivery_id, delivery in self._inflight.items()
            if now - delivery.delivered_at >= self._retry_timeout_s
        ]
        for delivery_id in expired:
            delivery = self._inflight.pop(delivery_id)
            retried = EventEnvelope(
                event_type=delivery.envelope.event_type,
                from_actor=delivery.envelope.from_actor,
                to_actor=delivery.envelope.to_actor,
                payload=dict(delivery.envelope.payload),
                step=delivery.envelope.step,
                event_index=delivery.envelope.event_index,
                version_vector=dict(delivery.envelope.version_vector),
                attempt=delivery.envelope.attempt + 1,
            )
            self._ready.appendleft(retried)

    def depth(self) -> int:
        return len(self._ready) + len(self._inflight)


@dataclass(slots=True)
class ActorService:
    actor_id: ActorId
    state: dict[str, Any] = field(default_factory=dict)

    def process(self, envelope: EventEnvelope) -> list[EventEnvelope]:
        if envelope.event_type == "agent_step" and self.actor_id == "world":
            total = int(self.state.get("world_updates", 0)) + 1
            self.state["world_updates"] = total
            agent_id = str(envelope.payload["agent_id"])
            step = int(envelope.step)
            return [
                EventEnvelope(
                    event_type="world_update",
                    from_actor="world",
                    to_actor="knowledge",
                    payload={"agent_id": agent_id, "world_updates": total},
                    step=step,
                )
            ]

        if envelope.event_type == "world_update" and self.actor_id == "knowledge":
            total = int(self.state.get("knowledge_updates", 0)) + 1
            self.state["knowledge_updates"] = total
            agent_id = str(envelope.payload["agent_id"])
            step = int(envelope.step)
            return [
                EventEnvelope(
                    event_type="knowledge_recorded",
                    from_actor="knowledge",
                    to_actor="governance",
                    payload={"agent_id": agent_id, "knowledge_updates": total},
                    step=step,
                )
            ]

        if envelope.event_type == "knowledge_recorded" and self.actor_id == "governance":
            total = int(self.state.get("governance_decisions", 0)) + 1
            self.state["governance_decisions"] = total
            return []

        return []


class RuntimeOrchestrator:
    """Actor runtime with sequenced envelopes and mailbox-based delivery."""

    def __init__(self, agent_count: int) -> None:
        self.agent_ids = [f"agent-{i}" for i in range(agent_count)]
        self.sequencer = Sequencer()
        self.event_log: list[EventEnvelope] = []
        self.mailboxes: dict[ActorId, Mailbox] = {
            "world": Mailbox("world"),
            "knowledge": Mailbox("knowledge"),
            "governance": Mailbox("governance"),
        }
        self.services: dict[ActorId, ActorService] = {
            "world": ActorService("world"),
            "knowledge": ActorService("knowledge"),
            "governance": ActorService("governance"),
        }
        for agent_id in self.agent_ids:
            self.mailboxes[agent_id] = Mailbox(agent_id)
            self.services[agent_id] = ActorService(agent_id)

    def _append_event(self, envelope: EventEnvelope) -> EventEnvelope:
        stamped = self.sequencer.stamp(envelope)
        self.event_log.append(stamped)
        self.mailboxes[stamped.to_actor].push(stamped)
        return stamped

    def submit_agent_step(self, *, agent_id: str, step: int) -> EventEnvelope:
        return self._append_event(
            EventEnvelope(
                event_type="agent_step",
                from_actor=agent_id,
                to_actor="world",
                payload={"agent_id": agent_id},
                step=step,
            )
        )

    async def drain(self) -> int:
        processed = 0
        while True:
            progressed = False
            for actor_id, mailbox in self.mailboxes.items():
                if actor_id not in {"world", "knowledge", "governance"}:
                    continue
                delivery = mailbox.receive()
                if delivery is None:
                    continue
                progressed = True
                emitted = self.services[actor_id].process(delivery.envelope)
                mailbox.ack(delivery.delivery_id)
                for envelope in emitted:
                    self._append_event(envelope)
                processed += 1
            if not progressed:
                break
            await asyncio.sleep(0)
        return processed

    def state_snapshot(self) -> dict[str, int]:
        return {
            "world_updates": int(self.services["world"].state.get("world_updates", 0)),
            "knowledge_updates": int(self.services["knowledge"].state.get("knowledge_updates", 0)),
            "governance_decisions": int(
                self.services["governance"].state.get("governance_decisions", 0)
            ),
        }


def replay_event_log(agent_count: int, event_log: list[EventEnvelope]) -> dict[str, int]:
    runtime = RuntimeOrchestrator(agent_count=agent_count)
    for envelope in event_log:
        runtime.mailboxes[envelope.to_actor].push(envelope)
    while True:
        progressed = False
        for actor_id in ("world", "knowledge", "governance"):
            delivery = runtime.mailboxes[actor_id].receive()
            if delivery is None:
                continue
            progressed = True
            runtime.services[actor_id].process(delivery.envelope)
            runtime.mailboxes[actor_id].ack(delivery.delivery_id)
        if not progressed:
            break
    return runtime.state_snapshot()
