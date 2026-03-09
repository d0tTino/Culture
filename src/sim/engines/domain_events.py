from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DomainEvent:
    domain: str
    name: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TraitDriftApplied:
    agent_id: str
    step: int
    source: str
    cause: str
    deltas: dict[str, float]
    max_step: float
    input_signals: dict[str, float] = field(default_factory=dict)

    def to_domain_event(self) -> DomainEvent:
        return DomainEvent(
            domain="identity",
            name="TraitDriftApplied",
            payload={
                "agent_id": self.agent_id,
                "step": self.step,
                "source": self.source,
                "cause": self.cause,
                "deltas": dict(self.deltas),
                "max_step": float(self.max_step),
                "input_signals": dict(self.input_signals),
            },
        )


@dataclass(frozen=True, slots=True)
class LifecycleTransitioned:
    agent_id: str
    step: int
    from_state: str
    to_state: str
    reason: str
    legacy_artifacts: dict[str, Any] = field(default_factory=dict)
    memory_archival_policy: dict[str, Any] = field(default_factory=dict)

    def to_domain_event(self) -> DomainEvent:
        return DomainEvent(
            domain="population",
            name="LifecycleTransitioned",
            payload={
                "agent_id": self.agent_id,
                "step": int(self.step),
                "from_state": self.from_state,
                "to_state": self.to_state,
                "reason": self.reason,
                "legacy_artifacts": dict(self.legacy_artifacts),
                "memory_archival_policy": dict(self.memory_archival_policy),
            },
        )


@dataclass(frozen=True, slots=True)
class SuccessorRegistered:
    predecessor_id: str
    successor_id: str
    step: int
    inherited: dict[str, Any] = field(default_factory=dict)

    def to_domain_event(self) -> DomainEvent:
        return DomainEvent(
            domain="population",
            name="SuccessorRegistered",
            payload={
                "predecessor_id": self.predecessor_id,
                "successor_id": self.successor_id,
                "step": int(self.step),
                "relationship": "successor_of",
                "inherited": dict(self.inherited),
            },
        )


@dataclass(frozen=True, slots=True)
class SocialImpactApplied:
    departed_agent_id: str
    survivor_id: str
    step: int
    reason: str
    relationship_delta: float

    def to_domain_event(self) -> DomainEvent:
        return DomainEvent(
            domain="identity",
            name="SocialImpactApplied",
            payload={
                "departed_agent_id": self.departed_agent_id,
                "survivor_id": self.survivor_id,
                "step": int(self.step),
                "reason": self.reason,
                "relationship_delta": float(self.relationship_delta),
            },
        )


__all__ = [
    "DomainEvent",
    "LifecycleTransitioned",
    "SocialImpactApplied",
    "SuccessorRegistered",
    "TraitDriftApplied",
]
