from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field


class TraitDelta(BaseModel):
    """A bounded update applied to a single trait."""

    trait: str
    before: float
    proposed_delta: float
    bounded_delta: float
    after: float


class PersonalityTransition(BaseModel):
    """Event emitted whenever personality traits transition through the reducer."""

    step: int
    cause: str
    source: str
    max_step: float
    input_signals: dict[str, float] = Field(default_factory=dict)
    deltas: list[TraitDelta] = Field(default_factory=list)
    resulting_traits: dict[str, float] = Field(default_factory=dict)


class TraitTransitionLog(BaseModel):
    """Versioned transition log with replay verification guarantees."""

    schema_version: int = 1
    seed_traits: dict[str, float] = Field(default_factory=dict)
    transitions: list[PersonalityTransition] = Field(default_factory=list)
    hash_chain: list[str] = Field(default_factory=list)

    @staticmethod
    def _canonical_hash(previous_hash: str, transition: PersonalityTransition) -> str:
        payload = {
            "previous_hash": previous_hash,
            "transition": transition.model_dump(mode="python")
            if hasattr(transition, "model_dump")
            else transition.dict(),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def append(self, transition: PersonalityTransition) -> None:
        previous = self.hash_chain[-1] if self.hash_chain else "GENESIS"
        self.transitions.append(transition)
        self.hash_chain.append(self._canonical_hash(previous, transition))

    def verify_hash_chain(self) -> bool:
        expected: list[str] = []
        previous = "GENESIS"
        for transition in self.transitions:
            current = self._canonical_hash(previous, transition)
            expected.append(current)
            previous = current
        return expected == self.hash_chain
