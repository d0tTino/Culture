from __future__ import annotations

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
