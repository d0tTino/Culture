"""Typed data models that drive Council Mode interactions."""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import Field


@dataclass(slots=True)
class CouncilMemberConfig:
    """Configuration describing a single council member persona."""

    member_id: str
    display_name: str
    role: str
    description: str
    system_prompt: str
    decision_weight: float = 1.0
    max_turn_tokens: int | None = None
    metadata: Mapping[str, Any] | None = None


CouncilMembers = Annotated[Sequence[CouncilMemberConfig], Field(min_length=1)]


@dataclass(slots=True)
class CouncilConfig:
    """Top-level configuration for enabling Council Mode in a scenario."""

    enabled: bool
    members: CouncilMembers
    quorum: int | None = None
    consensus_threshold: float = 0.67
    max_rounds: int = 1
    auto_record_transcript: bool = True
    metadata: Mapping[str, Any] | None = None

    def requires_quorum(self) -> bool:
        """Return ``True`` when the council needs to meet a quorum."""

        return self.quorum is not None and self.quorum > 0


@dataclass(slots=True)
class CouncilQuestion:
    """Represents a structured question posed to the council."""

    question_id: str
    prompt: str
    context: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(slots=True)
class MemberAnswer:
    """Stores an individual council member's answer and supporting details."""

    member_id: str
    answer: str
    confidence: float | None = None
    reasoning: str | None = None
    citations: MutableSequence[str] = field(default_factory=list)
    metadata: Mapping[str, Any] | None = None


@dataclass(slots=True)
class CouncilOutcome:
    """Final aggregated outcome once the council deliberation concludes."""

    question: CouncilQuestion
    answers: Sequence[MemberAnswer]
    resolution: str
    winning_member_ids: Sequence[str] = field(default_factory=list)
    summary: str | None = None
    metadata: Mapping[str, Any] | None = None

    def has_consensus(self) -> bool:
        """Return ``True`` if a consensus was declared."""

        return bool(self.winning_member_ids)
