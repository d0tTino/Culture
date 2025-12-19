"""Typed data models that drive Council Mode interactions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class CouncilMemberConfig(BaseModel):
    """Configuration describing a single council member persona."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    member_id: str = Field(validation_alias=AliasChoices("member_id", "id"))
    display_name: str = Field(validation_alias=AliasChoices("display_name", "name"))
    persona: str = ""
    role: str = "Generalist"
    model: str | None = None
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int | None = Field(
        default=None, validation_alias=AliasChoices("max_tokens", "max_turn_tokens")
    )
    is_active: bool = True
    system_prompt: str = ""
    description: str = ""
    decision_weight: float = Field(default=1.0, ge=0.0)
    metadata: Mapping[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        normalized = dict(data)
        normalized.setdefault("member_id", normalized.get("id") or normalized.get("name"))
        normalized.setdefault(
            "display_name", normalized.get("name") or normalized.get("member_id") or "member"
        )
        persona = normalized.get("persona") or normalized.get("description")
        normalized.setdefault("persona", persona or "")
        normalized.setdefault("description", persona or "")
        normalized.setdefault("metadata", normalized.get("metadata") or {})

        max_turn_tokens = normalized.get("max_turn_tokens")
        if max_turn_tokens is not None and "max_tokens" not in normalized:
            normalized["max_tokens"] = max_turn_tokens

        if not normalized.get("system_prompt"):
            role = normalized.get("role") or "Generalist"
            normalized["system_prompt"] = (
                f"You are {normalized['display_name']}, a {role}. Offer concise, grounded answers."
            )

        return normalized


class CouncilConfig(BaseModel):
    """Top-level configuration for enabling Council Mode in a scenario."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    enabled: bool = True
    voting_mode: str = "single_winner"
    members: list[CouncilMemberConfig]
    quorum: int | None = None
    consensus_threshold: float = 0.67
    max_rounds: int = 1
    auto_record_transcript: bool = True
    max_concurrent_calls: int | None = None
    du_budget_per_question: float | None = None
    metadata: Mapping[str, Any] | None = None

    @field_validator("members")
    @classmethod
    def _validate_members(cls, value: list[CouncilMemberConfig]) -> list[CouncilMemberConfig]:
        if not value:
            raise ValueError("Council configuration must include at least one member")
        active_members = [member for member in value if member.is_active]
        if not active_members:
            raise ValueError("At least one council member must be active")
        return active_members


class CouncilQuestion(BaseModel):
    """Represents a structured question posed to the council."""

    question_id: str
    prompt: str
    context: str | None = None
    rag_documents: list[str] = Field(default_factory=list)
    metadata: Mapping[str, Any] | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class MemberAnswer(BaseModel):
    """Stores an individual council member's answer and supporting details."""

    member_id: str
    answer: str
    confidence: float | None = None
    reasoning: str | None = None
    citations: list[str] = Field(default_factory=list)
    votes: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: Mapping[str, Any] | None = None


class CouncilOutcome(BaseModel):
    """Final aggregated outcome once the council deliberation concludes."""

    question: CouncilQuestion
    answers: Sequence[MemberAnswer]
    resolution: str
    winner_id: str | None = None
    winning_member_ids: list[str] = Field(default_factory=list)
    votes: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None
    metadata: Mapping[str, Any] | None = None

    @model_validator(mode="after")
    def _sync_winner_ids(self) -> CouncilOutcome:
        if self.winner_id and not self.winning_member_ids:
            self.winning_member_ids = [self.winner_id]
        elif self.winning_member_ids and not self.winner_id:
            self.winner_id = self.winning_member_ids[0]
        return self

    def has_consensus(self) -> bool:
        """Return ``True`` if a consensus was declared."""

        return bool(self.winning_member_ids)
