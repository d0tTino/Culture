"""Typed data models that drive Council Mode interactions."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

CouncilRole = Literal["Facilitator", "Innovator", "Analyzer", "Red Team", "Generalist"]
CouncilVotingMode = Literal["judge_llm", "peer_vote", "heuristic"]

_ROLE_ALIASES = {
    "facilitator": "Facilitator",
    "moderator": "Facilitator",
    "innovator": "Innovator",
    "analyzer": "Analyzer",
    "red team": "Red Team",
    "redteam": "Red Team",
    "generalist": "Generalist",
}
_VOTING_MODE_ALIASES = {
    "judge_llm": "judge_llm",
    "judge": "judge_llm",
    "llm_judge": "judge_llm",
    "single_winner": "judge_llm",
    "singlewinner": "judge_llm",
    "peer_vote": "peer_vote",
    "peer-vote": "peer_vote",
    "peer": "peer_vote",
    "consensus": "peer_vote",
    "heuristic": "heuristic",
    "deterministic": "heuristic",
    "rule_based": "heuristic",
}


def _normalize_role(value: Any) -> str:
    if value is None:
        return "Generalist"
    text = str(value).strip()
    if not text:
        return "Generalist"
    key = re.sub(r"[\s_-]+", " ", text).strip().lower()
    return _ROLE_ALIASES.get(key, text)


def _normalize_voting_mode(value: Any) -> str:
    if value is None:
        return "single_winner"
    text = str(value).strip()
    if not text:
        return "single_winner"
    key = re.sub(r"[\s-]+", "_", text).strip().lower()
    return _VOTING_MODE_ALIASES.get(key, text)


class CouncilMemberConfig(BaseModel):
    """Configuration describing a single council member persona."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    member_id: str = Field(validation_alias=AliasChoices("member_id", "id"))
    display_name: str = Field(validation_alias=AliasChoices("display_name", "name"))
    persona: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(ge=1, validation_alias=AliasChoices("max_tokens", "max_turn_tokens"))
    role: CouncilRole = Field(min_length=1)
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
        role = _normalize_role(normalized.get("role") or "Generalist")
        normalized["role"] = role

        persona = (
            normalized.get("persona")
            or normalized.get("description")
            or f"{normalized['display_name']} focuses on the {role} perspective."
        )
        normalized.setdefault("persona", persona)
        normalized.setdefault("description", normalized.get("description") or persona)

        metadata = normalized.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        normalized.setdefault("metadata", metadata)
        if "model" not in normalized and metadata.get("model"):
            normalized["model"] = metadata.get("model")

        max_turn_tokens = normalized.get("max_turn_tokens")
        if max_turn_tokens is not None and "max_tokens" not in normalized:
            normalized["max_tokens"] = max_turn_tokens
        normalized.setdefault("max_tokens", 256)
        normalized.setdefault("temperature", 0.3)
        normalized.setdefault("is_active", True)

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
    voting_mode: CouncilVotingMode = "judge_llm"
    members: list[CouncilMemberConfig] = Field(default_factory=list)
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
        if not any(member.is_active for member in value):
            raise ValueError("At least one council member must be active")
        return value

    @field_validator("voting_mode", mode="before")
    @classmethod
    def _normalize_voting_mode(cls, value: Any) -> Any:
        return _normalize_voting_mode(value)


class CouncilQuestion(BaseModel):
    """Represents a structured question posed to the council."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    question_id: str = Field(validation_alias=AliasChoices("question_id", "id", "name"))
    prompt: str
    user_id: str | None = Field(default=None, validation_alias=AliasChoices("user_id", "userId"))
    extra_context: str | None = Field(
        default=None, validation_alias=AliasChoices("extra_context", "context")
    )
    rag_documents: list[str] = Field(default_factory=list)
    metadata: Mapping[str, Any] | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)

    @property
    def context(self) -> str | None:
        return self.extra_context

    @context.setter
    def context(self, value: str | None) -> None:
        self.extra_context = value


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
