"""Typed data models that drive Council Mode interactions."""

from __future__ import annotations

import json
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
        return "judge_llm"
    text = str(value).strip()
    if not text:
        return "judge_llm"
    key = re.sub(r"[\s-]+", "_", text).strip().lower()
    return _VOTING_MODE_ALIASES.get(key, text)


def _normalize_extra_context(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        if not value:
            return None
        return {"text": value}
    return {"value": value}


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
    du_budget: float | None = Field(
        default=None, ge=0.0, validation_alias=AliasChoices("du_budget", "duBudget")
    )
    ip_budget: float | None = Field(
        default=None, ge=0.0, validation_alias=AliasChoices("ip_budget", "ipBudget")
    )
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
            persona_summary = normalized.get("persona") or normalized.get("description")
            normalized["system_prompt"] = (
                f"You are {normalized['display_name']} ({role}). Embrace this persona: {persona_summary}. "
                "Stay in-character while keeping answers concise and grounded."
            )

        return normalized


class CouncilConfig(BaseModel):
    """Top-level configuration for enabling Council Mode in a scenario."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    enabled: bool = True
    voting_mode: CouncilVotingMode = "judge_llm"
    allow_remote_models: bool = False
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

        seen_member_ids: set[str] = set()
        duplicate_member_ids: set[str] = set()
        seen_display_names: set[str] = set()
        duplicate_display_names: set[str] = set()
        for member in value:
            if member.member_id in seen_member_ids:
                duplicate_member_ids.add(member.member_id)
            else:
                seen_member_ids.add(member.member_id)

            display_name = member.display_name.strip()
            if not display_name:
                continue
            if display_name in seen_display_names:
                duplicate_display_names.add(display_name)
            else:
                seen_display_names.add(display_name)

        if duplicate_member_ids:
            duplicates = ", ".join(sorted(duplicate_member_ids))
            raise ValueError(f"Council member IDs must be unique; duplicates: {duplicates}")

        if duplicate_display_names:
            duplicates = ", ".join(sorted(duplicate_display_names))
            raise ValueError(
                f"Council member display names should be unique; duplicates: {duplicates}"
            )

        return value

    @field_validator("voting_mode", mode="before")
    @classmethod
    def _normalize_voting_mode(cls, value: Any) -> Any:
        return _normalize_voting_mode(value)


class CouncilQuestion(BaseModel):
    """Represents a structured question posed to the council."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    question_id: str = Field(
        validation_alias=AliasChoices("question_id", "questionId", "id", "name")
    )
    prompt: str = Field(validation_alias=AliasChoices("prompt", "question"))
    user_id: str | None = Field(default=None, validation_alias=AliasChoices("user_id", "userId"))
    extra_context: dict[str, Any] | None = Field(
        default=None, validation_alias=AliasChoices("extra_context", "extraContext", "context")
    )
    rag_documents: list[str] = Field(default_factory=list)
    metadata: Mapping[str, Any] | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra_context", mode="before")
    @classmethod
    def _normalize_extra_context(cls, value: Any) -> dict[str, Any] | None:
        return _normalize_extra_context(value)

    @property
    def question(self) -> str:
        return self.prompt

    @question.setter
    def question(self, value: str) -> None:
        self.prompt = value

    @property
    def context(self) -> str | None:
        if not self.extra_context:
            return None
        text = self.extra_context.get("text")
        if isinstance(text, str):
            return text
        summary = self.extra_context.get("summary")
        if isinstance(summary, str):
            return summary
        try:
            return json.dumps(self.extra_context, ensure_ascii=False)
        except TypeError:
            return str(self.extra_context)

    @context.setter
    def context(self, value: str | None) -> None:
        self.extra_context = _normalize_extra_context(value)


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
    winner_answer: str | None = None
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
        if self.winner_id and not self.winner_answer:
            answer_map = {answer.member_id: answer.answer for answer in self.answers}
            self.winner_answer = answer_map.get(self.winner_id)
        return self

    def has_consensus(self) -> bool:
        """Return ``True`` if a consensus was declared."""

        return bool(self.winning_member_ids)
