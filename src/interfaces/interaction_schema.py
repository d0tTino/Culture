from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.governance.decision_kernel import DecisionProvenance

ENVELOPE_INTENTS = {
    "human_message",
    "direct_message",
    "broadcast",
    "knowledge_board",
    "spawn",
    "moderation",
    "control",
    "inject_event",
}


class InteractionResult(BaseModel):
    status: Literal["ok", "rejected", "error"]
    user_message: str
    reason_code: str
    data: dict[str, Any] | None = None
    decision_provenance: DecisionProvenance


class InteractionContext(BaseModel):
    sender_id: str = "human"
    channel_id: str | None = None
    source: str = "unknown"
    permissions: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InteractionRouting(BaseModel):
    sender_id: str = "human"
    source: str = "unknown"
    channel_id: str | None = None
    recipient_id: str | None = None
    target_agent_id: str | None = None


class InteractionAuthScope(BaseModel):
    permissions: set[str] = Field(default_factory=set)


class InteractionBudgetAttribution(BaseModel):
    budget_agent_id: str | None = None
    attribution_scope: str = "default"


class InteractionEnvelope(BaseModel):
    intent: Literal[
        "human_message",
        "direct_message",
        "broadcast",
        "knowledge_board",
        "spawn",
        "moderation",
        "control",
        "inject_event",
    ]
    content: str | None = None
    action: str | None = None
    value: float | None = None
    tags: list[str] | None = None
    prompt: str | None = None
    text: str | None = None
    agent_id: str | None = None
    role: str | dict[str, Any] | None = None
    persona: str | None = None
    backstory: str | None = None
    traits: dict[str, float] | None = None
    routing: InteractionRouting = Field(default_factory=InteractionRouting)
    auth: InteractionAuthScope = Field(default_factory=InteractionAuthScope)
    budget: InteractionBudgetAttribution = Field(default_factory=InteractionBudgetAttribution)
    metadata: dict[str, Any] = Field(default_factory=dict)
