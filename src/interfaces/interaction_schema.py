from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

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
    model_config = ConfigDict(extra="forbid")

    sender_id: str = "human"
    source: str = "unknown"
    channel_id: str | None = None
    recipient_id: str | None = None
    target_agent_id: str | None = None


class InteractionAuthScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    permissions: set[str] = Field(default_factory=set)


class InteractionBudgetAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget_agent_id: str | None = None
    attribution_scope: str = "default"


class BaseInteractionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
    routing: InteractionRouting = Field(default_factory=InteractionRouting)
    auth: InteractionAuthScope = Field(default_factory=InteractionAuthScope)
    budget: InteractionBudgetAttribution = Field(default_factory=InteractionBudgetAttribution)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanMessageEnvelope(BaseInteractionEnvelope):
    intent: Literal["human_message"] = "human_message"
    text: str


class DirectMessageEnvelope(BaseInteractionEnvelope):
    intent: Literal["direct_message"] = "direct_message"
    text: str


class BroadcastEnvelope(BaseInteractionEnvelope):
    intent: Literal["broadcast"] = "broadcast"
    text: str


class KnowledgeBoardEnvelope(BaseInteractionEnvelope):
    intent: Literal["knowledge_board"] = "knowledge_board"
    text: str


class SpawnEnvelope(BaseInteractionEnvelope):
    intent: Literal["spawn"] = "spawn"
    agent_id: str | None = None
    role: str | dict[str, Any] | None = None
    persona: str | None = None
    backstory: str | None = None
    traits: dict[str, float] | None = None


class ControlEnvelope(BaseInteractionEnvelope):
    intent: Literal["control"] = "control"
    action: str
    value: float | None = None
    tags: list[str] | None = None
    agent_id: str | None = None


class ModerationEnvelope(BaseInteractionEnvelope):
    intent: Literal["moderation"] = "moderation"
    action: str
    agent_id: str | None = None
    value: float | None = None


class InjectEventEnvelope(BaseInteractionEnvelope):
    intent: Literal["inject_event"] = "inject_event"
    text: str
    scope: str = "global"
    agent_id: str | None = None


InteractionEnvelope = Annotated[
    HumanMessageEnvelope
    | DirectMessageEnvelope
    | BroadcastEnvelope
    | KnowledgeBoardEnvelope
    | SpawnEnvelope
    | ControlEnvelope
    | ModerationEnvelope
    | InjectEventEnvelope,
    Field(discriminator="intent"),
]

_INTERACTION_ENVELOPE_ADAPTER = TypeAdapter(InteractionEnvelope)


def parse_interaction_envelope(payload: dict[str, Any]) -> InteractionEnvelope:
    return _INTERACTION_ENVELOPE_ADAPTER.validate_python(payload)
