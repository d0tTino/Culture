from __future__ import annotations

from typing import Annotated, Any, Literal, cast

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

INTERACTION_SCHEMA_NAME = "interaction-envelope"
INTERACTION_SCHEMA_VERSION = "2.0"


class InteractionResult(BaseModel):
    schema_name: Literal["interaction-envelope"] = "interaction-envelope"
    schema_version: Literal["2.0"] = "2.0"
    status: Literal["ok", "rejected", "error"]
    user_message: str
    reason_code: str
    correlation_id: str | None = None
    data: dict[str, Any] | None = None
    decision_provenance: DecisionProvenance


class InteractionContext(BaseModel):
    sender_id: str = "human"
    channel_id: str | None = None
    source: str = "unknown"
    permissions: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InteractionIdentity(BaseModel):
    """Transport-agnostic identity metadata used by interaction policy."""

    principal_id: str = ""
    source: str = "unknown"
    channel_id: str | None = None
    is_admin: bool = False
    attributes: dict[str, Any] = Field(default_factory=dict)


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


class BaseInteractionIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_name: Literal["interaction-envelope"] = "interaction-envelope"
    schema_version: Literal["2.0"] = "2.0"
    correlation_id: str | None = None
    intent: str
    routing: InteractionRouting = Field(default_factory=InteractionRouting)
    auth: InteractionAuthScope = Field(default_factory=InteractionAuthScope)
    budget: InteractionBudgetAttribution = Field(default_factory=InteractionBudgetAttribution)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanMessageIntent(BaseInteractionIntent):
    intent: Literal["human_message"] = "human_message"
    text: str


class DirectMessageIntent(BaseInteractionIntent):
    intent: Literal["direct_message"] = "direct_message"
    text: str


class BroadcastIntent(BaseInteractionIntent):
    intent: Literal["broadcast"] = "broadcast"
    text: str


class KnowledgeBoardIntent(BaseInteractionIntent):
    intent: Literal["knowledge_board"] = "knowledge_board"
    text: str


class SpawnIntent(BaseInteractionIntent):
    intent: Literal["spawn"] = "spawn"
    agent_id: str | None = None
    role: str | dict[str, Any] | None = None
    persona: str | None = None
    backstory: str | None = None
    traits: dict[str, float] | None = None


class ControlIntent(BaseInteractionIntent):
    intent: Literal["control"] = "control"
    action: str
    value: float | None = None
    tags: list[str] | None = None
    agent_id: str | None = None


class ModerationIntent(BaseInteractionIntent):
    intent: Literal["moderation"] = "moderation"
    action: str
    agent_id: str | None = None
    value: float | None = None


class InjectEventIntent(BaseInteractionIntent):
    intent: Literal["inject_event"] = "inject_event"
    text: str
    scope: str = "global"
    agent_id: str | None = None


InteractionIntent = Annotated[
    HumanMessageIntent
    | DirectMessageIntent
    | BroadcastIntent
    | KnowledgeBoardIntent
    | SpawnIntent
    | ControlIntent
    | ModerationIntent
    | InjectEventIntent,
    Field(discriminator="intent"),
]

_INTERACTION_INTENT_ADAPTER: TypeAdapter[InteractionIntent] = TypeAdapter(InteractionIntent)


def parse_interaction_intent(payload: dict[str, Any]) -> InteractionIntent:
    normalized = dict(payload)
    routing = (
        dict(normalized.get("routing") or {})
        if isinstance(normalized.get("routing"), dict)
        else {}
    )
    for key in ("sender_id", "source", "channel_id", "recipient_id", "target_agent_id"):
        if key in normalized and key not in routing:
            routing[key] = normalized[key]
    if routing:
        normalized["routing"] = routing
    return cast(InteractionIntent, _INTERACTION_INTENT_ADAPTER.validate_python(normalized))


# Backward-compatible aliases during transport migration.
BaseInteractionEnvelope = BaseInteractionIntent
HumanMessageEnvelope = HumanMessageIntent
DirectMessageEnvelope = DirectMessageIntent
BroadcastEnvelope = BroadcastIntent
KnowledgeBoardEnvelope = KnowledgeBoardIntent
SpawnEnvelope = SpawnIntent
ControlEnvelope = ControlIntent
ModerationEnvelope = ModerationIntent
InjectEventEnvelope = InjectEventIntent
InteractionEnvelope = InteractionIntent
parse_interaction_envelope = parse_interaction_intent
