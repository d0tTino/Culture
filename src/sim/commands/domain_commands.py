from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.interfaces.interaction_schema import InteractionContext, InteractionEnvelope


class DomainCommand(BaseModel):
    kind: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        raise NotImplementedError


class HumanMessageCommand(DomainCommand):
    kind: Literal["human_message"] = "human_message"
    content: str
    recipient_id: str | None = None
    target_agent_id: str | None = None

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return InteractionEnvelope(
            intent="human_message",
            content=self.content,
            routing={
                "sender_id": context.sender_id,
                "source": context.source,
                "channel_id": context.channel_id,
                "recipient_id": self.recipient_id,
                "target_agent_id": self.target_agent_id,
            },
            auth={"permissions": set(context.permissions)},
            metadata=self.metadata,
        )


class DirectMessageCommand(DomainCommand):
    kind: Literal["direct_message"] = "direct_message"
    content: str
    recipient_id: str | None = None
    target_agent_id: str | None = None
    budget_agent_id: str | None = None

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return InteractionEnvelope(
            intent="direct_message",
            content=self.content,
            routing={
                "sender_id": context.sender_id,
                "source": context.source,
                "channel_id": context.channel_id,
                "recipient_id": self.recipient_id,
                "target_agent_id": self.target_agent_id,
            },
            auth={"permissions": set(context.permissions)},
            budget={"budget_agent_id": self.budget_agent_id},
            metadata=self.metadata,
        )


class BroadcastCommand(DomainCommand):
    kind: Literal["broadcast"] = "broadcast"
    content: str
    budget_agent_id: str | None = None

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return InteractionEnvelope(
            intent="broadcast",
            content=self.content,
            routing={
                "sender_id": context.sender_id,
                "source": context.source,
                "channel_id": context.channel_id,
            },
            auth={"permissions": set(context.permissions)},
            budget={"budget_agent_id": self.budget_agent_id},
            metadata=self.metadata,
        )


class KnowledgeBoardCommand(DomainCommand):
    kind: Literal["knowledge_board"] = "knowledge_board"
    content: str

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return InteractionEnvelope(
            intent="knowledge_board",
            content=self.content,
            routing={"sender_id": context.sender_id, "source": context.source},
            auth={"permissions": set(context.permissions)},
            metadata=self.metadata,
        )


class SpawnAgentCommand(DomainCommand):
    kind: Literal["spawn"] = "spawn"
    agent_id: str | None = None
    role: str | dict[str, Any] | None = None
    persona: str | None = None
    backstory: str | None = None
    traits: dict[str, float] | None = None

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return InteractionEnvelope(
            intent="spawn",
            action="spawn",
            agent_id=self.agent_id,
            role=self.role,
            persona=self.persona,
            backstory=self.backstory,
            traits=self.traits,
            routing={"sender_id": context.sender_id, "source": context.source},
            auth={"permissions": set(context.permissions)},
            metadata=self.metadata,
        )


class ControlCommand(DomainCommand):
    kind: Literal["control"] = "control"
    action: str
    value: float | None = None
    tags: list[str] | None = None
    agent_id: str | None = None

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return InteractionEnvelope(
            intent="control",
            action=self.action,
            value=self.value,
            tags=self.tags,
            agent_id=self.agent_id,
            routing={"sender_id": context.sender_id, "source": context.source},
            auth={"permissions": set(context.permissions)},
            metadata=self.metadata,
        )


class ModerationCommand(DomainCommand):
    kind: Literal["moderation"] = "moderation"
    action: str
    agent_id: str | None = None
    value: float | None = None

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return InteractionEnvelope(
            intent="moderation",
            action=self.action,
            agent_id=self.agent_id,
            value=self.value,
            routing={"sender_id": context.sender_id, "source": context.source},
            auth={"permissions": set(context.permissions)},
            metadata=self.metadata,
        )


class InjectEventCommand(DomainCommand):
    kind: Literal["inject_event"] = "inject_event"
    text: str
    scope: str = "global"
    agent_id: str | None = None

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return InteractionEnvelope(
            intent="inject_event",
            text=self.text,
            prompt=self.scope,
            agent_id=self.agent_id,
            routing={"sender_id": context.sender_id, "source": context.source},
            auth={"permissions": set(context.permissions)},
            metadata=self.metadata,
        )


DomainCommandT = (
    HumanMessageCommand
    | DirectMessageCommand
    | BroadcastCommand
    | KnowledgeBoardCommand
    | SpawnAgentCommand
    | ControlCommand
    | ModerationCommand
    | InjectEventCommand
)
