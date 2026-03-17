from __future__ import annotations

from src.interfaces.command_dto import (
    BroadcastDTO,
    ControlDTO,
    DirectMessageDTO,
    HumanMessageDTO,
    InjectEventDTO,
    KnowledgeBoardDTO,
    ModerationDTO,
    SpawnAgentDTO,
)
from src.interfaces.interaction_schema import (
    BroadcastEnvelope,
    ControlEnvelope,
    DirectMessageEnvelope,
    HumanMessageEnvelope,
    InjectEventEnvelope,
    InteractionContext,
    InteractionEnvelope,
    KnowledgeBoardEnvelope,
    ModerationEnvelope,
    SpawnEnvelope,
)


class DomainCommand:
    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        raise NotImplementedError


class HumanMessageCommand(HumanMessageDTO, DomainCommand):

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return HumanMessageEnvelope(
            text=self.content,
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


class DirectMessageCommand(DirectMessageDTO, DomainCommand):

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return DirectMessageEnvelope(
            text=self.content,
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


class BroadcastCommand(BroadcastDTO, DomainCommand):

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return BroadcastEnvelope(
            text=self.content,
            routing={
                "sender_id": context.sender_id,
                "source": context.source,
                "channel_id": context.channel_id,
            },
            auth={"permissions": set(context.permissions)},
            budget={"budget_agent_id": self.budget_agent_id},
            metadata=self.metadata,
        )


class KnowledgeBoardCommand(KnowledgeBoardDTO, DomainCommand):

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return KnowledgeBoardEnvelope(
            text=self.content,
            routing={"sender_id": context.sender_id, "source": context.source},
            auth={"permissions": set(context.permissions)},
            metadata=self.metadata,
        )


class SpawnAgentCommand(SpawnAgentDTO, DomainCommand):

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return SpawnEnvelope(
            agent_id=self.agent_id,
            role=self.role,
            persona=self.persona,
            backstory=self.backstory,
            traits=self.traits,
            routing={"sender_id": context.sender_id, "source": context.source},
            auth={"permissions": set(context.permissions)},
            metadata=self.metadata,
        )


class ControlCommand(ControlDTO, DomainCommand):

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return ControlEnvelope(
            action=self.action,
            value=self.value,
            tags=self.tags,
            agent_id=self.agent_id,
            routing={"sender_id": context.sender_id, "source": context.source},
            auth={"permissions": set(context.permissions)},
            metadata=self.metadata,
        )


class ModerationCommand(ModerationDTO, DomainCommand):

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return ModerationEnvelope(
            action=self.action,
            agent_id=self.agent_id,
            value=self.value,
            routing={"sender_id": context.sender_id, "source": context.source},
            auth={"permissions": set(context.permissions)},
            metadata=self.metadata,
        )


class InjectEventCommand(InjectEventDTO, DomainCommand):

    def to_envelope(self, *, context: InteractionContext) -> InteractionEnvelope:
        return InjectEventEnvelope(
            text=self.text,
            scope=self.scope,
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
