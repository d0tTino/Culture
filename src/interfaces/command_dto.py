from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CommandDTO(BaseModel):
    kind: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanMessageDTO(CommandDTO):
    kind: Literal["human_message"] = "human_message"
    content: str
    recipient_id: str | None = None
    target_agent_id: str | None = None


class DirectMessageDTO(CommandDTO):
    kind: Literal["direct_message"] = "direct_message"
    content: str
    recipient_id: str | None = None
    target_agent_id: str | None = None
    budget_agent_id: str | None = None


class BroadcastDTO(CommandDTO):
    kind: Literal["broadcast"] = "broadcast"
    content: str
    budget_agent_id: str | None = None


class KnowledgeBoardDTO(CommandDTO):
    kind: Literal["knowledge_board"] = "knowledge_board"
    content: str


class SpawnAgentDTO(CommandDTO):
    kind: Literal["spawn"] = "spawn"
    agent_id: str | None = None
    role: str | dict[str, Any] | None = None
    persona: str | None = None
    backstory: str | None = None
    traits: dict[str, float] | None = None


class ControlDTO(CommandDTO):
    kind: Literal["control"] = "control"
    action: str
    value: float | None = None
    tags: list[str] | None = None
    agent_id: str | None = None


class ModerationDTO(CommandDTO):
    kind: Literal["moderation"] = "moderation"
    action: str
    agent_id: str | None = None
    value: float | None = None


class InjectEventDTO(CommandDTO):
    kind: Literal["inject_event"] = "inject_event"
    text: str
    scope: str = "global"
    agent_id: str | None = None


CommandDTOT = (
    HumanMessageDTO
    | DirectMessageDTO
    | BroadcastDTO
    | KnowledgeBoardDTO
    | SpawnAgentDTO
    | ControlDTO
    | ModerationDTO
    | InjectEventDTO
)
