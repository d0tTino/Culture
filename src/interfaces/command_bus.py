from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from src.interfaces.interaction_commands import (
    BroadcastCommand,
    DirectMessageCommand,
    InteractionCommand,
    InteractionContext,
    InteractionResult,
    KnowledgeBoardCommand,
    ModerationCommand,
    SpawnAgentCommand,
)

if TYPE_CHECKING:
    from src.interfaces.interaction_commands import InteractionService


class HumanMessage(BaseModel):
    command_type: Literal["human_message"] = "human_message"
    content: str
    sender_id: str = "human"
    source: str = "unknown"
    channel_id: str | None = None
    permissions: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)
    broadcast: bool = False
    recipient_id: str | None = None
    target_agent_id: str | None = None
    budget_agent_id: str | None = None


class BroadcastRequest(BaseModel):
    command_type: Literal["broadcast"] = "broadcast"
    content: str
    recipient_id: str | None = None
    target_agent_id: str | None = None
    budget_agent_id: str | None = None


class DirectMessageRequest(BaseModel):
    command_type: Literal["direct_message"] = "direct_message"
    content: str
    recipient_id: str | None = None
    target_agent_id: str | None = None
    budget_agent_id: str | None = None


class SpawnAgentRequest(BaseModel):
    command_type: Literal["spawn"] = "spawn"
    agent_id: str
    role: str | dict[str, Any] | None = None
    persona: str | None = None
    backstory: str | None = None
    traits: dict[str, float] | None = None


class InjectEventRequest(BaseModel):
    command_type: Literal["inject_event"] = "inject_event"
    text: str
    author: str = "human"
    scope: str = "global"


class ModerationAction(BaseModel):
    command_type: Literal["moderation"] = "moderation"
    action: str
    value: float | None = None
    tags: list[str] | None = None
    prompt: str | None = None
    text: str | None = None
    agent_id: str | None = None


class ControlRequest(BaseModel):
    command_type: Literal["control"] = "control"
    action: Literal["pause", "resume", "pause_all", "start", "stop", "set_speed", "kill_agent"]
    value: float | None = None
    agent_id: str | None = None


BusCommand = (
    HumanMessage
    | BroadcastRequest
    | DirectMessageRequest
    | SpawnAgentRequest
    | InjectEventRequest
    | ModerationAction
    | ControlRequest
)


class CommandBus:
    def __init__(self, interaction_service: InteractionService) -> None:
        self.interaction_service = interaction_service

    async def dispatch(
        self,
        command: BusCommand,
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        if isinstance(command, HumanMessage):
            ctx = InteractionContext(
                sender_id=command.sender_id,
                channel_id=command.channel_id,
                source=command.source,
                permissions=command.permissions,
                metadata=command.metadata,
            )
            message = command.content.strip()
            if message.startswith("/kb "):
                return await self.interaction_service.execute(
                    KnowledgeBoardCommand(content=message[4:]),
                    context=ctx,
                )
            if command.broadcast or message.startswith("/broadcast"):
                content = (
                    message[len("/broadcast ") :] if message.startswith("/broadcast ") else ""
                )
                req = BroadcastRequest(
                    content=content if message.startswith("/broadcast") else message,
                    recipient_id=command.recipient_id,
                    target_agent_id=command.target_agent_id,
                    budget_agent_id=command.budget_agent_id,
                )
                return await self.dispatch(req, context=ctx)
            return await self.dispatch(
                DirectMessageRequest(
                    content=message,
                    recipient_id=command.recipient_id,
                    target_agent_id=command.target_agent_id,
                    budget_agent_id=command.budget_agent_id,
                ),
                context=ctx,
            )

        typed_command: InteractionCommand
        if isinstance(command, BroadcastRequest):
            typed_command = BroadcastCommand(**command.model_dump(exclude={"command_type"}))
        elif isinstance(command, DirectMessageRequest):
            typed_command = DirectMessageCommand(**command.model_dump(exclude={"command_type"}))
        elif isinstance(command, SpawnAgentRequest):
            typed_command = SpawnAgentCommand(**command.model_dump(exclude={"command_type"}))
        elif isinstance(command, InjectEventRequest):
            typed_command = ModerationCommand(
                action="inject_event",
                text=command.text,
                prompt=command.scope,
                agent_id=command.author,
            )
        elif isinstance(command, ModerationAction):
            typed_command = ModerationCommand(**command.model_dump(exclude={"command_type"}))
        elif isinstance(command, ControlRequest):
            typed_command = ModerationCommand(
                action=command.action,
                value=command.value,
                agent_id=command.agent_id,
            )
        else:
            return InteractionResult(
                status="error",
                user_message="Unknown command.",
                reason_code="unsupported_command",
            )
        return await self.interaction_service.execute(typed_command, context=context)

    async def dispatch_payload(
        self,
        payload: Mapping[str, Any],
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        command = parse_bus_command(payload)
        return await self.dispatch(command, context=context)


def parse_bus_command(payload: Mapping[str, Any]) -> BusCommand:
    cmd_type = str(payload.get("command_type") or payload.get("command") or "").strip()
    if cmd_type in {"human_message"}:
        return HumanMessage(**dict(payload))
    if cmd_type == "broadcast":
        return BroadcastRequest(**dict(payload))
    if cmd_type in {"direct_message", "dm"}:
        data = dict(payload)
        data["command_type"] = "direct_message"
        return DirectMessageRequest(**data)
    if cmd_type in {"spawn"}:
        return SpawnAgentRequest(**dict(payload))
    if cmd_type in {"inject_event"}:
        data = dict(payload)
        data["command_type"] = "inject_event"
        if "author" not in data and "agent_id" in data:
            data["author"] = data["agent_id"]
        if "text" not in data and "content" in data:
            data["text"] = data["content"]
        return InjectEventRequest(**data)
    if cmd_type in {"pause", "resume", "pause_all", "start", "stop", "set_speed", "kill_agent"}:
        data = {
            "command_type": "control",
            "action": cmd_type,
            "value": payload.get("value"),
            "agent_id": payload.get("agent_id"),
        }
        return ControlRequest(**data)
    if cmd_type in {"kb", "knowledge_board"}:
        return HumanMessage(
            content=f"/kb {payload.get('content', '')!s}",
            sender_id=str(payload.get("sender_id", "human")),
            source=str(payload.get("source", "unknown")),
        )

    return ModerationAction(
        action=cmd_type or str(payload.get("action", "")).strip(),
        value=float(payload["value"]) if payload.get("value") is not None else None,
        tags=[str(tag) for tag in payload.get("tags", [])]
        if isinstance(payload.get("tags"), list)
        else None,
        prompt=str(payload["prompt"]) if payload.get("prompt") is not None else None,
        text=str(payload["text"]) if payload.get("text") is not None else None,
        agent_id=str(payload["agent_id"]) if payload.get("agent_id") is not None else None,
    )
