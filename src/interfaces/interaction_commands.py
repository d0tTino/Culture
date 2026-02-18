from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from src.agents.core.agent_state import AgentActionIntent
from src.infra import config
from src.infra import ledger as infra_ledger
from src.infra.event_log import log_event
from src.interfaces.dashboard_backend import SimulationEvent, emit_event
from src.interfaces.interaction_policy import check_cooldown, context_is_authorized
from src.sim.knowledge_board import BoardEntry

if TYPE_CHECKING:
    from src.sim.simulation import Simulation

logger = logging.getLogger(__name__)


class InteractionResult(BaseModel):
    status: Literal["ok", "rejected", "error"]
    user_message: str
    reason_code: str
    data: dict[str, Any] | None = None


class InteractionContext(BaseModel):
    sender_id: str = "human"
    channel_id: str | None = None
    source: str = "unknown"
    permissions: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BroadcastCommand(BaseModel):
    command_type: Literal["broadcast"] = "broadcast"
    content: str
    budget_agent_id: str | None = None
    recipient_id: str | None = None
    target_agent_id: str | None = None


class DirectMessageCommand(BaseModel):
    command_type: Literal["direct_message"] = "direct_message"
    content: str
    recipient_id: str | None = None
    target_agent_id: str | None = None
    budget_agent_id: str | None = None


class SpawnAgentCommand(BaseModel):
    command_type: Literal["spawn"] = "spawn"
    agent_id: str
    role: str | dict[str, Any] | None = None
    persona: str | None = None
    backstory: str | None = None
    traits: dict[str, float] | None = None


class ModerationCommand(BaseModel):
    command_type: Literal["moderation"] = "moderation"
    action: str
    value: float | None = None
    tags: list[str] | None = None
    prompt: str | None = None
    text: str | None = None
    agent_id: str | None = None


class KnowledgeBoardCommand(BaseModel):
    command_type: Literal["knowledge_board"] = "knowledge_board"
    content: str


InteractionCommand = (
    BroadcastCommand
    | DirectMessageCommand
    | SpawnAgentCommand
    | ModerationCommand
    | KnowledgeBoardCommand
)


class InteractionService:
    """Validates, authorizes, and dispatches user interaction commands."""

    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation

    async def execute(
        self,
        command: InteractionCommand,
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        ctx = context or InteractionContext()
        await emit_event(
            SimulationEvent(
                type="human_command",
                data={
                    "command_type": command.command_type,
                    "sender_id": ctx.sender_id,
                    "source": ctx.source,
                    "step": self.simulation.current_step,
                },
            )
        )

        if isinstance(command, KnowledgeBoardCommand):
            return await self._dispatch_knowledge_board(command, context=ctx)
        if isinstance(command, (BroadcastCommand, DirectMessageCommand)):
            return await self._dispatch_message(command, context=ctx)
        if isinstance(command, SpawnAgentCommand):
            if not context_is_authorized(ctx, required={"admin", "moderator"}):
                return InteractionResult(
                    status="rejected",
                    user_message="You are not authorized to spawn agents.",
                    reason_code="unauthorized",
                )
            await self.simulation.handle_control_command(
                {
                    "command": "spawn",
                    "agent_id": command.agent_id,
                    "role": command.role,
                    "persona": command.persona,
                    "backstory": command.backstory,
                    "traits": command.traits,
                }
            )
            return InteractionResult(
                status="ok",
                user_message=f"Spawn request submitted for {command.agent_id}.",
                reason_code="spawn_submitted",
            )
        if isinstance(command, ModerationCommand):
            if not context_is_authorized(ctx, required={"admin", "moderator"}):
                return InteractionResult(
                    status="rejected",
                    user_message="You are not authorized to run moderation commands.",
                    reason_code="unauthorized",
                )
            payload: dict[str, Any] = {"command": command.action}
            if command.value is not None:
                payload["value"] = command.value
            if command.tags is not None:
                payload["tags"] = command.tags
            if command.prompt is not None:
                payload["prompt"] = command.prompt
            if command.text is not None:
                payload["text"] = command.text
            if command.agent_id is not None:
                payload["agent_id"] = command.agent_id
            if command.action == "inject_event" and command.prompt is not None:
                payload["scope"] = command.prompt
            if command.action == "inject_event" and command.agent_id is not None:
                payload["author"] = command.agent_id
            state = await self.simulation.handle_control_command(payload)
            return InteractionResult(
                status="ok",
                user_message="Command accepted.",
                reason_code="moderation_applied",
                data=state if isinstance(state, dict) else None,
            )
        return InteractionResult(
            status="error",
            user_message="Unknown command.",
            reason_code="unsupported_command",
        )

    async def _dispatch_knowledge_board(
        self,
        command: KnowledgeBoardCommand,
        *,
        context: InteractionContext,
    ) -> InteractionResult:
        content = command.content.strip()
        if not content:
            return InteractionResult(
                status="rejected",
                user_message="Knowledge Board entry cannot be empty.",
                reason_code="empty_message",
            )
        board = self.simulation.knowledge_board
        if board is None:
            return InteractionResult(
                status="rejected",
                user_message="Knowledge Board is not available.",
                reason_code="kb_unavailable",
            )
        now = time.monotonic()
        retry_after = check_cooldown(
            key="knowledge_board",
            now=now,
            cooldown=self.simulation._kb_cooldown,
            state={"knowledge_board": self.simulation._last_kb_time},
        )
        if retry_after is not None:
            return InteractionResult(
                status="rejected",
                user_message="Knowledge Board is cooling down. Please try again shortly.",
                reason_code="kb_rate_limited",
            )

        self.simulation._last_kb_time = now
        async with board.lock:
            board.add_entry(
                BoardEntry(
                    content_full=content,
                    entry_type="human_message",
                    tags=["human", "knowledge_board"],
                ),
                context.sender_id,
                self.simulation.current_step,
                self.simulation.vector.to_dict(),
            )
        return InteractionResult(
            status="ok",
            user_message="Posted to Knowledge Board.",
            reason_code="kb_posted",
        )

    async def _dispatch_message(
        self,
        command: BroadcastCommand | DirectMessageCommand,
        *,
        context: InteractionContext,
    ) -> InteractionResult:
        text = command.content.strip()
        if not text:
            return InteractionResult(
                status="rejected",
                user_message="Message cannot be empty.",
                reason_code="empty_message",
            )

        now = time.monotonic()
        channel_id = context.channel_id
        relay_scope = (
            context.sender_id if channel_id is None else f"{context.sender_id}:{channel_id}"
        )
        retry_after = check_cooldown(
            key=relay_scope,
            now=now,
            cooldown=self.simulation._relay_cooldown,
            state=self.simulation._last_relay_times,
        )
        if retry_after is not None:
            await emit_event(
                SimulationEvent(
                    type="human_command_rate_limited",
                    data={
                        "sender_id": context.sender_id,
                        "scope": relay_scope,
                        "retry_after_seconds": retry_after,
                        "step": self.simulation.current_step,
                    },
                )
            )
            return InteractionResult(
                status="rejected",
                user_message=(
                    f"Rate limited: please wait {retry_after:.1f}s before sending another message."
                ),
                reason_code="rate_limited",
                data={"retry_after_seconds": retry_after},
            )
        if not self.simulation.agents:
            return InteractionResult(
                status="rejected",
                user_message="No agents are available to receive messages.",
                reason_code="no_agents",
            )

        broadcast = isinstance(command, BroadcastCommand)
        target = self._resolve_target(command)
        budget_agent_id = self._resolve_budget_agent_id(command, target.agent_id)
        budget_agent = next(
            (agent for agent in self.simulation.agents if agent.agent_id == budget_agent_id),
            None,
        )
        state = budget_agent.state if budget_agent is not None else None

        if broadcast:
            ip_cost = float(
                config.get_config("IP_COST_BROADCAST_MESSAGE")
                or config.get_config("IP_COST_SEND_DIRECT_MESSAGE")
                or 0.0
            )
            du_cost = float(
                config.get_config("DU_COST_BROADCAST_ACTION")
                or config.get_config("DU_COST_PER_ACTION")
                or 0.0
            )
        else:
            ip_cost = float(config.get_config("IP_COST_SEND_DIRECT_MESSAGE") or 0.0)
            du_cost = float(config.get_config("DU_COST_PER_ACTION") or 0.0)

        try:
            from src.sim import simulation as simulation_module

            simulation_module.get_resource_manager().ensure_du_budget(budget_agent_id, du_cost)
        except Exception as exc:  # pragma: no cover - defensive
            logger.info("Rejecting interaction for %s: %s", budget_agent_id, exc)
            return InteractionResult(
                status="rejected",
                user_message=str(exc),
                reason_code="du_budget_exceeded",
            )

        if state is not None and (state.ip < ip_cost or state.du < du_cost):
            return InteractionResult(
                status="rejected",
                user_message="Insufficient IP/DU",
                reason_code="insufficient_resources",
            )

        if state is not None:
            state.ip -= ip_cost
            state.du -= du_cost

        try:
            await infra_ledger.ledger.spend(
                budget_agent_id,
                ip=ip_cost,
                du=du_cost,
                reason="human_broadcast" if broadcast else "human_dm",
            )
        except Exception:  # pragma: no cover - optional
            logger.debug("Ledger spend failed", exc_info=True)

        world_time = self.simulation._world_time_snapshot()
        turn_index = self.simulation.current_step
        msgs: list[dict[str, Any]] = []
        if broadcast:
            for agent in self.simulation.agents:
                msgs.append(
                    {
                        "step": self.simulation.current_step,
                        "turn_index": turn_index,
                        "world_time": world_time,
                        "sender_id": context.sender_id,
                        "recipient_id": agent.agent_id,
                        "content": text,
                        "action_intent": AgentActionIntent.SEND_DIRECT_MESSAGE.value,
                        "sentiment_score": None,
                    }
                )
        else:
            msgs.append(
                {
                    "step": self.simulation.current_step,
                    "turn_index": turn_index,
                    "world_time": world_time,
                    "sender_id": context.sender_id,
                    "recipient_id": target.agent_id,
                    "content": text,
                    "action_intent": AgentActionIntent.SEND_DIRECT_MESSAGE.value,
                    "sentiment_score": None,
                }
            )

        async with self.simulation._msg_lock:
            self.simulation.pending_messages_for_next_round.extend(msgs)
            self.simulation.messages_to_perceive_this_round.extend(msgs)

        log_event(
            {
                "type": "human_command",
                "step": self.simulation.current_step,
                "turn_index": turn_index,
                "world_time": world_time,
                "tick": world_time.get("world_tick", 0),
                "sender_id": context.sender_id,
                "target_agent_id": target.agent_id,
                "budget_agent_id": budget_agent_id,
                "broadcast": broadcast,
                "recipient_id": command.recipient_id,
                "text": text,
                "ip_cost": ip_cost,
                "du_cost": du_cost,
                "messages": [dict(msg) for msg in msgs],
                "reason_code": "message_dispatched",
            }
        )

        if self.simulation.discord_bot and self.simulation.discord_bot.last_channel_id is not None:
            chan = self.simulation.discord_bot.last_channel_id
            self.simulation.discord_bot.channel_map[target.agent_id] = chan
            self.simulation.discord_bot.channel_to_agent[chan] = target.agent_id

        return InteractionResult(
            status="ok",
            user_message="Broadcast sent." if broadcast else "Message sent.",
            reason_code="message_dispatched",
            data={"target_agent_id": target.agent_id, "budget_agent_id": budget_agent_id},
        )

    def _resolve_target(self, command: BroadcastCommand | DirectMessageCommand) -> Any:
        target: Any | None = None
        if command.target_agent_id:
            target = next(
                (
                    agent
                    for agent in self.simulation.agents
                    if agent.agent_id == command.target_agent_id
                ),
                None,
            )
        if target is None and command.recipient_id:
            target = next(
                (
                    agent
                    for agent in self.simulation.agents
                    if agent.agent_id == command.recipient_id
                ),
                None,
            )
        if (
            target is None
            and self.simulation.discord_bot
            and self.simulation.discord_bot.last_agent_id
        ):
            last_id = self.simulation.discord_bot.last_agent_id
            target = next(
                (agent for agent in self.simulation.agents if agent.agent_id == last_id), None
            )
        if target is None:
            target = self.simulation.agents[self.simulation.current_agent_index]
        return target

    def _resolve_budget_agent_id(
        self,
        command: BroadcastCommand | DirectMessageCommand,
        fallback: str,
    ) -> str:
        configured_budget_id = config.get_config("HUMAN_COMMAND_BUDGET_AGENT_ID")
        if command.budget_agent_id:
            return command.budget_agent_id
        if isinstance(configured_budget_id, str) and configured_budget_id:
            return configured_budget_id
        return fallback

    async def execute_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        cmd_type = str(payload.get("command") or payload.get("command_type") or "").strip()
        try:
            if cmd_type == "broadcast":
                command = BroadcastCommand(
                    content=str(payload.get("content", "")),
                    recipient_id=self._optional_str(payload.get("recipient_id")),
                    target_agent_id=self._optional_str(payload.get("target_agent_id")),
                    budget_agent_id=self._optional_str(payload.get("budget_agent_id")),
                )
            elif cmd_type in {"direct_message", "dm"}:
                command = DirectMessageCommand(
                    content=str(payload.get("content", "")),
                    recipient_id=self._optional_str(payload.get("recipient_id")),
                    target_agent_id=self._optional_str(payload.get("target_agent_id")),
                    budget_agent_id=self._optional_str(payload.get("budget_agent_id")),
                )
            elif cmd_type in {"kb", "knowledge_board"}:
                command = KnowledgeBoardCommand(content=str(payload.get("content", "")))
            elif cmd_type == "spawn":
                command = SpawnAgentCommand(
                    agent_id=str(payload.get("agent_id", "")).strip(),
                    role=payload.get("role"),
                    persona=self._optional_str(payload.get("persona")),
                    backstory=self._optional_str(payload.get("backstory")),
                    traits=payload.get("traits"),
                )
            else:
                command = ModerationCommand(
                    action=cmd_type or str(payload.get("action", "")).strip(),
                    value=self._optional_float(payload.get("value")),
                    tags=self._optional_tags(payload.get("tags")),
                    prompt=self._optional_str(payload.get("prompt")),
                    text=self._optional_str(payload.get("text")),
                    agent_id=self._optional_str(payload.get("agent_id")),
                )
        except Exception as exc:
            return InteractionResult(
                status="rejected",
                user_message=f"Invalid command payload: {exc}",
                reason_code="invalid_payload",
            )
        return await self.execute(command, context=context)

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _optional_tags(value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [str(tag) for tag in value]
        return [str(value)]
