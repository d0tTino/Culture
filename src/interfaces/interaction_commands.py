from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from src.agents.core.agent_state import AgentActionIntent
from src.governance.decision_kernel import DecisionProvenance, PolicyDecisionService
from src.infra import config
from src.infra import ledger as infra_ledger
from src.infra.event_log import log_event
from src.interfaces.dashboard_backend import SimulationEvent, emit_event
from src.sim.knowledge_board import BoardEntry

if TYPE_CHECKING:
    from src.sim.simulation import Simulation

logger = logging.getLogger(__name__)

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


class InteractionService:
    """Validates, authorizes, and dispatches user interaction commands."""

    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation
        self.decision_service = PolicyDecisionService()

    async def execute(
        self,
        command: InteractionEnvelope,
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        ctx = context or InteractionContext()
        await emit_event(
            SimulationEvent(
                type="human_command",
                data={
                    "command_type": command.intent,
                    "sender_id": ctx.sender_id,
                    "source": ctx.source,
                    "step": self.simulation.current_step,
                },
            )
        )

        entry_decision = self.decision_service.decide(
            envelope=command,
            context=ctx,
            simulation=self.simulation,
            stage="entry",
        )
        if entry_decision.decision == "transform":
            envelope = command.model_copy(update=entry_decision.transformed_updates)
        else:
            envelope = command

        if entry_decision.decision == "deny":
            return InteractionResult(
                status="rejected",
                user_message=entry_decision.user_message or "Command rejected.",
                reason_code=entry_decision.reason_code,
                data=entry_decision.metadata or None,
                decision_provenance=entry_decision.provenance,
            )
        if entry_decision.decision == "requires_vote":
            return InteractionResult(
                status="rejected",
                user_message=entry_decision.user_message or "Command requires vote.",
                reason_code=entry_decision.reason_code,
                decision_provenance=entry_decision.provenance,
            )

        if envelope.intent == "knowledge_board":
            return await self._dispatch_knowledge_board(envelope, context=ctx)
        if envelope.intent in {"broadcast", "direct_message"}:
            return await self._dispatch_message(envelope, context=ctx)
        if envelope.intent == "spawn":
            await self.simulation.handle_control_command(
                {
                    "command": "spawn",
                    "agent_id": envelope.agent_id,
                    "role": envelope.role,
                    "persona": envelope.persona,
                    "backstory": envelope.backstory,
                    "traits": envelope.traits,
                }
            )
            return InteractionResult(
                status="ok",
                user_message=f"Spawn request submitted for {envelope.agent_id}.",
                reason_code="spawn_submitted",
                decision_provenance=entry_decision.provenance,
            )
        if envelope.intent in {"moderation", "control", "inject_event"}:
            payload: dict[str, Any] = {"command": envelope.action}
            if envelope.value is not None:
                payload["value"] = envelope.value
            if envelope.tags is not None:
                payload["tags"] = envelope.tags
            if envelope.prompt is not None:
                payload["prompt"] = envelope.prompt
            if envelope.text is not None:
                payload["text"] = envelope.text
            if envelope.agent_id is not None:
                payload["agent_id"] = envelope.agent_id
            if envelope.action == "inject_event" and envelope.prompt is not None:
                payload["scope"] = envelope.prompt
            if envelope.action == "inject_event" and envelope.agent_id is not None:
                payload["author"] = envelope.agent_id
            state = await self.simulation.handle_control_command(payload)
            return InteractionResult(
                status="ok",
                user_message="Command accepted.",
                reason_code="moderation_applied",
                data=state if isinstance(state, dict) else None,
                decision_provenance=entry_decision.provenance,
            )
        return InteractionResult(
            status="error",
            user_message="Unknown command.",
            reason_code="unsupported_command",
            decision_provenance=entry_decision.provenance,
        )

    async def _dispatch_knowledge_board(
        self,
        command: InteractionEnvelope,
        *,
        context: InteractionContext,
    ) -> InteractionResult:
        content = str(command.content or "").strip()
        if not content:
            return InteractionResult(
                status="rejected",
                user_message="Knowledge Board entry cannot be empty.",
                reason_code="empty_message",
                decision_provenance=DecisionProvenance(
                    policy_id="interaction-policy-v1",
                    rule_id="policy.validation.knowledge_board.empty",
                ),
            )
        board = self.simulation.knowledge_board
        if board is None:
            return InteractionResult(
                status="rejected",
                user_message="Knowledge Board is not available.",
                reason_code="kb_unavailable",
                decision_provenance=DecisionProvenance(
                    policy_id="interaction-policy-v1",
                    rule_id="policy.validation.knowledge_board.unavailable",
                ),
            )
        decision = self.decision_service.decide(
            envelope=command,
            context=context,
            simulation=self.simulation,
            stage="knowledge_board",
        )
        if decision.decision != "allow":
            return InteractionResult(
                status="rejected",
                user_message=decision.user_message or "Knowledge board command rejected.",
                reason_code=decision.reason_code,
                data=decision.metadata or None,
                decision_provenance=decision.provenance,
            )
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
            decision_provenance=decision.provenance,
        )

    async def _dispatch_message(
        self,
        command: InteractionEnvelope,
        *,
        context: InteractionContext,
    ) -> InteractionResult:
        text = str(command.content or "").strip()
        if not text:
            return InteractionResult(
                status="rejected",
                user_message="Message cannot be empty.",
                reason_code="empty_message",
                decision_provenance=DecisionProvenance(
                    policy_id="interaction-policy-v1",
                    rule_id="policy.validation.message.empty",
                ),
            )

        decision = self.decision_service.decide(
            envelope=command,
            context=context,
            simulation=self.simulation,
            stage="message",
        )
        if decision.decision == "requires_vote":
            return InteractionResult(
                status="rejected",
                user_message=decision.user_message or "Command requires vote.",
                reason_code=decision.reason_code,
                decision_provenance=decision.provenance,
            )

        if decision.decision != "allow":
            retry_after = float(decision.metadata.get("retry_after_seconds") or 0.0)
            relay_scope = str(decision.metadata.get("scope") or context.sender_id)
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
                user_message=decision.user_message or "Command rejected.",
                reason_code=decision.reason_code,
                data=decision.metadata or None,
                decision_provenance=decision.provenance,
            )

        broadcast = command.intent == "broadcast"
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
                decision_provenance=decision.provenance,
            )

        if state is not None and (state.ip < ip_cost or state.du < du_cost):
            return InteractionResult(
                status="rejected",
                user_message="Insufficient IP/DU",
                reason_code="insufficient_resources",
                decision_provenance=decision.provenance,
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
                "recipient_id": command.routing.recipient_id,
                "text": text,
                "ip_cost": ip_cost,
                "du_cost": du_cost,
                "messages": [dict(msg) for msg in msgs],
                "reason_code": "message_dispatched",
                "decision_provenance": decision.provenance.model_dump(),
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
            decision_provenance=decision.provenance,
        )

    def _resolve_target(self, command: InteractionEnvelope) -> Any:
        target: Any | None = None
        if command.routing.target_agent_id:
            target = next(
                (
                    agent
                    for agent in self.simulation.agents
                    if agent.agent_id == command.routing.target_agent_id
                ),
                None,
            )
        if target is None and command.routing.recipient_id:
            target = next(
                (
                    agent
                    for agent in self.simulation.agents
                    if agent.agent_id == command.routing.recipient_id
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
        command: InteractionEnvelope,
        fallback: str,
    ) -> str:
        configured_budget_id = config.get_config("HUMAN_COMMAND_BUDGET_AGENT_ID")
        if command.budget.budget_agent_id:
            return command.budget.budget_agent_id
        if isinstance(configured_budget_id, str) and configured_budget_id:
            return configured_budget_id
        return fallback

    async def execute_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        envelope = self._envelope_from_payload(payload, context)
        return await self.execute(envelope, context=context)

    def _envelope_from_payload(
        self,
        payload: Mapping[str, Any],
        context: InteractionContext | None,
    ) -> InteractionEnvelope:
        cmd_type = str(payload.get("command") or payload.get("command_type") or "").strip()
        ctx = context or InteractionContext()
        try:
            intent = cmd_type if cmd_type else "moderation"
            if intent == "dm":
                intent = "direct_message"
            if intent == "kb":
                intent = "knowledge_board"
            if intent in {"pause", "resume", "pause_all", "start", "stop", "set_speed", "kill_agent"}:
                intent = "control"
            if intent == "inject_event":
                intent = "inject_event"
            if intent not in ENVELOPE_INTENTS:
                intent = "moderation"
            return InteractionEnvelope(
                intent=intent,
                content=self._optional_str(payload.get("content")),
                action=(cmd_type or self._optional_str(payload.get("action"))) if intent != "control" else cmd_type,
                value=self._optional_float(payload.get("value")),
                tags=self._optional_tags(payload.get("tags")),
                prompt=self._optional_str(payload.get("prompt")) or self._optional_str(payload.get("scope")),
                text=self._optional_str(payload.get("text")) or self._optional_str(payload.get("content")),
                agent_id=self._optional_str(payload.get("agent_id")) or self._optional_str(payload.get("author")),
                role=payload.get("role"),
                persona=self._optional_str(payload.get("persona")),
                backstory=self._optional_str(payload.get("backstory")),
                traits=payload.get("traits"),
                routing=InteractionRouting(
                    sender_id=str(payload.get("sender_id", ctx.sender_id)),
                    source=str(payload.get("source", ctx.source)),
                    channel_id=self._optional_str(payload.get("channel_id")) or ctx.channel_id,
                    recipient_id=self._optional_str(payload.get("recipient_id")),
                    target_agent_id=self._optional_str(payload.get("target_agent_id")),
                ),
                auth=InteractionAuthScope(
                    permissions=set(payload.get("permissions", []))
                    if isinstance(payload.get("permissions"), list | set | tuple)
                    else set(ctx.permissions),
                ),
                budget=InteractionBudgetAttribution(
                    budget_agent_id=self._optional_str(payload.get("budget_agent_id"))
                ),
                metadata={k: v for k, v in payload.items()},
            )
        except Exception as exc:
            raise ValueError(f"Invalid command payload: {exc}") from exc

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
