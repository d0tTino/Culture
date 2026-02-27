from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from src.agents.core.agent_state import AgentActionIntent, AgentLifecycleState, PersonalityTraits
from src.agents.core.roles import ensure_profile, get_role_trait_template
from src.governance.decision_kernel import DecisionProvenance, PolicyDecisionService
from src.infra import config
from src.infra import ledger as infra_ledger
from src.infra.event_log import log_event
from src.interfaces.command_bus import parse_bus_command
from src.interfaces.dashboard_backend import SimulationEvent, emit_event
from src.interfaces.interaction_schema import (
    InteractionContext,
    InteractionEnvelope,
    InteractionResult,
)
from src.shared.typing import SimulationMessage
from src.sim.knowledge_board import BoardEntry

if TYPE_CHECKING:
    from src.sim.simulation import Simulation

logger = logging.getLogger(__name__)


class SimulationCommandService:
    """Canonical command API used by every transport and internal caller."""

    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation
        self.decision_service = PolicyDecisionService()

    async def execute_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        context: InteractionContext | None = None,
    ) -> InteractionResult:
        try:
            envelope = parse_bus_command(payload, context=context)
        except Exception as exc:
            return InteractionResult(
                status="rejected",
                user_message=f"Invalid command payload: {exc}",
                reason_code="invalid_payload",
                decision_provenance=DecisionProvenance(
                    policy_id="interaction-policy-v1",
                    rule_id="policy.validation.payload",
                ),
            )
        return await self.execute(envelope, context=context)

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
        envelope = (
            command.model_copy(update=entry_decision.transformed_updates)
            if entry_decision.decision == "transform"
            else command
        )

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
        if envelope.intent in {"human_message", "broadcast", "direct_message"}:
            return await self._dispatch_message(envelope, context=ctx)
        if envelope.intent in {"spawn", "moderation", "control", "inject_event"}:
            return await self._dispatch_control(envelope, provenance=entry_decision.provenance)

        return InteractionResult(
            status="error",
            user_message="Unknown command.",
            reason_code="unsupported_command",
            decision_provenance=entry_decision.provenance,
        )

    async def _dispatch_control(
        self,
        envelope: InteractionEnvelope,
        *,
        provenance: DecisionProvenance,
    ) -> InteractionResult:
        state = await self._apply_control_action(envelope)
        return InteractionResult(
            status="ok",
            user_message="Command accepted.",
            reason_code="moderation_applied",
            data=state,
            decision_provenance=provenance,
        )

    async def _apply_control_action(self, envelope: InteractionEnvelope) -> dict[str, Any] | None:
        sim = self.simulation
        action = str(envelope.action or "").strip()
        if envelope.intent == "spawn" and action != "spawn":
            action = "spawn"

        if action == "pause":
            sim.paused = True
        elif action == "resume":
            sim.paused = False
        elif action == "pause_all":
            sim.paused = True
            kernel = getattr(sim, "event_kernel", None)
            if kernel is not None and hasattr(kernel, "pause"):
                try:
                    kernel.pause()
                except Exception:  # pragma: no cover
                    logger.debug("Kernel pause failed", exc_info=True)
        elif action == "start":
            sim.paused = False
        elif action == "stop":
            sim.simulation_complete = True
            await sim.stop_event_listener()
        elif action == "spawn":
            await self._spawn_agent(envelope)
        elif action == "kill_agent":
            if envelope.agent_id:
                agent = next((a for a in sim.agents if a.agent_id == str(envelope.agent_id)), None)
                if agent is not None:
                    await sim.retire_agent(
                        agent,
                        remove_from_simulation=True,
                        lifecycle_state=AgentLifecycleState.DECEASED,
                        reason="kill_agent_command",
                    )
        elif action == "set_speed":
            try:
                sim.speed = float(envelope.value or 1)
            except (TypeError, ValueError):
                pass
        elif action == "post_kb":
            await self._post_knowledge_board(envelope)
        elif action == "inject_event":
            await self._inject_world_event(envelope)

        return {"paused": sim.paused, "speed": sim.speed, "simulation_complete": sim.simulation_complete}

    async def _spawn_agent(self, envelope: InteractionEnvelope) -> None:
        sim = self.simulation
        if not envelope.agent_id:
            return
        agent_id = str(envelope.agent_id)
        if any(agent.agent_id == agent_id for agent in sim.agents):
            await emit_event(
                SimulationEvent(
                    type="spawn_rejected",
                    data={"reason": "duplicate_agent_id", "agent_id": agent_id, "step": sim.current_step},
                )
            )
            return

        from src.agents.core.base_agent import Agent

        role_profile = ensure_profile(envelope.role) if envelope.role is not None else None
        initial_state: dict[str, Any] = {}
        if role_profile is not None:
            initial_state["current_role"] = role_profile
        if isinstance(envelope.traits, dict):
            merged_traits = get_role_trait_template(
                role_profile.name if role_profile is not None else "Innovator"
            )
            for trait_name, raw_value in envelope.traits.items():
                merged_traits[str(trait_name)] = float(raw_value)
            initial_state["traits"] = PersonalityTraits(**merged_traits)

        new_agent = Agent(agent_id=agent_id, name=agent_id, initial_state=initial_state or None)
        await sim.spawn_agent(new_agent)

    async def _post_knowledge_board(self, envelope: InteractionEnvelope) -> None:
        sim = self.simulation
        text = envelope.text
        author = envelope.agent_id or "human"
        if text and sim.knowledge_board:
            async with sim.knowledge_board.lock:
                sim.knowledge_board.add_entry(
                    BoardEntry(content_full=text, entry_type="human_message", tags=["human", "moderation"]),
                    str(author),
                    sim.current_step,
                    sim.vector.to_dict(),
                )
            await emit_event(
                SimulationEvent(
                    type="knowledge_board",
                    data={"agent_id": str(author), "content": text, "step": sim.current_step},
                )
            )

    async def _inject_world_event(self, envelope: InteractionEnvelope) -> None:
        sim = self.simulation
        text = str(envelope.text or "").strip()
        if not text:
            return
        author = str(envelope.agent_id or "human")
        scope = str(envelope.prompt or "global")
        event_payload = {
            "type": "world_event",
            "author": author,
            "step": sim.current_step,
            "timestamp": time.time(),
            "scope": scope,
            "text": text,
        }
        msg: SimulationMessage = {
            "step": sim.current_step,
            "sender_id": author,
            "recipient_id": None,
            "content": f"[World Event] {text}",
            "action_intent": None,
            "sentiment_score": None,
        }
        async with sim._msg_lock:
            sim.pending_messages_for_next_round.append(msg)
            sim.messages_to_perceive_this_round.append(msg)
        await sim.event_kernel.emit_environment_event(event_payload)
        if sim.knowledge_board:
            async with sim.knowledge_board.lock:
                sim.knowledge_board.add_entry(
                    BoardEntry(content_full=text, entry_type="world_event", tags=["event", scope]),
                    author,
                    sim.current_step,
                    sim.vector.to_dict(),
                )

    async def _dispatch_knowledge_board(self, command: InteractionEnvelope, *, context: InteractionContext) -> InteractionResult:
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
                BoardEntry(content_full=content, entry_type="human_message", tags=["human", "knowledge_board"]),
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

    async def _dispatch_message(self, command: InteractionEnvelope, *, context: InteractionContext) -> InteractionResult:
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
        budget_agent = next((a for a in self.simulation.agents if a.agent_id == budget_agent_id), None)
        state = budget_agent.state if budget_agent is not None else None

        if broadcast:
            ip_cost = float(config.get_config("IP_COST_BROADCAST_MESSAGE") or config.get_config("IP_COST_SEND_DIRECT_MESSAGE") or 0.0)
            du_cost = float(config.get_config("DU_COST_BROADCAST_ACTION") or config.get_config("DU_COST_PER_ACTION") or 0.0)
        else:
            ip_cost = float(config.get_config("IP_COST_SEND_DIRECT_MESSAGE") or 0.0)
            du_cost = float(config.get_config("DU_COST_PER_ACTION") or 0.0)

        from src.sim import simulation as simulation_module

        try:
            simulation_module.get_resource_manager().ensure_du_budget(budget_agent_id, du_cost)
        except Exception as exc:
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
        except Exception:
            logger.debug("Ledger spend failed", exc_info=True)

        world_time = self.simulation._world_time_snapshot()
        turn_index = self.simulation.current_step
        recipients = self.simulation.agents if broadcast else [target]
        msgs = [
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
            for agent in recipients
        ]
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
        target = None
        if command.routing.target_agent_id:
            target = next((a for a in self.simulation.agents if a.agent_id == command.routing.target_agent_id), None)
        if target is None and command.routing.recipient_id:
            target = next((a for a in self.simulation.agents if a.agent_id == command.routing.recipient_id), None)
        if target is None and self.simulation.discord_bot and self.simulation.discord_bot.last_agent_id:
            target = next((a for a in self.simulation.agents if a.agent_id == self.simulation.discord_bot.last_agent_id), None)
        return target or self.simulation.agents[self.simulation.current_agent_index]

    def _resolve_budget_agent_id(self, command: InteractionEnvelope, fallback: str) -> str:
        configured_budget_id = config.get_config("HUMAN_COMMAND_BUDGET_AGENT_ID")
        if command.budget.budget_agent_id:
            return command.budget.budget_agent_id
        if isinstance(configured_budget_id, str) and configured_budget_id:
            return configured_budget_id
        return fallback
