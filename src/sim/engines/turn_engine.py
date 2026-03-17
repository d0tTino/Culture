from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any, cast

from src.agents.core.agent_state import AgentActionIntent
from src.interfaces.metrics import STEP_PHASE_LATENCY_MS, STEP_PHASE_QUEUE_DEPTH
from src.shared.telemetry import trace_agent_action
from src.sim.contracts.tick_context import TickContext
from src.sim.engines.domain_events import DomainEvent
from src.sim.runtime.step_context import StepContext


class TurnEngine:
    """Coordinates agent planning and deterministic commit boundaries."""

    async def plan(
        self, simulation: Any, context: StepContext, tick: TickContext
    ) -> list[DomainEvent]:
        if context.max_turns <= 1:
            return []

        batch_size = min(context.max_turns, len(simulation.agents))
        base_step = simulation.current_step + 1
        tick_snapshot = simulation._build_tick_read_snapshot(turn_index=base_step)

        phase_start = time.perf_counter()
        plans: list[dict[str, Any]] = []
        for idx in range(batch_size):
            agent_index = (simulation.current_agent_index + idx) % len(simulation.agents)
            agent = simulation.agents[agent_index]
            plans.append(
                {
                    "batch_index": idx,
                    "agent_index": agent_index,
                    "agent_id": agent.agent_id,
                    "simulation_step": base_step + idx,
                    "resource": "agent_turn",
                    "target": agent.agent_id,
                    "tick_snapshot": tick_snapshot,
                    "snapshot": simulation._build_step_perception_snapshot(base_step + idx),
                }
            )
        simulation._set_labeled_gauge(
            STEP_PHASE_LATENCY_MS,
            phase="perception_snapshot",
            value=(time.perf_counter() - phase_start) * 1000,
        )
        simulation._set_labeled_gauge(
            STEP_PHASE_QUEUE_DEPTH,
            phase="planning_batch_size",
            value=len(plans),
        )

        async def _run_plan(plan: Mapping[str, Any]) -> Mapping[str, Any]:
            return await simulation.agents[int(plan["agent_index"])].run_turn(
                simulation_step=int(plan["simulation_step"]),
                environment_perception=dict(cast(Mapping[str, Any], plan["snapshot"])),
                memory_service=simulation.memory_service,
                vector_store_manager=simulation.vector_store_manager,
                knowledge_board=simulation.knowledge_board,
            )

        phase_start = time.perf_counter()
        planning_results = await asyncio.gather(*[_run_plan(plan) for plan in plans])
        simulation._set_labeled_gauge(
            STEP_PHASE_LATENCY_MS,
            phase="concurrent_planning",
            value=(time.perf_counter() - phase_start) * 1000,
        )

        planned: list[dict[str, Any]] = []
        for plan, output in zip(plans, planning_results, strict=False):
            if not isinstance(output, Mapping):
                continue
            action_intent = str(output.get("action_intent", AgentActionIntent.IDLE.value))
            intent = {
                "batch_index": int(plan["batch_index"]),
                "agent_index": int(plan["agent_index"]),
                "agent_id": str(plan["agent_id"]),
                "simulation_step": int(plan["simulation_step"]),
                "action_intent": action_intent,
                "requested_action_intent": action_intent,
                "message_content": output.get("message_content"),
                "message_recipient_id": output.get("message_recipient_id"),
                "map_action": output.get("map_action"),
                "resource": simulation._action_resource_key(
                    {
                        "agent_id": plan["agent_id"],
                        "action_intent": action_intent,
                        "map_action": output.get("map_action"),
                        "message_recipient_id": output.get("message_recipient_id"),
                    }
                ),
                "target": str(output.get("target", plan["target"])),
                "metadata": {
                    "governance_sensitive": simulation._is_governance_sensitive(
                        {"action_intent": action_intent}
                    ),
                    "tick_snapshot_turn": tick_snapshot["turn_index"],
                },
            }
            intent["ordering_key"] = list(simulation._deterministic_commit_sort_key(intent))
            planned.append(intent)

        return [
            DomainEvent(
                domain="turn",
                name="planned_turns_ready",
                payload={
                    "planned_outputs": planned,
                    "tick_step": tick.step,
                },
            )
        ]

    async def commit(
        self, simulation: Any, context: StepContext, tick: TickContext
    ) -> list[DomainEvent]:
        if context.planned_outputs:
            phase_start = time.perf_counter()
            intents = [dict(intent) for intent in context.planned_outputs]
            accepted, rejected = simulation._resolve_tick_conflicts(intents)
            committed = [*accepted, *rejected]
            accepted_turns = 0
            for commit_index, intent in enumerate(committed):
                intent["commit_index"] = commit_index
                if intent.get("merge_outcome") == "accepted":
                    accepted_turns += 1

            simulation._set_labeled_gauge(
                STEP_PHASE_LATENCY_MS,
                phase="deterministic_commit_apply",
                value=(time.perf_counter() - phase_start) * 1000,
            )
            simulation._set_labeled_gauge(
                STEP_PHASE_QUEUE_DEPTH,
                phase="commit_batch_size",
                value=len(committed),
            )
            return [
                DomainEvent(
                    domain="turn",
                    name="planned_turns_committed",
                    payload={
                        "committed_outputs": committed,
                        "tick_step": tick.step,
                        "turn_count": len(intents),
                        "accepted_turn_count": accepted_turns,
                    },
                )
            ]
        agent_id = simulation.agents[simulation.current_agent_index].get_id()
        with trace_agent_action("tick", agent_id=agent_id, step=tick.step):
            events = await simulation.event_kernel.step(context.max_turns)
        return [
            DomainEvent(
                domain="turn",
                name="scheduler_events_ready",
                payload={"events": events},
            )
        ]

    async def prepare_commit(
        self, simulation: Any, context: StepContext, tick: TickContext
    ) -> list[DomainEvent]:
        if context.planned_outputs or not simulation.event_kernel.empty():
            return []

        agent_id = simulation.agents[simulation.current_agent_index].get_id()
        return [
            DomainEvent(
                domain="turn",
                name="bootstrap_agent_event_requested",
                payload={
                    "agent_index": simulation.current_agent_index,
                    "agent_id": agent_id,
                    "tick_step": tick.step,
                },
            )
        ]
