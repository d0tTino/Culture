from __future__ import annotations

import logging
from typing import Any

from src.interfaces.metrics import STEP_PHASE_QUEUE_DEPTH
from src.shared.telemetry import trace_agent_action
from src.sim.runtime.step_context import StepContext

logger = logging.getLogger(__name__)


class PerceptionPhase:
    """Prepare runtime listener and pre-step metrics."""

    async def execute(self, simulation: Any, context: StepContext) -> None:
        context.phase_order.append("perception")
        await simulation.start_event_listener()
        context.queue_depth = simulation.event_kernel.queue_depth()
        simulation._set_labeled_gauge(
            STEP_PHASE_QUEUE_DEPTH,
            phase="queue_pre_step",
            value=context.queue_depth,
        )


class DecisionPhase:
    """Run optional parallel planning pipeline."""

    async def execute(self, simulation: Any, context: StepContext) -> None:
        context.phase_order.append("decision")
        if context.max_turns <= 1:
            return
        context.planned_outputs = await simulation._run_step_pipeline(max_turns=context.max_turns)


class ActionPhase:
    """Execute scheduler step or bootstrap immediate event when needed."""

    async def execute(self, simulation: Any, context: StepContext) -> None:
        context.phase_order.append("action")
        if context.planned_outputs:
            return

        if simulation.event_kernel.empty():
            simulation.vector.increment(simulation.agents[simulation.current_agent_index].get_id())
            simulation.event_kernel.schedule_immediate_nowait(
                simulation._create_agent_event(simulation.current_agent_index),
                agent_id=simulation.agents[simulation.current_agent_index].get_id(),
                vector=simulation.vector,
            )

        agent_id = simulation.agents[simulation.current_agent_index].get_id()
        with trace_agent_action("tick", agent_id=agent_id, step=simulation.current_step):
            context.events = await simulation.event_kernel.step(context.max_turns)


class PostStepPhase:
    """Run evaluation hooks after events are produced for the step."""

    async def execute(self, simulation: Any, context: StepContext) -> None:
        context.phase_order.append("post_step")
        if context.planned_outputs:
            return
        await simulation.engine.emit_evaluation_events(context.events)
