from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace

from src.infra.event_log import log_event
from src.interfaces.dashboard_backend import SimulationEvent, emit_event
from src.interfaces.metrics import STEP_PHASE_QUEUE_DEPTH
from src.shared.telemetry import trace_agent_action
from src.sim.persistence.trace_hash_service import TraceHashService

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class SimulationEngine:
    """Deterministic event-step engine for simulation progression/state mutation."""

    def __init__(self, simulation: Any) -> None:
        self.simulation = simulation

    async def run_step(self, max_turns: int = 1) -> int:
        sim = self.simulation
        if not sim.agents:
            logger.warning("No agents in simulation to run.")
            return 0

        await sim.start_event_listener()

        queue_depth = len(getattr(sim.event_kernel, "_queue", []))
        sim._set_labeled_gauge(STEP_PHASE_QUEUE_DEPTH, phase="queue_pre_step", value=queue_depth)

        if max_turns > 1:
            planned = await sim._run_step_pipeline(max_turns=max_turns)
            if planned:
                return len(planned)

        if sim.event_kernel.empty():
            sim.vector.increment(sim.agents[sim.current_agent_index].get_id())
            sim.event_kernel.schedule_immediate_nowait(
                sim._create_agent_event(sim.current_agent_index),
                agent_id=sim.agents[sim.current_agent_index].get_id(),
                vector=sim.vector,
            )

        agent_id = sim.agents[sim.current_agent_index].get_id()
        with trace_agent_action("tick", agent_id=agent_id, step=sim.current_step):
            events = await sim.event_kernel.step(max_turns)
            await self._run_evaluation_hooks(events)
            return len(events)

    async def _run_evaluation_hooks(self, events: list[dict[str, Any]]) -> None:
        sim = self.simulation
        metrics: dict[str, Any] = {}
        for hook in sim.evaluation_hooks:
            try:
                with tracer.start_as_current_span("simulation.evaluation_hook") as span:
                    span.set_attribute("hook.name", getattr(hook, "__name__", repr(hook)))
                    span.set_attribute("simulation.step", sim.current_step)
                    result = hook(sim, events) or {}
                    for key, value in result.items():
                        span.set_attribute(f"metric.{key}", value)
                    metrics.update(result)
            except Exception:
                logger.exception("Evaluation hook failed")
        if not metrics:
            return

        sim.metrics.append({"step": sim.current_step, **metrics})
        eval_event = log_event({"type": "evaluation", "step": sim.current_step, **metrics})
        if eval_event is None:
            eval_event = {"type": "evaluation", "step": sim.current_step, **metrics}
            eval_event["trace_hash"] = TraceHashService.compute(eval_event)
        await emit_event(SimulationEvent(type="evaluation", data=eval_event))
