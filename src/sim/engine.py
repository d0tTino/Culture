from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace

from src.infra import config
from src.infra.event_log import log_domain_event_batch, log_event
from src.interfaces.dashboard_backend import SimulationEvent, emit_event
from src.sim.kernel.simulation_kernel import SimulationKernel
from src.sim.persistence.trace_hash_service import TraceHashService
from src.sim.runtime import StepContext

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class SimulationEngine:
    """Deterministic event-step engine for simulation progression/state mutation."""

    def __init__(self, simulation: Any) -> None:
        self.simulation = simulation
        self.kernel = SimulationKernel()
        self.last_step_context: StepContext | None = None

    async def run_step(self, max_turns: int = 1) -> int:
        sim = self.simulation
        if not sim.agents:
            logger.warning("No agents in simulation to run.")
            return 0

        context = StepContext(max_turns=max_turns)
        result = await self.kernel.run_tick(sim, context)
        batch_events = context.planned_outputs if context.planned_outputs else context.events
        canonical_batch_events: list[dict[str, Any]] = []
        for event in batch_events:
            if isinstance(event, dict):
                canonical_batch_events.append(event)
                continue
            if hasattr(event, "step") and hasattr(event, "count"):
                canonical_batch_events.append(sim.event_kernel.event_metadata(event))
                continue
            canonical_batch_events.append({"event": str(event)})
        context.persisted_batch = log_domain_event_batch(
            step=sim.current_step,
            events=canonical_batch_events,
            phase_order=context.phase_order,
        )
        if not context.planned_outputs:
            await self.emit_evaluation_events(context.events)
        interval = int(config.SNAPSHOT_INTERVAL_STEPS)
        if interval > 0 and sim.current_step % interval == 0:
            await sim.persist_snapshot(reason="periodic")
        self.last_step_context = context
        return result

    async def emit_evaluation_events(self, events: list[dict[str, Any]]) -> None:
        sim = self.simulation
        metrics: dict[str, Any] = {}
        for hook in sim.evaluation_hooks:
            try:
                with tracer.start_as_current_span("simulation.evaluation_hook") as span:
                    span.set_attribute("hook.name", getattr(hook, "__name__", repr(hook)))
                    span.set_attribute("simulation.step", sim.current_step)
                    try:
                        result = hook(sim, events) or {}
                    except TypeError:
                        result = hook(events) or {}
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
