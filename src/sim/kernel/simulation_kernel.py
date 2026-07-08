from __future__ import annotations

from typing import Any

from src.sim.contracts.phases import KERNEL_PHASE_SEQUENCE
from src.sim.engines.interaction_engine import InteractionEngine
from src.sim.engines.persistence_engine import PersistenceEngine
from src.sim.engines.reducers import DomainEventReducer
from src.sim.engines.society_engine import SocietyEngine
from src.sim.engines.turn_engine import TurnEngine
from src.sim.engines.world_engine import WorldEngine
from src.sim.runtime.step_context import StepContext


class SimulationKernel:
    """Authoritative tick orchestrator across deterministic kernel phases."""

    def __init__(self) -> None:
        self.interaction_engine = InteractionEngine()
        self.turn_engine = TurnEngine()
        self.world_engine = WorldEngine()
        self.society_engine = SocietyEngine()
        self.persistence_engine = PersistenceEngine()
        self.reducer = DomainEventReducer()

    async def run_tick(self, simulation: Any, context: StepContext) -> int:
        """Run one canonical tick through the only supported phase sequence."""

        tick = self.world_engine.build_tick_context(simulation)
        context.tick_context = tick
        baseline = self._phase_boundary_state(simulation, context)
        self._assert_phase_boundary(simulation, context, baseline=baseline, phase="tick_start")

        context.phase_order.append(KERNEL_PHASE_SEQUENCE[0])
        self.reducer.apply(
            simulation, context, await self.interaction_engine.ingress(simulation, tick)
        )
        self._assert_phase_boundary(
            simulation, context, baseline=baseline, phase=KERNEL_PHASE_SEQUENCE[0]
        )

        context.phase_order.append(KERNEL_PHASE_SEQUENCE[1])
        self.reducer.apply(
            simulation, context, await self.turn_engine.plan(simulation, context, tick)
        )
        self._assert_phase_boundary(
            simulation, context, baseline=baseline, phase=KERNEL_PHASE_SEQUENCE[1]
        )

        context.phase_order.append(KERNEL_PHASE_SEQUENCE[2])
        self.reducer.apply(
            simulation,
            context,
            await self.turn_engine.prepare_commit(simulation, context, tick),
        )
        self._assert_phase_boundary(simulation, context, baseline=baseline, phase="pre_commit")
        self.reducer.apply(
            simulation, context, await self.turn_engine.commit(simulation, context, tick)
        )
        self._assert_phase_boundary(
            simulation, context, baseline=baseline, phase=KERNEL_PHASE_SEQUENCE[2]
        )

        context.phase_order.append(KERNEL_PHASE_SEQUENCE[3])
        _ = self.society_engine.snapshot(simulation, tick)
        _ = self.persistence_engine.capture_tick(simulation, tick)
        self._assert_phase_boundary(
            simulation, context, baseline=baseline, phase=KERNEL_PHASE_SEQUENCE[3]
        )

        context.phase_order.append(KERNEL_PHASE_SEQUENCE[4])
        self._assert_phase_boundary(
            simulation, context, baseline=baseline, phase=KERNEL_PHASE_SEQUENCE[4]
        )
        return len(context.planned_outputs) if context.planned_outputs else len(context.events)

    @staticmethod
    def _world_time_tuple(simulation: Any) -> tuple[int, int, int, int]:
        temporal = simulation.world_state.temporal
        return (
            int(getattr(temporal, "world_season", 0) or 0),
            int(getattr(temporal, "world_day", 0) or 0),
            int(getattr(temporal, "world_hour", 0) or 0),
            int(getattr(temporal, "world_tick", 0) or 0),
        )

    @classmethod
    def _phase_boundary_state(cls, simulation: Any, context: StepContext) -> dict[str, Any]:
        return {
            "world_time": cls._world_time_tuple(simulation),
            "trace_hash": getattr(simulation, "_last_trace_hash", ""),
            "planned_count": len(context.planned_outputs),
            "event_count": len(context.events),
        }

    @classmethod
    def _assert_phase_boundary(
        cls,
        simulation: Any,
        context: StepContext,
        *,
        baseline: dict[str, Any],
        phase: str,
    ) -> None:
        """Validate boundary invariants shared by every kernel phase."""

        planned_count = len(context.planned_outputs)
        event_count = len(context.events)
        if planned_count < 0 or event_count < 0:  # Defensive: list lengths cannot be negative.
            raise AssertionError(f"Negative event count at phase {phase}")
        if context.max_turns < 1:
            raise AssertionError(
                f"max_turns must be positive at phase {phase}: {context.max_turns}"
            )

        current_world_time = cls._world_time_tuple(simulation)
        if current_world_time < baseline["world_time"]:
            raise AssertionError(
                f"World time moved backwards at phase {phase}: "
                f"{current_world_time} < {baseline['world_time']}"
            )

        tick_hash = ""
        if context.tick_context is not None:
            tick_hash = str(context.tick_context.replay_metadata.get("trace_hash", ""))
        current_hash = str(getattr(simulation, "_last_trace_hash", ""))
        if tick_hash and current_hash != tick_hash:
            raise AssertionError(
                f"Snapshot hash continuity violated at phase {phase}: "
                f"current={current_hash!r} tick={tick_hash!r}"
            )
