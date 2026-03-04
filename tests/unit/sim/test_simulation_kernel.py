from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from src.sim.runtime.step_context import StepContext
from src.sim.simulation_kernel import SimulationKernel


@dataclass
class DummyVector:
    value: int = 0

    def increment(self, _agent_id: str) -> None:
        self.value += 1

    def to_dict(self) -> dict[str, int]:
        return {"A": self.value}


class DummyKernelQueue:
    def __init__(self) -> None:
        self.scheduled = 0

    def queue_depth(self) -> int:
        return 0

    def empty(self) -> bool:
        return True

    def schedule_immediate_nowait(self, *_args: Any, **_kwargs: Any) -> None:
        self.scheduled += 1

    async def step(self, _max_turns: int) -> list[dict[str, Any]]:
        return [{"type": "tick", "step": 1}]


class DummySimulation:
    def __init__(self) -> None:
        self.current_step = 1
        self.current_agent_index = 0
        self.agents = [SimpleNamespace(get_id=lambda: "A")]
        self.vector = DummyVector()
        self.event_kernel = DummyKernelQueue()
        self.environment_state = SimpleNamespace(
            weather="clear",
            council_window_active=False,
            world_day=0,
            active_global_modifiers=[],
        )
        self.engine = SimpleNamespace(emit_evaluation_events=self._emit_eval)
        self._evaluated = False

    async def _emit_eval(self, _events: list[dict[str, Any]]) -> None:
        self._evaluated = True

    async def start_event_listener(self) -> None:
        return None

    async def _run_step_pipeline(self, max_turns: int) -> list[dict[str, Any]]:
        return [{"turns": max_turns}]

    def _build_world_context_projection(self, actor_id: str | None = None) -> Any:
        _ = actor_id
        return SimpleNamespace()

    def _world_time_from_projection(self, _projection: Any) -> dict[str, Any]:
        return {"world_day": 0, "world_hour": 0, "world_tick": 0}

    def _create_agent_event(self, _index: int) -> Any:
        async def _noop() -> None:
            return None

        return _noop

    def _set_labeled_gauge(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kernel_runs_tick_and_populates_context() -> None:
    simulation = DummySimulation()
    kernel = SimulationKernel()
    context = StepContext(max_turns=1)

    count = await kernel.run_tick(simulation, context)

    assert count == 1
    assert context.phase_order == ["perception", "decision", "action", "post_step"]
    assert context.tick_context is not None
    assert simulation._evaluated
