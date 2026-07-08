from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.sim.simulation import Simulation


@dataclass
class DummyState:
    ip: float = 0.0
    du: float = 0.0
    mood_level: float = 0.0
    short_term_memory: list[dict[str, Any]] = field(default_factory=list)
    messages_sent_count: int = 0
    last_message_step: int | None = None
    collective_ip: float = 0.0
    collective_du: float = 0.0


class ScriptedAgent:
    def __init__(self, agent_id: str, recipient: str) -> None:
        self.agent_id = agent_id
        self._state = DummyState()
        self._recipient = recipient

    def get_id(self) -> str:
        return self.agent_id

    @property
    def state(self) -> DummyState:
        return self._state

    async def run_turn(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "action_intent": "send_direct_message",
            "message_content": f"golden:{self.agent_id}",
            "message_recipient_id": self._recipient,
            "map_action": {"action": "gather", "x": 2, "y": 2},
        }


def _normalize_trace(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ignored = {"simulation_step", "commit_index", "ordering_key"}
    normalized: list[dict[str, Any]] = []
    for event in events:
        normalized.append({k: v for k, v in event.items() if k not in ignored})
    return normalized


@pytest.mark.unit
@pytest.mark.asyncio
async def test_golden_trace_compat_pipeline_matches_kernel_run_step() -> None:
    compat_sim = Simulation([ScriptedAgent("A", "R"), ScriptedAgent("B", "R")], seed=123)
    kernel_sim = Simulation([ScriptedAgent("A", "R"), ScriptedAgent("B", "R")], seed=123)

    try:
        with pytest.deprecated_call(match="compatibility adapter"):
            compat_events = await compat_sim._run_step_pipeline(max_turns=2)

        _ = await kernel_sim.run_step(max_turns=2)
        kernel_context = kernel_sim.engine.last_step_context
        assert kernel_context is not None
        kernel_events = (
            kernel_context.planned_outputs
            if kernel_context.planned_outputs
            else kernel_context.events
        )

        assert _normalize_trace(compat_events) == _normalize_trace(kernel_events)
    finally:
        await compat_sim.stop_event_listener()
        compat_sim.close()
        await kernel_sim.stop_event_listener()
        kernel_sim.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kernel_phase_boundaries_reject_snapshot_hash_discontinuity() -> None:
    sim = Simulation([ScriptedAgent("A", "R")], seed=123)
    try:
        sim._last_trace_hash = "snapshot-before"
        original_plan = sim.engine.kernel.turn_engine.plan

        async def _mutating_plan(*args: Any, **kwargs: Any) -> Any:
            sim._last_trace_hash = "snapshot-after"
            return await original_plan(*args, **kwargs)

        sim.engine.kernel.turn_engine.plan = _mutating_plan  # type: ignore[method-assign]

        with pytest.raises(AssertionError, match="Snapshot hash continuity"):
            await sim.run_step(max_turns=2)
    finally:
        await sim.stop_event_listener()
        sim.close()
