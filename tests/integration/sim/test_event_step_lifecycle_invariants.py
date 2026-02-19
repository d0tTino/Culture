from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.sim.simulation import EVENT_STEP_LIFECYCLE_MUST_NOT_CHANGE, Simulation


class DummyAgentState:
    def __init__(self) -> None:
        self.ip = 0.0
        self.du = 0.0
        self.short_term_memory = []
        self.messages_sent_count = 0
        self.last_message_step = None
        self.collective_ip = 0.0
        self.collective_du = 0.0
        self.is_alive = True
        self.inheritance = 0.0
        self.parent_id: str | None = None
        self.genes: dict[str, float] = {}


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._state = DummyAgentState()

    def get_id(self) -> str:
        return self.agent_id

    @property
    def state(self) -> DummyAgentState:
        return self._state


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lifecycle_invariant_run_step_order() -> None:
    sim = Simulation([DummyAgent("A")])
    calls: list[str] = []

    async def _start() -> None:
        calls.append("start")

    async def _step(_limit: int) -> list[object]:
        calls.append("step")
        return []

    sim.start_event_listener = _start  # type: ignore[assignment]
    sim.event_kernel.step = _step  # type: ignore[assignment]

    await sim.run_step()

    assert calls == ["start", "step"]
    assert EVENT_STEP_LIFECYCLE_MUST_NOT_CHANGE["run_step_order"]
    await sim.stop_event_listener()
    sim.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lifecycle_invariant_bootstrap_seed_when_queue_empty() -> None:
    sim = Simulation([DummyAgent("A")])
    sim.event_kernel._queue.clear()
    sim.event_kernel.step = AsyncMock(return_value=[])  # type: ignore[assignment]

    before = sim.vector.to_dict().get("A", 0)
    await sim.run_step()
    after = sim.vector.to_dict().get("A", 0)

    assert after == before + 1
    assert not sim.event_kernel.empty()
    assert EVENT_STEP_LIFECYCLE_MUST_NOT_CHANGE["bootstrap_semantics"]
    await sim.stop_event_listener()
    sim.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lifecycle_invariant_evaluation_event_emits_step(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = Simulation([DummyAgent("A")])
    sim.event_kernel.step = AsyncMock(return_value=[])  # type: ignore[assignment]
    emitted: list[object] = []

    async def _emit(evt: object) -> None:
        emitted.append(evt)

    monkeypatch.setattr("src.sim.engine.emit_event", _emit)
    await sim.run_step()

    evaluation = [evt for evt in emitted if getattr(evt, "type", "") == "evaluation"]
    assert evaluation
    assert "step" in (evaluation[-1].data or {})
    assert EVENT_STEP_LIFECYCLE_MUST_NOT_CHANGE["metrics_event_semantics"]
    await sim.stop_event_listener()
    sim.close()
