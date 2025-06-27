from collections.abc import Awaitable
from typing import Any, Callable, ClassVar

import pytest

from src.sim.event_kernel import EventKernel
from src.sim.version_vector import VersionVector

pytestmark = pytest.mark.stress


def _make_cb(order: list[int], n: int) -> Callable[[], Awaitable[None]]:
    async def _cb() -> None:
        order.append(n)

    return _cb


@pytest.mark.asyncio
async def test_mass_event_dispatch_deterministic_replay() -> None:
    event_count = 10_000
    kernel = EventKernel()
    order: list[int] = []

    for i in range(event_count):
        vv = VersionVector({"A": i + 1})
        kernel.schedule_nowait(_make_cb(order, i), vector=vv)

    executed = await kernel.dispatch(event_count)
    assert len(executed) == event_count
    assert kernel.empty()
    assert order == list(range(event_count))
    vector_snapshot = kernel.vector.to_dict()

    replay = EventKernel()
    order_replay: list[int] = []
    for i in range(event_count):
        vv = VersionVector({"A": i + 1})
        replay.schedule_nowait(_make_cb(order_replay, i), vector=vv)

    executed_replay = await replay.dispatch(event_count)
    assert len(executed_replay) == event_count
    assert order_replay == order
    assert replay.vector.to_dict() == vector_snapshot


class _DummyState:
    ip: float = 0.0
    du: float = 0.0
    short_term_memory: ClassVar[list[Any]] = []
    messages_sent_count: int = 0
    last_message_step: int = 0

    def update_collective_metrics(self, ip: float, du: float) -> None:
        pass


class _DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = _DummyState()

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, state: _DummyState) -> None:
        self.state = state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict[str, Any] | None = None,
        vector_store_manager: Any | None = None,
        knowledge_board: Any | None = None,
    ) -> dict[str, int]:
        return {"step": simulation_step}


@pytest.mark.asyncio
async def test_run_turns_concurrent_no_race() -> None:
    from src.sim.simulation import Simulation

    agent_count = 1000
    agents = [_DummyAgent(str(i)) for i in range(agent_count)]
    sim = Simulation(agents=agents)

    results = await sim.run_turns_concurrent(agents)

    assert len(results) == agent_count
    assert sim.current_step == agent_count
