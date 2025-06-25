import pytest

from src.sim.event_kernel import EventKernel
from src.sim.version_vector import VersionVector

pytestmark = pytest.mark.stress


def _make_cb(order: list[int], n: int):
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
    assert executed == event_count
    assert kernel.empty()
    assert order == list(range(event_count))
    vector_snapshot = kernel.vector.to_dict()

    replay = EventKernel()
    order_replay: list[int] = []
    for i in range(event_count):
        vv = VersionVector({"A": i + 1})
        replay.schedule_nowait(_make_cb(order_replay, i), vector=vv)

    executed_replay = await replay.dispatch(event_count)
    assert executed_replay == event_count
    assert order_replay == order
    assert replay.vector.to_dict() == vector_snapshot
