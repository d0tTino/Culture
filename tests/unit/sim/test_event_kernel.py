import pytest

from src.sim.event_kernel import EventKernel
from src.sim.version_vector import VersionVector

pytestmark = pytest.mark.unit


def _make_cb(order: list[int], n: int):
    async def _cb() -> None:
        order.append(n)

    return _cb


@pytest.mark.asyncio
async def test_schedule_order_fifo() -> None:
    kernel = EventKernel()
    order: list[int] = []

    kernel.schedule_immediate_nowait(_make_cb(order, 1))
    await kernel.schedule_immediate(_make_cb(order, 2))
    await kernel.schedule_immediate(_make_cb(order, 3))

    events = await kernel.dispatch(10)

    assert len(events) == 3
    assert order == [1, 2, 3]
    assert kernel.empty()


@pytest.mark.asyncio
async def test_dispatch_limit() -> None:
    kernel = EventKernel()
    order: list[int] = []

    for i in range(3):
        kernel.schedule_immediate_nowait(_make_cb(order, i))

    events = await kernel.dispatch(2)
    assert len(events) == 2
    assert order == [0, 1]
    assert not kernel.empty()

    events += await kernel.dispatch(2)
    assert len(events) == 3
    assert order == [0, 1, 2]
    assert kernel.empty()


@pytest.mark.asyncio
async def test_token_budget_enforced() -> None:
    kernel = EventKernel()
    kernel.set_budget("A", 2)
    order: list[int] = []

    kernel.schedule_immediate_nowait(_make_cb(order, 1), agent_id="A")
    kernel.schedule_immediate_nowait(_make_cb(order, 2), agent_id="A")
    with pytest.raises(ValueError):
        kernel.schedule_immediate_nowait(_make_cb(order, 3), agent_id="A")

    events = await kernel.dispatch(10)
    assert len(events) == 2
    assert order == [1, 2]
    assert kernel.get_budget("A") == 0


@pytest.mark.asyncio
async def test_vector_merging() -> None:
    kernel = EventKernel()
    order: list[int] = []

    vv1 = VersionVector({"A": 1})
    vv2 = VersionVector({"B": 2})

    kernel.schedule_immediate_nowait(_make_cb(order, 1), vector=vv1)
    kernel.schedule_immediate_nowait(_make_cb(order, 2), vector=vv2)

    events = await kernel.dispatch(10)

    assert len(events) == 2
    assert kernel.vector.to_dict() == {"A": 1, "B": 2}
    assert all(e.trace_hash for e in events)


@pytest.mark.asyncio
async def test_schedule_in_future() -> None:
    kernel = EventKernel()
    order: list[int] = []

    await kernel.schedule_in(2, _make_cb(order, 3))
    await kernel.schedule_in(1, _make_cb(order, 2))
    kernel.schedule_immediate_nowait(_make_cb(order, 1))

    events = []
    events += await kernel.dispatch(1)
    assert order == [1]
    events += await kernel.dispatch(1)
    assert order == [1, 2]
    events += await kernel.dispatch(1)
    assert order == [1, 2, 3]
    assert [e.step for e in events] == [0, 1, 2]
