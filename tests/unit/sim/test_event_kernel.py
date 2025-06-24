import pytest

from src.sim.event_kernel import EventKernel

pytestmark = pytest.mark.unit

def _make_cb(order: list[int], n: int):
    async def _cb() -> None:
        order.append(n)
    return _cb


@pytest.mark.asyncio
async def test_schedule_order_fifo() -> None:
    kernel = EventKernel()
    order: list[int] = []

    kernel.schedule_nowait(_make_cb(order, 1))
    await kernel.schedule(_make_cb(order, 2))
    await kernel.schedule(_make_cb(order, 3))

    executed = await kernel.dispatch(10)

    assert executed == 3
    assert order == [1, 2, 3]
    assert kernel.empty()


@pytest.mark.asyncio
async def test_dispatch_limit() -> None:
    kernel = EventKernel()
    order: list[int] = []

    for i in range(3):
        kernel.schedule_nowait(_make_cb(order, i))

    executed = await kernel.dispatch(2)
    assert executed == 2
    assert order == [0, 1]
    assert not kernel.empty()

    executed += await kernel.dispatch(2)
    assert executed == 3
    assert order == [0, 1, 2]
    assert kernel.empty()
