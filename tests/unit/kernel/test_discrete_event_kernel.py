import pytest

from src.sim.kernel import DiscreteEventKernel

pytestmark = pytest.mark.unit


def _make_cb(order: list[int], n: int):
    async def _cb() -> None:
        order.append(n)

    return _cb


@pytest.mark.asyncio
async def test_ordering() -> None:
    kernel = DiscreteEventKernel()
    order: list[int] = []
    kernel.schedule_immediate_nowait(_make_cb(order, 1))
    kernel.schedule_immediate_nowait(_make_cb(order, 2))
    kernel.schedule_in_nowait(1, _make_cb(order, 3))
    await kernel.run()
    assert order == [1, 2, 3]


@pytest.mark.asyncio
async def test_step_and_resume() -> None:
    kernel = DiscreteEventKernel()
    order: list[int] = []
    for i in range(3):
        kernel.schedule_immediate_nowait(_make_cb(order, i))
    events = await kernel.step(2)
    assert [e.seq for e in events] == [0, 1]
    assert order == [0, 1]
    await kernel.step(1)
    assert order == [0, 1, 2]


@pytest.mark.asyncio
async def test_pause_resume() -> None:
    kernel = DiscreteEventKernel()
    order: list[int] = []

    async def first() -> None:
        order.append(1)
        kernel.pause()

    kernel.schedule_immediate_nowait(first)
    kernel.schedule_in_nowait(1, _make_cb(order, 2))

    await kernel.run()
    assert order == [1]
    await kernel.resume()
    assert order == [1, 2]
