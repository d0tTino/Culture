import pytest

from src.sim.event_kernel import EventKernel

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_events_dispatched_in_step_order() -> None:
    kernel = EventKernel()
    results: list[int] = []

    async def cb(step: int) -> None:
        results.append(step)

    await kernel.schedule_at(5, lambda: cb(5))
    await kernel.schedule_at(1, lambda: cb(1))
    await kernel.schedule_at(3, lambda: cb(3))

    await kernel.dispatch(10)
    assert results == [1, 3, 5]


@pytest.mark.asyncio
async def test_optimistic_concurrency_skips_past_events() -> None:
    kernel = EventKernel()
    results: list[int] = []

    async def cb(step: int) -> None:
        results.append(step)

    await kernel.schedule_at(1, lambda: cb(1))
    await kernel.dispatch(1)

    # schedule an event in the past
    await kernel.schedule_at(0, lambda: cb(0))
    await kernel.schedule_at(2, lambda: cb(2))
    await kernel.dispatch(10)

    assert results == [1, 2]
