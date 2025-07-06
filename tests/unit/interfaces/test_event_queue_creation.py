import asyncio

import pytest

from src.interfaces import dashboard_backend as db


async def _clear_event_queue() -> None:
    queue = db.get_event_queue()
    while not queue.empty():
        _ = await queue.get()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_event_queue_running_loop() -> None:
    await _clear_event_queue()
    q1 = db.get_event_queue()
    await q1.put(db.SimulationEvent(event_type="a", data={}))
    q2 = db.get_event_queue()
    assert q1 is q2
    _ = await q1.get()


@pytest.mark.unit
def test_get_event_queue_no_running_loop() -> None:
    q1 = db.get_event_queue()

    async def call_in_loop() -> asyncio.Queue[db.SimulationEvent | None]:
        return db.get_event_queue()

    loop = asyncio.new_event_loop()
    try:
        q2 = loop.run_until_complete(call_in_loop())
    finally:
        loop.close()

    assert q1 is not q2
