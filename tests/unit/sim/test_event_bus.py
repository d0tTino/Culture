import asyncio

import pytest

from src.interfaces.dashboard_backend import SimulationEvent
from src.sim import event_bus

pytestmark = pytest.mark.unit


def test_shutdown_allows_fresh_event_bus_in_new_loop() -> None:
    loop1 = asyncio.new_event_loop()
    asyncio.set_event_loop(loop1)
    bus1 = event_bus.get_event_bus()
    bus1.shutdown()
    assert event_bus._event_bus is None
    assert event_bus._event_bus_loop is None
    loop1.close()

    loop2 = asyncio.new_event_loop()
    asyncio.set_event_loop(loop2)

    async def _get_bus() -> event_bus.EventBus:
        return event_bus.get_event_bus()

    bus2 = loop2.run_until_complete(_get_bus())
    assert bus2 is not bus1
    assert event_bus._event_bus_loop is loop2
    bus2.shutdown()
    loop2.close()
    asyncio.set_event_loop(None)


def test_shutdown_and_restart_same_loop() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bus1 = event_bus.get_event_bus()
    q1 = bus1.subscribe()
    bus1.shutdown()
    assert q1.get_nowait() is None
    assert not bus1._queues

    async def _get_bus_and_publish() -> event_bus.EventBus:
        bus2 = event_bus.get_event_bus()
        q2 = bus2.subscribe()
        await bus2.publish(SimulationEvent(type="t", data={}))
        assert await q2.get() is not None
        return bus2

    bus2 = loop.run_until_complete(_get_bus_and_publish())
    assert bus2 is not bus1
    assert event_bus._event_bus_loop is loop
    assert q1.empty()
    bus2.shutdown()
    loop.close()
    asyncio.set_event_loop(None)
