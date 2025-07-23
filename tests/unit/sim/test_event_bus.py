import asyncio

import pytest

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
