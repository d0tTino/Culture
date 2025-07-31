from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from typing_extensions import Self

from src.interfaces import metrics

EVENT_BUS_QUEUE_SIZE = metrics.Gauge(
    "event_bus_queue_size",
    "Number of queues subscribed to the EventBus",
)

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from src.interfaces.dashboard_backend import SimulationEvent


class EventBus:
    """Simple publish/subscribe event bus."""

    def __init__(self: Self) -> None:
        self._queues: list[asyncio.Queue[SimulationEvent | None]] = []
        EVENT_BUS_QUEUE_SIZE.set(0)

    def subscribe(self: Self) -> asyncio.Queue[SimulationEvent | None]:
        """Return a new queue subscribed to published events."""
        q: asyncio.Queue[SimulationEvent | None] = asyncio.Queue()
        self._queues.append(q)
        EVENT_BUS_QUEUE_SIZE.set(len(self._queues))
        return q

    def unsubscribe(self: Self, q: asyncio.Queue[SimulationEvent | None]) -> None:
        """Remove ``q`` from the subscriber list if present."""
        try:
            self._queues.remove(q)
            EVENT_BUS_QUEUE_SIZE.set(len(self._queues))
        except ValueError:  # pragma: no cover - defensive
            pass

    async def publish(self: Self, event: SimulationEvent) -> None:
        """Publish ``event`` to all subscribers."""
        for q in list(self._queues):
            await q.put(event)
        EVENT_BUS_QUEUE_SIZE.set(len(self._queues))

    def shutdown(self: Self) -> None:
        """Send ``None`` to all subscribers and clear them."""
        for q in list(self._queues):
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:  # pragma: no cover - defensive
                pass
        self._queues.clear()
        EVENT_BUS_QUEUE_SIZE.set(0)
        global _event_bus, _event_bus_loop
        _event_bus = None
        _event_bus_loop = None


_event_bus: EventBus | None = None
_event_bus_loop: asyncio.AbstractEventLoop | None = None


def get_event_bus() -> EventBus:
    """Return an ``EventBus`` bound to the current event loop."""
    global _event_bus, _event_bus_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - no running loop
        loop = asyncio.new_event_loop()
    if _event_bus is None or _event_bus_loop is not loop:
        _event_bus = EventBus()
        _event_bus_loop = loop
    return _event_bus


__all__ = ["EventBus", "get_event_bus"]
