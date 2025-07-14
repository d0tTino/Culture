from __future__ import annotations

import asyncio
from typing import Any

from typing_extensions import Self


class EventBus:
    """Simple publish/subscribe event bus."""

    def __init__(self: Self) -> None:
        self._queues: list[asyncio.Queue[Any | None]] = []

    def subscribe(self: Self) -> asyncio.Queue[Any | None]:
        """Return a new queue subscribed to published events."""
        q: asyncio.Queue[Any | None] = asyncio.Queue()
        self._queues.append(q)
        return q

    def unsubscribe(self: Self, q: asyncio.Queue[Any | None]) -> None:
        """Remove ``q`` from the subscriber list if present."""
        try:
            self._queues.remove(q)
        except ValueError:  # pragma: no cover - defensive
            pass

    async def publish(self: Self, event: Any) -> None:
        """Publish ``event`` to all subscribers."""
        for q in list(self._queues):
            await q.put(event)

    def shutdown(self: Self) -> None:
        """Send ``None`` to all subscribers and clear them."""
        for q in list(self._queues):
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:  # pragma: no cover - defensive
                pass
        self._queues.clear()


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
