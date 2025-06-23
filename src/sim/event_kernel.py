from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Callable


@dataclass
class Event:
    """A scheduled event in the simulation."""

    callback: Callable[[], Awaitable[None]]


class EventKernel:
    """Simple FIFO event scheduler."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event] = asyncio.Queue()

    async def schedule(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Schedule a new event callback."""
        await self._queue.put(Event(callback))

    def schedule_nowait(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Synchronously schedule a new event callback."""
        self._queue.put_nowait(Event(callback))

    async def dispatch(self, limit: int) -> int:
        """Dispatch up to ``limit`` queued events."""
        executed = 0
        while executed < limit and not self._queue.empty():
            event = await self._queue.get()
            await event.callback()
            executed += 1
        return executed

    def empty(self) -> bool:
        return self._queue.empty()
