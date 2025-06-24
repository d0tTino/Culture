from __future__ import annotations

import heapq
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Callable


@dataclass(order=True)
class Event:
    """A scheduled event in the simulation."""

    step: int
    count: int
    callback: Callable[[], Awaitable[None]] = field(compare=False)


class EventKernel:
    """Priority-based event scheduler."""

    def __init__(self) -> None:
        self._queue: list[Event] = []
        self._counter = 0
        self.current_step = 0

    async def schedule(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Schedule ``callback`` to run at ``current_step``."""
        await self.schedule_at(self.current_step, callback)

    def schedule_nowait(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Synchronously schedule ``callback`` at ``current_step``."""
        self.schedule_at_nowait(self.current_step, callback)

    async def schedule_at(self, step: int, callback: Callable[[], Awaitable[None]]) -> None:
        """Schedule ``callback`` to run at a specific ``step``."""
        self.schedule_at_nowait(step, callback)

    def schedule_at_nowait(self, step: int, callback: Callable[[], Awaitable[None]]) -> None:
        heapq.heappush(self._queue, Event(step, self._counter, callback))
        self._counter += 1

    async def dispatch(self, limit: int) -> int:
        """Dispatch up to ``limit`` queued events in sorted order."""
        executed = 0
        while executed < limit and self._queue:
            event = heapq.heappop(self._queue)
            if event.step < self.current_step:
                continue
            self.current_step = event.step
            await event.callback()
            executed += 1
        return executed

    def empty(self) -> bool:
        return not self._queue
