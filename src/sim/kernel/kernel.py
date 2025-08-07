"""Priority-queue based discrete event scheduler."""

from __future__ import annotations

import heapq
from collections.abc import Awaitable, Callable
from typing import Any

from typing_extensions import Self

from .event import Event


class DiscreteEventKernel:
    """A minimal discrete-event simulation kernel."""

    def __init__(self: Self) -> None:
        self._queue: list[Event] = []
        self._seq = 0
        self.now: int = 0
        self._paused = False

    # ------------------------------------------------------------------
    # scheduling helpers
    def schedule_at_nowait(
        self: Self, ts: int, callback: Callable[[], Awaitable[None]], **_kwargs: Any
    ) -> None:
        """Schedule ``callback`` to run at ``ts``."""
        heapq.heappush(self._queue, Event(ts, self._seq, callback))
        self._seq += 1

    async def schedule_at(
        self: Self, ts: int, callback: Callable[[], Awaitable[None]], **_kwargs: Any
    ) -> None:
        self.schedule_at_nowait(ts, callback, **_kwargs)

    def schedule_immediate_nowait(
        self: Self, callback: Callable[[], Awaitable[None]], **_kwargs: Any
    ) -> None:
        self.schedule_at_nowait(self.now, callback, **_kwargs)

    async def schedule_immediate(
        self: Self, callback: Callable[[], Awaitable[None]], **_kwargs: Any
    ) -> None:
        self.schedule_immediate_nowait(callback, **_kwargs)

    def schedule_in_nowait(
        self: Self, delay: int, callback: Callable[[], Awaitable[None]], **_kwargs: Any
    ) -> None:
        self.schedule_at_nowait(self.now + delay, callback, **_kwargs)

    async def schedule_in(
        self: Self, delay: int, callback: Callable[[], Awaitable[None]], **_kwargs: Any
    ) -> None:
        self.schedule_in_nowait(delay, callback, **_kwargs)

    # ------------------------------------------------------------------
    def empty(self: Self) -> bool:
        return not self._queue

    def pause(self: Self) -> None:
        self._paused = True

    async def resume(self: Self) -> list[Event]:
        self._paused = False
        return await self.run()

    async def run(self: Self) -> list[Event]:
        """Run until the queue is empty or paused."""
        executed: list[Event] = []
        while self._queue and not self._paused:
            executed.extend(await self.step(1))
        return executed

    async def step(self: Self, n: int = 1) -> list[Event]:
        """Execute up to ``n`` events."""
        executed: list[Event] = []
        for _ in range(n):
            if not self._queue or self._paused:
                break
            event = heapq.heappop(self._queue)
            self.now = max(self.now, event.ts)
            await event.callback()
            executed.append(event)
        return executed

    async def fast_forward(self: Self, until_ts: int) -> list[Event]:
        """Execute events with ``ts`` <= ``until_ts``."""
        executed: list[Event] = []
        while self._queue and self._queue[0].ts <= until_ts and not self._paused:
            event = heapq.heappop(self._queue)
            self.now = event.ts
            await event.callback()
            executed.append(event)
        return executed

    # Stubs to satisfy existing Simulation interface ---------------------------------
    async def emit_environment_event(
        self: Self, _event: dict[str, Any]
    ) -> None:  # pragma: no cover
        """Stub for compatibility with the older kernel."""
        return None

    async def forward_external_events(
        self: Self, _handler: Callable[[str], Awaitable[None]]
    ) -> None:  # pragma: no cover
        """Stub for compatibility with the older kernel."""
        return None
