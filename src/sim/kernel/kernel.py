"""Deprecated adapter over :class:`src.sim.event_kernel.EventKernel`."""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable
from typing import Any

from typing_extensions import Self

from src.sim.event_kernel import EventKernel

from .event import Event


class DiscreteEventKernel:
    """Compatibility adapter retained for tests and fixtures only."""

    def __init__(self: Self) -> None:
        warnings.warn(
            "DiscreteEventKernel is deprecated; use EventKernel directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._kernel = EventKernel()

    @property
    def now(self: Self) -> int:
        return self._kernel.current_step

    def _to_legacy_event(self: Self, event: Any) -> Event:
        return Event(ts=int(event.step), seq=int(event.count), callback=event.callback)

    # scheduling helpers -------------------------------------------------
    def schedule_at_nowait(
        self: Self, ts: int, callback: Callable[[], Awaitable[None]], **kwargs: Any
    ) -> None:
        self._kernel.schedule_at_nowait(ts, callback, **kwargs)

    async def schedule_at(
        self: Self, ts: int, callback: Callable[[], Awaitable[None]], **kwargs: Any
    ) -> None:
        await self._kernel.schedule_at(ts, callback, **kwargs)

    def schedule_immediate_nowait(
        self: Self, callback: Callable[[], Awaitable[None]], **kwargs: Any
    ) -> None:
        self._kernel.schedule_immediate_nowait(callback, **kwargs)

    async def schedule_immediate(
        self: Self, callback: Callable[[], Awaitable[None]], **kwargs: Any
    ) -> None:
        await self._kernel.schedule_immediate(callback, **kwargs)

    def schedule_in_nowait(
        self: Self, delay: int, callback: Callable[[], Awaitable[None]], **kwargs: Any
    ) -> None:
        self._kernel.schedule_in_nowait(delay, callback, **kwargs)

    async def schedule_in(
        self: Self, delay: int, callback: Callable[[], Awaitable[None]], **kwargs: Any
    ) -> None:
        await self._kernel.schedule_in(delay, callback, **kwargs)

    # run/flow control ---------------------------------------------------
    def empty(self: Self) -> bool:
        return self._kernel.empty()

    def pause(self: Self) -> None:
        self._kernel.pause()

    async def resume(self: Self) -> list[Event]:
        return [self._to_legacy_event(event) for event in await self._kernel.resume()]

    async def run(self: Self) -> list[Event]:
        events = await self._kernel.dispatch(limit=self._kernel.queue_depth())
        return [self._to_legacy_event(event) for event in events]

    async def step(self: Self, n: int = 1) -> list[Event]:
        return [self._to_legacy_event(event) for event in await self._kernel.step(n)]

    async def fast_forward(self: Self, until_ts: int) -> list[Event]:
        executed: list[Event] = []
        while not self.empty() and self._kernel._queue[0].step <= until_ts:
            next_events = await self.step(1)
            if not next_events:
                break
            executed.extend(next_events)
        return executed

    # compatibility stubs ------------------------------------------------
    async def emit_environment_event(self: Self, event: dict[str, Any]) -> None:
        await self._kernel.emit_environment_event(event)

    async def forward_external_events(
        self: Self, handler: Callable[[str], Awaitable[None]]
    ) -> None:
        async def _adapted_handler(content: str, _metadata: dict[str, Any] | None) -> None:
            await handler(content)

        await self._kernel.forward_external_events(_adapted_handler)
