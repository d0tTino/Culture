from __future__ import annotations

import heapq
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Callable, Self


@dataclass(order=True)
class Event:
    """A scheduled event in the simulation."""

    step: int
    count: int
    tokens: int = field(compare=False)
    agent_id: str | None = field(compare=False)
    callback: Callable[[], Awaitable[None]] = field(compare=False)


class EventKernel:
    """Priority-based event scheduler."""

    def __init__(self: Self) -> None:
        self._queue: list[Event] = []
        self._counter = 0
        self.current_step = 0
        self._budgets: dict[str, int] = {}

    def set_budget(self: Self, agent_id: str, tokens: int) -> None:
        """Set the token budget for an agent."""
        self._budgets[agent_id] = tokens

    def add_tokens(self: Self, agent_id: str, tokens: int) -> None:
        """Increase the token budget for an agent."""
        self._budgets[agent_id] = self._budgets.get(agent_id, 0) + tokens

    async def schedule(
        self: Self,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
    ) -> None:
        """Schedule ``callback`` to run at ``current_step``."""
        await self.schedule_at(self.current_step, callback, agent_id=agent_id, tokens=tokens)

    def schedule_nowait(
        self: Self,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
    ) -> None:
        """Synchronously schedule ``callback`` at ``current_step``."""
        self.schedule_at_nowait(self.current_step, callback, agent_id=agent_id, tokens=tokens)

    async def schedule_at(
        self: Self,
        step: int,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
    ) -> None:
        """Schedule ``callback`` to run at a specific ``step``."""
        self.schedule_at_nowait(step, callback, agent_id=agent_id, tokens=tokens)

    def schedule_at_nowait(
        self: Self,
        step: int,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
    ) -> None:
        heapq.heappush(self._queue, Event(step, self._counter, tokens, agent_id, callback))
        self._counter += 1

    async def dispatch(self: Self, limit: int) -> int:
        """Dispatch up to ``limit`` queued events in sorted order."""
        executed = 0
        while executed < limit and self._queue:
            event = heapq.heappop(self._queue)
            if event.step < self.current_step:
                continue
            self.current_step = event.step
            if event.agent_id is not None:
                budget = self._budgets.get(event.agent_id, 0)
                if budget < event.tokens:
                    heapq.heappush(
                        self._queue,
                        Event(
                            event.step + 1,
                            self._counter,
                            event.tokens,
                            event.agent_id,
                            event.callback,
                        ),
                    )
                    self._counter += 1
                    continue
                self._budgets[event.agent_id] = budget - event.tokens
            await event.callback()
            executed += 1
        return executed

    def empty(self: Self) -> bool:
        return not self._queue
