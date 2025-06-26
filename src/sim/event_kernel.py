from __future__ import annotations

import heapq
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Callable, Self

from src.infra.snapshot import compute_trace_hash

from .version_vector import VersionVector


@dataclass(order=True)
class Event:
    """A scheduled event in the simulation."""

    step: int
    count: int
    tokens: int = field(compare=False)
    agent_id: str | None = field(compare=False)
    callback: Callable[[], Awaitable[None]] = field(compare=False)
    vector: VersionVector = field(default_factory=VersionVector, compare=False)
    trace_hash: str = field(default="", compare=False)


class EventKernel:
    """Priority-based event scheduler."""

    def __init__(self: Self) -> None:
        self._queue: list[Event] = []
        self._counter = 0
        self.current_step = 0
        self._budgets: dict[str, int] = {}
        self.vector = VersionVector()

    def set_budget(self: Self, agent_id: str, tokens: int) -> None:
        """Set the token budget for an agent."""
        self._budgets[agent_id] = tokens

    def add_tokens(self: Self, agent_id: str, tokens: int) -> None:
        """Increase the token budget for an agent."""
        self._budgets[agent_id] = self._budgets.get(agent_id, 0) + tokens

    def get_budget(self: Self, agent_id: str) -> int:
        """Return remaining token budget for ``agent_id``."""
        return int(self._budgets.get(agent_id, 0))

    async def schedule(
        self: Self,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
        vector: VersionVector | None = None,
    ) -> None:
        """Schedule ``callback`` to run at ``current_step``."""
        await self.schedule_at(
            self.current_step,
            callback,
            agent_id=agent_id,
            tokens=tokens,
            vector=vector,
        )

    def schedule_nowait(
        self: Self,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
        vector: VersionVector | None = None,
    ) -> None:
        """Synchronously schedule ``callback`` at ``current_step``."""
        self.schedule_at_nowait(
            self.current_step,
            callback,
            agent_id=agent_id,
            tokens=tokens,
            vector=vector,
        )

    async def schedule_at(
        self: Self,
        step: int,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
        vector: VersionVector | None = None,
    ) -> None:
        """Schedule ``callback`` to run at a specific ``step``."""
        self.schedule_at_nowait(
            step,
            callback,
            agent_id=agent_id,
            tokens=tokens,
            vector=vector,
        )

    def schedule_at_nowait(
        self: Self,
        step: int,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
        vector: VersionVector | None = None,
    ) -> None:
        if agent_id is not None:
            budget = self._budgets.get(agent_id, 0)
            if budget < tokens:
                raise ValueError(f"Agent {agent_id} exceeded token budget")
            self._budgets[agent_id] = budget - tokens

        vv = vector or VersionVector()
        event_data = {
            "step": step,
            "count": self._counter,
            "tokens": tokens,
            "agent_id": agent_id,
            "vector": vv.to_dict(),
        }
        trace_hash = compute_trace_hash(event_data)
        heapq.heappush(
            self._queue,
            Event(step, self._counter, tokens, agent_id, callback, vv, trace_hash),
        )
        self._counter += 1

    async def dispatch(self: Self, limit: int) -> list[Event]:
        """Dispatch up to ``limit`` queued events in sorted order."""
        executed: list[Event] = []
        while len(executed) < limit and self._queue:
            event = heapq.heappop(self._queue)
            if event.step < self.current_step:
                continue
            self.current_step = event.step
            self.vector.merge(event.vector)
            await event.callback()
            executed.append(event)
        return executed

    def empty(self: Self) -> bool:
        return not self._queue
