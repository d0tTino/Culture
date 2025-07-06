from __future__ import annotations

import asyncio
import heapq
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable

from typing_extensions import Self

from src.infra.event_log import log_event
from src.infra.snapshot import compute_trace_hash
from src.interfaces.dashboard_backend import (
    SimulationEvent,
    emit_event,
    emit_map_action_event,
    get_event_queue,
)

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

    async def schedule_immediate(
        self: Self,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
        vector: VersionVector | None = None,
    ) -> None:
        """Schedule ``callback`` to run at the current kernel step."""
        await self.schedule_at(
            self.current_step,
            callback,
            agent_id=agent_id,
            tokens=tokens,
            vector=vector,
        )

    # Backwards compatibility
    async def schedule(
        self: Self,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
        vector: VersionVector | None = None,
    ) -> None:
        await self.schedule_immediate(
            callback,
            agent_id=agent_id,
            tokens=tokens,
            vector=vector,
        )

    def schedule_immediate_nowait(
        self: Self,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
        vector: VersionVector | None = None,
    ) -> None:
        """Synchronously schedule ``callback`` at the current kernel step."""
        self.schedule_at_nowait(
            self.current_step,
            callback,
            agent_id=agent_id,
            tokens=tokens,
            vector=vector,
        )

    # Backwards compatibility
    def schedule_nowait(
        self: Self,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
        vector: VersionVector | None = None,
    ) -> None:
        self.schedule_immediate_nowait(
            callback,
            agent_id=agent_id,
            tokens=tokens,
            vector=vector,
        )

    async def schedule_in(
        self: Self,
        delay: int,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
        vector: VersionVector | None = None,
    ) -> None:
        """Schedule ``callback`` to run ``delay`` steps in the future."""
        await self.schedule_at(
            self.current_step + delay,
            callback,
            agent_id=agent_id,
            tokens=tokens,
            vector=vector,
        )

    def schedule_in_nowait(
        self: Self,
        delay: int,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
        vector: VersionVector | None = None,
    ) -> None:
        self.schedule_at_nowait(
            self.current_step + delay,
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

    async def emit_environment_event(self: Self, event: dict[str, Any]) -> None:
        """Log and forward an environment event."""
        event_with_hash = log_event(event)
        if event_with_hash is None:
            event_with_hash = {**event, "trace_hash": compute_trace_hash(event)}
        if event.get("type") == "map_action":
            await emit_map_action_event(
                event.get("agent_id", ""),
                event.get("step", 0),
                event.get("action", ""),
                **{
                    k: v
                    for k, v in event.items()
                    if k not in {"type", "agent_id", "step", "action", "trace_hash"}
                },
            )
        else:
            await emit_event(SimulationEvent(event_type=event["type"], data=event_with_hash))

    async def forward_external_events(
        self: Self, handler: Callable[[str], Awaitable[None]]
    ) -> None:
        """Forward broadcast events from the shared queue to ``handler``.

        This coroutine listens indefinitely on the global event queue and
        passes the ``content`` of any broadcast events to ``handler``. The
        loop exits when the queue yields ``None``.
        """
        queue = get_event_queue()
        try:
            while True:
                evt: SimulationEvent | None = await queue.get()
                if evt is None:
                    break
                if evt.event_type == "broadcast" and evt.data:
                    content = evt.data.get("content")
                    if isinstance(content, str):
                        await handler(content)
        except asyncio.CancelledError:
            pass
