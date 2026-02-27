from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from src.sim.version_vector import VersionVector


class SchedulerProtocol(Protocol):
    """Narrow protocol for deterministic simulation scheduling."""

    async def schedule_at(
        self,
        step: int,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
        vector: VersionVector | None = None,
    ) -> None: ...

    async def schedule_immediate(
        self,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
        vector: VersionVector | None = None,
    ) -> None: ...

    def schedule_at_nowait(
        self,
        step: int,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
        vector: VersionVector | None = None,
    ) -> None: ...

    def schedule_immediate_nowait(
        self,
        callback: Callable[[], Awaitable[None]],
        *,
        agent_id: str | None = None,
        tokens: int = 1,
        vector: VersionVector | None = None,
    ) -> None: ...

    async def dispatch(self, limit: int) -> list[Any]: ...

    async def step(self, limit: int) -> list[Any]: ...

    def pause(self) -> None: ...

    async def resume(self) -> list[Any]: ...

    def empty(self) -> bool: ...

    def queue_depth(self) -> int: ...

    def event_metadata(self, event: Any) -> Mapping[str, Any]: ...

