from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.sim.event_bus import get_event_bus

if TYPE_CHECKING:
    from src.interfaces.dashboard_backend import SimulationEvent
    from src.interfaces.discord_bot import SimulationDiscordBot


class DiscordSimulationEventListener:
    """Discord adapter that consumes SimulationEvent bus side effects."""

    def __init__(self, bot: SimulationDiscordBot) -> None:
        self.bot = bot
        self._queue: asyncio.Queue[SimulationEvent | None] = get_event_bus().subscribe()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        get_event_bus().unsubscribe(self._queue)

    async def _run(self) -> None:
        while True:
            evt = await self._queue.get()
            if evt is None:
                return
            if evt.type == "spawn_rejected" and evt.data:
                await self.bot.send_simulation_update(
                    content=f"Rejected spawn request for duplicate agent_id '{evt.data.get('agent_id', '')}'."
                )
            if evt.type == "knowledge_board" and evt.data:
                embed = self.bot.create_knowledge_board_embed(
                    str(evt.data.get("agent_id", "")),
                    str(evt.data.get("content", "")),
                    int(evt.data.get("step", 0)),
                )
                await self.bot.send_simulation_update(
                    embed=embed,
                    agent_id=str(evt.data.get("agent_id", "")),
                )
