from __future__ import annotations

import asyncio
from typing import Any

from src.interfaces.dashboard_backend import SimulationEvent
from src.interfaces.interaction_commands import InteractionContext
from src.shared.typing import SimulationMessage
from src.sim.event_bus import get_event_bus


class ExternalEventIngestionService:
    """Owns event-listener lifecycle and external event routing."""

    def __init__(self, simulation: Any) -> None:
        self.simulation = simulation

    async def start(self) -> None:
        sim = self.simulation
        if sim._event_listener_task is None or sim._event_listener_task.done():
            sim._event_listener_task = asyncio.create_task(self._event_listener_loop())
        if sim._event_task is None or sim._event_task.done():
            sim._event_task = asyncio.create_task(
                sim.event_kernel.forward_external_events(sim._handle_human_command_from_bus)
            )

    async def stop(self) -> None:
        sim = self.simulation
        if sim._event_listener_task:
            sim._event_listener_task.cancel()
            try:
                await sim._event_listener_task
            except asyncio.CancelledError:  # pragma: no cover - expected
                pass
            sim._event_listener_task = None
        if sim._event_task:
            sim._event_task.cancel()
            try:
                await sim._event_task
            except asyncio.CancelledError:  # pragma: no cover - expected
                pass
            sim._event_task = None
        if sim._discord_listener is not None:
            await sim._discord_listener.stop()
            sim._discord_listener = None

    async def _event_listener_loop(self) -> None:
        bus = get_event_bus()
        queue = bus.subscribe()
        try:
            while True:
                evt: SimulationEvent | None = await queue.get()
                if evt is None:
                    break
                await self.route_event(evt)
        except asyncio.CancelledError:  # pragma: no cover - task cancelled
            pass
        finally:
            bus.unsubscribe(queue)

    async def route_event(self, evt: SimulationEvent) -> None:
        sim = self.simulation
        if not evt.data:
            return
        if evt.type == "control":
            context = InteractionContext(
                sender_id=str(evt.data.get("sender_id", evt.data.get("author", "external"))),
                channel_id=str(evt.data.get("channel_id")) if evt.data.get("channel_id") else None,
                source=str(evt.data.get("source", "event_bus")),
                permissions=(
                    set(evt.data.get("permissions", []))
                    if isinstance(evt.data.get("permissions"), list)
                    else set()
                ),
                metadata={k: v for k, v in evt.data.items()},
            )
            await sim.command_bus.dispatch_payload(evt.data, context=context)
            return
        if evt.type == "moderation":
            context = InteractionContext(
                sender_id=str(evt.data.get("sender_id", evt.data.get("author", "external"))),
                source=str(evt.data.get("source", "event_bus")),
                permissions={"admin", "moderator"},
                metadata={k: v for k, v in evt.data.items()},
            )
            await sim.command_bus.dispatch_payload(evt.data, context=context)
            return

        sender = str(evt.data.get("author", "external"))
        if sender in sim.muted_agents:
            return

        recipient = evt.data.get("recipient_id") if evt.type == "direct_message" else None
        msg: SimulationMessage = {
            "step": sim.current_step,
            "turn_index": sim.current_step,
            "world_time": sim.environment_system.world_time_snapshot(),
            "sender_id": sender,
            "recipient_id": recipient,
            "content": str(evt.data.get("content", "")),
            "action_intent": None,
            "sentiment_score": None,
        }
        async with sim._msg_lock:
            sim.pending_messages_for_next_round.append(msg)
            sim.messages_to_perceive_this_round.append(msg)
