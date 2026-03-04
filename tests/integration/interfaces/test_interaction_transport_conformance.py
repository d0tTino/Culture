from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.interfaces.interaction_policy import (
    discord_interaction_context,
    discord_message_to_intent_payload,
)
from src.interfaces.interaction_schema import InteractionContext


class DummyDiscordClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.event = lambda f: f
        self.user = "dummy"


class DummyState:
    def __init__(self) -> None:
        self.ip = 2.0
        self.du = 2.0
        self.short_term_memory: list[object] = []
        self.messages_sent_count = 0
        self.last_message_step = None
        self.collective_ip = 0.0
        self.collective_du = 0.0


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._state = DummyState()

    def get_id(self) -> str:
        return self.agent_id

    @property
    def state(self) -> DummyState:
        return self._state

    def update_state(self, state: DummyState) -> None:
        self._state = state

    async def run_turn(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {}


async def _pending_message_shape(sim) -> tuple[str, str | None, str | None]:
    async with sim._msg_lock:
        msg = sim.pending_messages_for_next_round[-1]
        return (
            str(msg.get("content", "")),
            msg.get("recipient_id"),
            msg.get("target_agent_id"),
        )


@pytest.mark.integration
@pytest.mark.anyio("asyncio")
async def test_intent_conformance_across_transports(tmp_path: Path) -> None:
    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    ledger = Ledger(tmp_path / "ledger.sqlite")
    with (
        patch("src.infra.ledger.ledger", ledger),
        patch("src.sim.simulation.ledger", ledger),
        patch("src.interfaces.discord_bot.discord.Client", DummyDiscordClient),
    ):
        # Discord transport adapter -> canonical intent
        discord_sim = Simulation([DummyAgent("A")])
        discord_payload, err = discord_message_to_intent_payload(
            content="hello world",
            sender_agent_id=None,
            fallback_agent_id="A",
            raw_metadata={"channel_id": 10, "user_id": 20},
        )
        assert err is None
        assert discord_payload is not None
        discord_result = await discord_sim.interaction_service.execute_from_payload(
            discord_payload,
            context=discord_interaction_context(
                user=SimpleNamespace(id=20),
                channel=SimpleNamespace(id=10),
            ),
        )
        discord_outcome = await _pending_message_shape(discord_sim)

        # Dashboard transport -> same canonical intent and same domain outcome
        dashboard_sim = Simulation([DummyAgent("A")])
        dashboard_result = await dashboard_sim.interaction_service.execute_from_payload(
            {
                "intent": "human_message",
                "text": "hello world",
                "target_agent_id": "A",
            },
            context=InteractionContext(sender_id="dashboard", source="dashboard"),
        )
        dashboard_outcome = await _pending_message_shape(dashboard_sim)

        # Future websocket/API transport -> same canonical intent through same service
        ws_sim = Simulation([DummyAgent("A")])
        ws_result = await ws_sim.interaction_service.execute_from_payload(
            {
                "intent": "human_message",
                "text": "hello world",
                "target_agent_id": "A",
            },
            context=InteractionContext(sender_id="api-user", source="websocket"),
        )
        ws_outcome = await _pending_message_shape(ws_sim)

        assert discord_result.status == dashboard_result.status == ws_result.status == "ok"
        assert discord_outcome == dashboard_outcome == ws_outcome

        discord_sim.close()
        dashboard_sim.close()
        ws_sim.close()
