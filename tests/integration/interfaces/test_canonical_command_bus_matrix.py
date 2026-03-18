from __future__ import annotations

import sys

import pytest

from src.interfaces.dashboard_backend import SimulationEvent
from src.interfaces.interaction_schema import InteractionContext

pytestmark = pytest.mark.integration


class DummyNeo4j:
    Driver = object
    GraphDatabase = object


class DummyState:
    def __init__(self) -> None:
        self.ip = 5.0
        self.du = 5.0
        self.short_term_memory = []
        self.messages_sent_count = 0
        self.last_message_step = None
        self.collective_ip = 0.0
        self.collective_du = 0.0


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._state = DummyState()
        from src.infra.ledger import ledger as _ledger

        _ledger.log_change(agent_id, 5.0, 5.0, "init")

    def get_id(self) -> str:
        return self.agent_id

    @property
    def state(self) -> DummyState:
        return self._state

    async def run_turn(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {}


@pytest.mark.asyncio
async def test_command_bus_matrix_routes_all_flows_through_canonical_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.sim.simulation import Simulation

    sim = Simulation([DummyAgent("alpha"), DummyAgent("beta")])
    seen: list[str] = []
    original = sim.command_bus.dispatch_payload

    async def recording_dispatch(
        payload: dict[str, object], *, context: InteractionContext | None = None
    ):
        seen.append(
            str(payload.get("intent") or payload.get("command") or payload.get("command_type"))
        )
        return await original(payload, context=context)

    monkeypatch.setattr(sim.command_bus, "dispatch_payload", recording_dispatch)

    await sim.command_bus.dispatch_payload(
        {
            "intent": "direct_message",
            "text": "hi beta",
            "recipient_id": "beta",
            "target_agent_id": "beta",
        },
        context=InteractionContext(sender_id="user-dm", source="discord"),
    )
    await sim.command_bus.dispatch_payload(
        {"intent": "broadcast", "text": "hello all"},
        context=InteractionContext(sender_id="user-broadcast", source="dashboard"),
    )
    await sim.external_event_ingestion.route_event(
        SimulationEvent(
            type="control",
            data={"intent": "inject_event", "text": "storm front", "source": "event_bus"},
        )
    )
    await sim.command_bus.dispatch_payload(
        {"intent": "moderation", "command": "mute", "agent_id": "beta"},
        context=InteractionContext(sender_id="mod-1", source="discord", permissions={"moderator"}),
    )

    async with sim._msg_lock:
        contents = [str(msg.get("content")) for msg in sim.pending_messages_for_next_round]

    assert seen == ["direct_message", "broadcast", "inject_event", "moderation"]
    assert any(content == "hi beta" for content in contents)
    assert any(content == "hello all" for content in contents)
    assert any("storm front" in content for content in contents)
    assert "beta" in sim.muted_agents

    await sim.stop_event_listener()
    sim.close()
