import sys
from unittest.mock import AsyncMock

import pytest


class DummyNeo4j:
    Driver = object
    GraphDatabase = object


class DummyAgentState:
    def __init__(self) -> None:
        self.ip = 0.0
        self.du = 0.0
        self.short_term_memory = []
        self.messages_sent_count = 0
        self.last_message_step = None
        self.collective_ip = 0.0
        self.collective_du = 0.0
        self.is_alive = True
        self.inheritance = 0.0
        self.parent_id: str | None = None
        self.genes: dict[str, float] = {}


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._state = DummyAgentState()

    def get_id(self) -> str:
        return self.agent_id

    @property
    def state(self) -> DummyAgentState:
        return self._state


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inject_event_enqueues_broadcast_and_world_event_kb_entry() -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.sim.simulation import Simulation

    sim = Simulation([DummyAgent("A")])
    sim.event_kernel.emit_environment_event = AsyncMock()

    await sim.handle_control_command(
        {
            "command": "inject_event",
            "text": "aurora visible",
            "scope": "global",
            "author": "gm",
        }
    )

    assert sim.pending_messages_for_next_round
    message = sim.pending_messages_for_next_round[-1]
    assert message["sender_id"] == "gm"
    assert message["recipient_id"] is None
    assert "aurora visible" in message["content"]

    assert sim.messages_to_perceive_this_round[-1] == message

    sim.event_kernel.emit_environment_event.assert_awaited_once()
    payload = sim.event_kernel.emit_environment_event.await_args.args[0]
    assert payload["type"] == "world_event"
    assert payload["author"] == "gm"
    assert payload["text"] == "aurora visible"
    assert payload["scope"] == "global"

    latest_entry = sim.knowledge_board.entries[-1]
    assert latest_entry["entry_type"] == "world_event"
    assert latest_entry["content_full"] == "aurora visible"

    await sim.stop_event_listener()
    sim.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inject_event_is_visible_to_all_agents_next_cycle() -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.sim.simulation import Simulation

    sim = Simulation([DummyAgent("A"), DummyAgent("B")])

    await sim.handle_control_command(
        {
            "command": "inject_event",
            "text": "market crash",
            "scope": "global",
            "author": "narrator",
        }
    )

    sim.messages_to_perceive_this_round = list(sim.pending_messages_for_next_round)
    perceived = sim.messages_to_perceive_this_round
    assert len(perceived) == 1
    assert perceived[0]["recipient_id"] is None
    assert "market crash" in perceived[0]["content"]

    await sim.stop_event_listener()
    sim.close()
