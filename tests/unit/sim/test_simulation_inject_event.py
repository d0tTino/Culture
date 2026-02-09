import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_backend_kb_write_commands_use_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.infra import config
    from src.sim.graph_knowledge_board import GraphKnowledgeBoard
    from src.sim.simulation import Simulation

    class DummyGraphBoard(GraphKnowledgeBoard):
        def __init__(self) -> None:
            self.lock = asyncio.Lock()
            self.add_entry = MagicMock()

    monkeypatch.setattr(config, "KNOWLEDGE_BOARD_BACKEND", "graph")
    monkeypatch.setattr("src.sim.simulation.GraphKnowledgeBoard", DummyGraphBoard)

    sim = Simulation([DummyAgent("A")])
    sim.event_kernel.emit_environment_event = AsyncMock()

    await sim._handle_human_command("/kb graph path")
    await sim.handle_control_command(
        {
            "command": "post_kb",
            "text": "moderator entry",
            "author": "mod",
        }
    )
    await sim.handle_control_command(
        {
            "command": "inject_event",
            "text": "storm warning",
            "scope": "global",
            "author": "gm",
        }
    )

    assert sim.knowledge_board.add_entry.call_count == 3
    call_args = sim.knowledge_board.add_entry.call_args_list
    assert call_args[0].args[1] == "human"
    assert call_args[1].args[1] == "mod"
    assert call_args[2].args[1] == "gm"

    await sim.stop_event_listener()
    sim.close()


@pytest.mark.unit
def test_apply_event_replaces_memory_backend_kb_state() -> None:
    from src.sim.simulation import Simulation

    sim = Simulation([DummyAgent("A")])
    event = {
        "type": "agent_action",
        "agent_id": "A",
        "knowledge_board": {
            "entries": [
                {
                    "entry_id": "e1",
                    "step": 1,
                    "agent_id": "A",
                    "entry_type": "note",
                    "tags": [],
                    "reference_metadata": None,
                    "content_full": "hello",
                    "content_display": "Step 1 (Agent: A): hello",
                    "content_summary": "hello",
                }
            ]
        },
    }

    sim.apply_event(event)

    assert sim.knowledge_board.to_dict()["entries"] == event["knowledge_board"]["entries"]
    assert sim.knowledge_board.get_state() == ["Step 1 (Agent: A): hello"]


@pytest.mark.unit
def test_apply_event_replaces_graph_backend_kb_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.infra import config
    from src.sim.graph_knowledge_board import GraphKnowledgeBoard
    from src.sim.simulation import Simulation
    from tests.integration.knowledge_board.test_graph_backend import DummyDriver

    monkeypatch.setattr(config, "KNOWLEDGE_BOARD_BACKEND", "graph")
    monkeypatch.setattr("src.sim.simulation.GraphKnowledgeBoard", lambda: GraphKnowledgeBoard(driver=DummyDriver()))

    sim = Simulation([DummyAgent("A")])
    event = {
        "type": "agent_action",
        "agent_id": "A",
        "knowledge_board": {
            "entries": [
                {
                    "entry_id": "g1",
                    "step": 2,
                    "agent_id": "A",
                    "entry_type": "note",
                    "tags": [],
                    "reference_metadata": None,
                    "content_full": "graph hello",
                    "content_display": "Step 2 (Agent: A): graph hello",
                    "content_summary": "graph hello",
                }
            ]
        },
    }

    sim.apply_event(event)

    assert sim.knowledge_board.to_dict()["entries"] == event["knowledge_board"]["entries"]
    assert sim.knowledge_board.get_state() == ["Step 2 (Agent: A): graph hello"]
