import asyncio
import sys
import types

import pytest

# Stub optional heavy dependencies before importing simulation
neo4j_stub = types.ModuleType("neo4j")
neo4j_stub.GraphDatabase = object  # type: ignore[attr-defined]
neo4j_stub.Driver = object  # type: ignore[attr-defined]
sys.modules.setdefault("neo4j", neo4j_stub)
sys.modules.setdefault("neo4j.exceptions", types.ModuleType("neo4j.exceptions"))
sys.modules.setdefault("weaviate", types.ModuleType("weaviate"))
sys.modules.setdefault("weaviate.classes", types.ModuleType("weaviate.classes"))

from src.agents.core.agent_state import AgentActionIntent
from src.interfaces import dashboard_backend as db
from src.sim import simulation as sim_module


class DummyAgentState:
    def __init__(self) -> None:
        self.ip = 0.0
        self.du = 0.0
        self.short_term_memory = []
        self.messages_sent_count = 0
        self.last_message_step = None
        self.collective_ip = 0.0
        self.collective_du = 0.0


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = DummyAgentState()
        self.received: list[dict] | None = None

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, state: DummyAgentState) -> None:
        self.state = state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager=None,
        knowledge_board=None,
    ) -> dict:
        self.received = (
            environment_perception.get("perceived_messages", []) if environment_perception else []
        )
        return {
            "message_content": "ack",
            "message_recipient_id": None,
            "action_intent": AgentActionIntent.CONTINUE_COLLABORATION.value,
        }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_discord_message_triggers_agent_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    q_events: asyncio.Queue[db.SimulationEvent | None] = asyncio.Queue()

    monkeypatch.setattr(sim_module, "event_queue", q_events)
    monkeypatch.setattr(db, "event_queue", q_events)

    agent = DummyAgent("A")
    sim = sim_module.Simulation([agent])

    await q_events.put(db.SimulationEvent(event_type="broadcast", data={"content": "hello"}))
    await asyncio.sleep(0.05)

    await sim.run_step(1)
    await asyncio.sleep(0.05)

    assert agent.received and agent.received[0]["content"] == "hello"
    assert (
        sim.pending_messages_for_next_round
        and sim.pending_messages_for_next_round[0]["content"] == "ack"
    )

    sim.close()
