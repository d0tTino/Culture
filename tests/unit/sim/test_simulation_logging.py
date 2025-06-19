import logging
import sys

import pytest

pytestmark = pytest.mark.unit


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


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._state = DummyAgentState()
        self.turns = 0

    def get_id(self) -> str:
        return self.agent_id

    @property
    def state(self) -> DummyAgentState:
        return self._state

    def update_state(self, state: DummyAgentState) -> None:
        self._state = state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager=None,
        knowledge_board=None,
    ) -> dict:
        self.turns += 1
        self._state.ip += 1.0
        self._state.du += 2.0
        return {}


@pytest.mark.asyncio
async def test_logs_use_start_values(caplog: pytest.LogCaptureFixture) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.sim.simulation import Simulation

    agent = DummyAgent("A")
    sim = Simulation([agent])

    with caplog.at_level(logging.INFO):
        await sim.run_step(max_turns=1)

    messages = [rec.getMessage() for rec in caplog.records]
    assert any("IP: 1.0 (from 0.0)" in m for m in messages)
    assert any("DU: 2.0 (from 0.0)" in m for m in messages)
