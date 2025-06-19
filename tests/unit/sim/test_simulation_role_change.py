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
    def __init__(self, agent_id: str, change_on_first_turn: bool = False) -> None:
        self.agent_id = agent_id
        self._state = DummyAgentState()
        self.change_on_first_turn = change_on_first_turn
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
        if self.change_on_first_turn:
            self._state.short_term_memory.append({"type": "role_change", "step": simulation_step})
            self.change_on_first_turn = False
        return {}


@pytest.mark.asyncio
async def test_role_change_grants_extra_turn() -> None:
    import sys

    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.sim.simulation import Simulation

    agent_a = DummyAgent("A", change_on_first_turn=True)
    agent_b = DummyAgent("B")
    sim = Simulation([agent_a, agent_b])

    turns = await sim.run_step(max_turns=2)

    assert turns == 2
    assert agent_a.turns == 2
    assert agent_b.turns == 0
    assert sim.current_agent_index == 1
