import pytest

from src.agents.core.agent_state import AgentLifecycleState
from src.sim.population_service import PopulationService

pytestmark = pytest.mark.unit


class DummyState:
    def __init__(self) -> None:
        self.agent_id = "A"
        self.current_role = type("Role", (), {"name": "Innovator"})()
        self.mood_level = 0.25
        self.relationships = {"B": 0.9}
        self.relationship_history = {}
        self.short_term_memory = []
        self.current_project_id = None
        self.goals = [{"goal": "x"}]
        self.role_embedding = [0.1, 0.2]
        self.reputation_score = 0.3
        self.lifecycle_state = AgentLifecycleState.ACTIVE
        self.lifecycle_history = []
        self.legacy_artifacts = {}
        self.memory_archival_policy = {}
        self.predecessor_id = None
        self.successor_id = None
        self.ip = 5.0
        self.du = 7.0
        self.inheritance = 0.0
        self.is_alive = True


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = DummyState()


class DummyKernel:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def emit_environment_event(self, event: dict[str, object]) -> None:
        self.events.append(event)


class DummyKnowledgeBoardService:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def post_event(self, **kwargs: object) -> bool:
        self.events.append(kwargs)
        return True


class DummySimulation:
    def __init__(self, agents: list[DummyAgent]) -> None:
        self.current_step = 10
        self.projects = {"p1": {"name": "Alpha", "members": ["A"]}}
        self.agents = agents
        self.event_kernel = DummyKernel()
        self.knowledge_board = object()
        self.knowledge_board_service = DummyKnowledgeBoardService()


@pytest.mark.asyncio
async def test_transition_from_active_generates_artifacts() -> None:
    service = PopulationService()
    agent = DummyAgent("A")
    other = DummyAgent("B")
    sim = DummySimulation([agent, other])

    result = await service.transition_lifecycle(
        simulation=sim,
        agent=agent,
        to_state=AgentLifecycleState.RETIRED,
        reason="manual",
    )

    assert result.changed is True
    assert agent.state.lifecycle_state == AgentLifecycleState.RETIRED
    assert agent.state.inheritance == 12.0
    assert agent.state.ip == 0.0
    assert agent.state.du == 0.0
    assert "epitaph" in agent.state.legacy_artifacts
    assert "unresolved_obligations" in agent.state.legacy_artifacts
    assert "inheritance_ledger" in agent.state.legacy_artifacts
    assert agent.state.memory_archival_policy["retain_summaries"] is True
    assert other.state.relationship_history["A"][-1] == (10, 0.0)


@pytest.mark.asyncio
async def test_register_successor_links_agents() -> None:
    service = PopulationService()
    predecessor = DummyAgent("A")
    successor = DummyAgent("B")
    predecessor.state.inheritance = 3.5
    sim = DummySimulation([predecessor, successor])

    payload = await service.register_successor(
        simulation=sim,
        predecessor=predecessor,
        successor=successor,
        inherit_role=True,
        inherit_context=True,
    )

    assert predecessor.state.successor_id == "B"
    assert successor.state.predecessor_id == "A"
    assert payload["relationship"] == "successor_of"
    assert "role" in payload["inherited"]
    assert "context" in payload["inherited"]
    assert payload["inherited"]["inheritance"] == 3.5
    assert successor.state.ip == 8.5
