from src.agents.core.agent_state import AgentLifecycleState
from src.sim.lifecycle_service import LifecycleService


class DummyState:
    def __init__(self) -> None:
        self.agent_id = "A"
        self.current_role = type("Role", (), {"name": "Innovator"})()
        self.mood_level = 0.25
        self.relationships = {"B": 0.9}
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


def test_transition_from_active_generates_artifacts() -> None:
    service = LifecycleService()
    agent = DummyAgent("A")
    projects = {"p1": {"name": "Alpha", "members": ["A"]}}

    result = service.transition(
        agent=agent,
        to_state=AgentLifecycleState.RETIRED,
        step=10,
        reason="manual",
        projects=projects,
    )

    assert result.changed is True
    assert agent.state.lifecycle_state == AgentLifecycleState.RETIRED
    assert agent.state.inheritance == 12.0
    assert agent.state.ip == 0.0
    assert agent.state.du == 0.0
    assert agent.state.legacy_artifacts["project_reassignment_tasks"]
    assert agent.state.memory_archival_policy["retain_summaries"] is True


def test_register_successor_links_agents() -> None:
    service = LifecycleService()
    predecessor = DummyAgent("A")
    successor = DummyAgent("B")

    payload = service.register_successor(
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
