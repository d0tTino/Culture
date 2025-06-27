import random
from types import SimpleNamespace

import pytest

from src.agents.core import roles
from src.agents.core.role_embeddings import ROLE_EMBEDDINGS
from src.agents.core.roles import create_role_profile
from src.agents.graphs import basic_agent_graph as bag
from src.infra import ledger as ledger_mod


class DummyController:
    def __init__(self, state: object | None = None) -> None:
        self.state = state
        self.added = []

    def add_memory(self, *args: object) -> None:
        self.added.append(args)


def make_agent_state() -> SimpleNamespace:
    return SimpleNamespace(
        agent_id="a",
        current_role=create_role_profile("Innovator"),
        steps_in_current_role=5,
        role_change_cooldown=3,
        ip=10.0,
        role_change_ip_cost=2.0,
        role_history=[],
        last_action_step=0,
        short_term_memory=[],
        du=0.0,
        role_embedding=ROLE_EMBEDDINGS.role_vectors[roles.ROLE_INNOVATOR],
        role_reputation={},
    )


class DummyLedger:
    def log_change(self, *args: object, **kwargs: object) -> None:
        pass

    def calculate_gas_price(self, *args: object, **kwargs: object) -> tuple[float, float]:
        return (1.0, 0.0)


@pytest.mark.unit
def test_process_role_change_success(monkeypatch: pytest.MonkeyPatch) -> None:
    state = make_agent_state()
    monkeypatch.setattr(ledger_mod, "ledger", DummyLedger())
    assert bag.process_role_change(state, "Analyzer") is True
    assert state.current_role.name == "Analyzer"
    assert state.ip == 8.0


@pytest.mark.unit
@pytest.mark.require_ollama
def test_process_role_change_invalid_role() -> None:
    pytest.skip("skip in CI")
    state = make_agent_state()
    ledger_mod.ledger = DummyLedger()
    original = ROLE_EMBEDDINGS.nearest_role_from_embedding

    def fake_nearest(_emb: list[float], threshold: float = 0.7) -> tuple[str | None, float]:
        return None, 0.0

    ROLE_EMBEDDINGS.nearest_role_from_embedding = fake_nearest

    try:
        assert not bag.process_role_change(state, "UnknownRole")
    finally:
        ROLE_EMBEDDINGS.nearest_role_from_embedding = original
    assert state.current_role.name == "Innovator"


@pytest.mark.unit
def test_process_role_change_similarity_threshold() -> None:
    state = make_agent_state()
    ledger_mod.ledger = DummyLedger()
    state.role_embedding = [0.0] * 8
    assert not bag.process_role_change(state, "Analyzer")
    state.role_embedding = ROLE_EMBEDDINGS.role_vectors[roles.ROLE_ANALYZER]
    state.steps_in_current_role = 5
    state.ip = 10.0
    assert bag.process_role_change(state, "Analyzer")


@pytest.mark.unit
def test_update_state_node_role_change(monkeypatch: pytest.MonkeyPatch) -> None:
    state = make_agent_state()
    controller = DummyController(state)
    monkeypatch.setattr(ledger_mod, "ledger", DummyLedger())
    monkeypatch.setattr(bag, "AgentController", lambda s: controller)
    monkeypatch.setattr(random, "random", lambda: 0.9)

    output = bag.update_state_node(
        {
            "agent_id": "a",
            "simulation_step": 1,
            "structured_output": SimpleNamespace(
                action_intent="propose_idea", requested_role_change="Analyzer"
            ),
            "state": state,
        }
    )

    assert state.current_role.name == "Analyzer"
    assert controller.added[0][0].startswith("Changed role")
    assert output["data_units"] == int(state.du)


@pytest.mark.unit
def test_route_helpers() -> None:
    out_broadcast = bag.route_broadcast_decision(
        {"structured_output": SimpleNamespace(message_content="hi")}
    )
    assert out_broadcast == "broadcast"
    out_exit = bag.route_broadcast_decision({"structured_output": None})
    assert out_exit == "exit"

    agent = make_agent_state()
    agent.relationships = {"b": 0.1}
    out_rel = bag.route_relationship_context({"state": agent})
    assert out_rel == "has_relationships"
    assert (
        bag.route_relationship_context({"state": SimpleNamespace(relationships={})})
        == "no_relationships"
    )

    intent_out = bag.route_action_intent(
        {"structured_output": SimpleNamespace(action_intent="join_project")}
    )
    assert intent_out == "handle_join_project"


@pytest.mark.unit
def test_compile_agent_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = "compiled"
    monkeypatch.setattr(bag, "build_graph", lambda: compiled)
    assert bag.compile_agent_graph() == compiled
