from types import SimpleNamespace

import pytest

from src.agents.core import roles
from src.agents.graphs import basic_agent_graph as bag


@pytest.mark.integration
def test_role_change_allowed_by_similarity() -> None:
    state = SimpleNamespace(
        agent_id="agent1",
        current_role=roles.ROLE_INNOVATOR,
        steps_in_current_role=5,
        role_change_cooldown=0,
        ip=10.0,
        role_change_ip_cost=2.0,
        role_history=[],
        last_action_step=0,
        short_term_memory=[],
        du=0.0,
        role_embedding=roles.ROLE_EMBEDDINGS[roles.ROLE_INNOVATOR],
        role_reputation={},
    )
    assert bag.process_role_change(state, roles.ROLE_ANALYZER)
    assert state.current_role == roles.ROLE_ANALYZER
    assert roles.ROLE_ANALYZER in state.role_reputation


@pytest.mark.integration
def test_role_change_blocked_by_low_similarity() -> None:
    state = SimpleNamespace(
        agent_id="agent2",
        current_role=roles.ROLE_INNOVATOR,
        steps_in_current_role=5,
        role_change_cooldown=0,
        ip=10.0,
        role_change_ip_cost=2.0,
        role_history=[],
        last_action_step=0,
        short_term_memory=[],
        du=0.0,
        role_embedding=[0.0] * 8,
        role_reputation={},
    )
    assert not bag.process_role_change(state, roles.ROLE_ANALYZER)
    assert state.current_role == roles.ROLE_INNOVATOR
