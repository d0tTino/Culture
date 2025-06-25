from types import SimpleNamespace

import pytest

from src.agents.core import roles
from src.agents.core.agent_controller import AgentController
from src.agents.graphs import basic_agent_graph as bag


@pytest.mark.integration
def test_role_change_heavy_churn() -> None:
    state = SimpleNamespace(
        agent_id="churner",
        current_role=roles.ROLE_INNOVATOR,
        steps_in_current_role=5,
        role_change_cooldown=2,
        ip=50.0,
        role_change_ip_cost=1.0,
        role_history=[],
        last_action_step=0,
        short_term_memory=[],
        du=0.0,
        role_embedding=roles.ROLE_EMBEDDINGS[roles.ROLE_INNOVATOR],
        role_reputation={},
    )

    # Attempt role change before cooldown expires
    state.steps_in_current_role = 1
    ip_before = state.ip
    assert not bag.process_role_change(state, roles.ROLE_ANALYZER)
    assert state.ip == ip_before
    assert state.current_role == roles.ROLE_INNOVATOR

    controller = AgentController(state)
    initial_emb = list(state.role_embedding)
    initial_ip = state.ip
    cycle = [roles.ROLE_ANALYZER, roles.ROLE_FACILITATOR, roles.ROLE_INNOVATOR]

    for i in range(30):
        state.steps_in_current_role = state.role_change_cooldown
        new_role = cycle[i % len(cycle)]
        prev_ip = state.ip
        prev_emb = list(state.role_embedding)
        assert bag.process_role_change(state, new_role)
        assert state.current_role == new_role
        assert state.ip == pytest.approx(prev_ip - state.role_change_ip_cost)
        assert new_role in state.role_reputation
        controller.gossip_update(roles.ROLE_EMBEDDINGS[new_role], 1.0)
        assert state.role_embedding != prev_emb

    assert state.ip == pytest.approx(initial_ip - 30 * state.role_change_ip_cost)
    assert state.role_embedding != initial_emb
    assert len(state.role_history) == 30
