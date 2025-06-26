from copy import deepcopy

import pytest

from src.agents.core.agent_controller import AgentController
from src.agents.core.agent_state import AgentState
from src.agents.core.role_embeddings import ROLE_EMBEDDINGS
from src.agents.core.roles import create_default_role_profiles


@pytest.fixture(autouse=True)
def _reset_role_embeddings() -> None:
    orig_vectors = deepcopy(ROLE_EMBEDDINGS.role_vectors)
    orig_rep = ROLE_EMBEDDINGS.reputation.copy()
    yield
    ROLE_EMBEDDINGS.role_vectors = orig_vectors
    ROLE_EMBEDDINGS.reputation = orig_rep


@pytest.mark.unit
def test_role_embedding_initialized() -> None:
    s1 = AgentState(agent_id="a1", name="A", current_role="Facilitator")
    s2 = AgentState(agent_id="a2", name="B", current_role="Analyzer")
    assert len(s1.role_embedding) == 8
    assert len(s2.role_embedding) == 8
    assert s1.role_embedding != s2.role_embedding


@pytest.mark.unit
def test_gossip_update_changes_embedding() -> None:
    state = AgentState(agent_id="a", name="A")
    controller = AgentController(state)
    original = list(state.role_embedding)
    other = [v + 0.5 for v in original]
    controller.gossip_update(other, 1.0)
    assert state.role_embedding != original


@pytest.mark.unit
def test_role_prompt_contains_embedding() -> None:
    state = AgentState(agent_id="a", name="A")
    state.reputation["b"] = 0.5
    prompt = state.role_prompt
    assert isinstance(prompt, str)
    assert f"{state.role_embedding[0]:.2f}" in prompt


@pytest.mark.unit
def test_create_default_profiles() -> None:
    profiles = create_default_role_profiles()
    assert set(profiles.keys())
    for name, prof in profiles.items():
        assert prof.name == name
        assert len(prof.embedding) == 8
        assert prof.reputation == 0.0


@pytest.mark.unit
def test_gossip_updates_reputation() -> None:
    state = AgentState(agent_id="a", name="A")
    controller = AgentController(state)
    other = [v + 0.5 for v in state.role_embedding]
    controller.gossip_update(other, 1.0)
    assert state.reputation_score != 0.0


@pytest.mark.unit
def test_global_gossip_updates_manager() -> None:
    state = AgentState(agent_id="a", name="A")
    controller = AgentController(state)
    other = [v + 0.3 for v in state.role_embedding]
    role, _ = ROLE_EMBEDDINGS.nearest_role_from_embedding(other)
    assert role is not None
    original_vec = deepcopy(ROLE_EMBEDDINGS.role_vectors[role])
    original_rep = ROLE_EMBEDDINGS.reputation[role]
    controller.gossip_update(other, 1.0)
    assert ROLE_EMBEDDINGS.role_vectors[role] != original_vec
    assert ROLE_EMBEDDINGS.reputation[role] != original_rep
