from __future__ import annotations

import pytest

from src.agents.core.agent_lifecycle import AgentLifecycleState, is_valid_lifecycle_transition
from src.agents.core.agent_state import AgentState


@pytest.mark.unit
def test_agent_state_serialization_roundtrip_preserves_invariants() -> None:
    state = AgentState(
        agent_id="agent1",
        name="TestAgent",
        current_role="Innovator",
        step_counter=7,
        mood_level=0.25,
        role_history=[],
        mood_history=[],
    )

    serialized = state.to_dict()
    restored = AgentState.from_dict(serialized)

    assert restored.agent_id == "agent1"
    assert restored.name == "TestAgent"
    assert restored.current_role.name == "Innovator"
    assert restored.role_history == [(7, "Innovator")]
    assert restored.mood_history == [(7, 0.25)]
    assert restored.role_embedding == list(restored.current_role.embedding)
    assert restored.reputation_score == pytest.approx(restored.current_role.reputation)


@pytest.mark.unit
def test_agent_state_from_dict_coerces_string_role_and_excludes_runtime_refs() -> None:
    restored = AgentState.from_dict(
        {
            "agent_id": "agent2",
            "name": "FromDict",
            "current_role": "Strategist",
            "memory_store_manager": None,
        }
    )

    assert restored.current_role.name == "Strategist"
    assert "llm_client" not in restored.to_dict()
    assert "memory_store_manager" not in restored.to_dict()


@pytest.mark.unit
def test_lifecycle_transition_invariants() -> None:
    assert is_valid_lifecycle_transition(AgentLifecycleState.ACTIVE, AgentLifecycleState.RETIRED)
    assert is_valid_lifecycle_transition(AgentLifecycleState.ACTIVE, AgentLifecycleState.DECEASED)
    assert is_valid_lifecycle_transition(AgentLifecycleState.ARCHIVED, AgentLifecycleState.ACTIVE)

    assert not is_valid_lifecycle_transition(
        AgentLifecycleState.DECEASED,
        AgentLifecycleState.ACTIVE,
    )
    assert not is_valid_lifecycle_transition(
        AgentLifecycleState.RETIRED,
        AgentLifecycleState.DECEASED,
    )
