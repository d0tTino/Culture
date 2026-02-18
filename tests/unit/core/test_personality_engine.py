import pytest

from src.agents.core.agent_state import AgentState, PersonalityTraits
from src.agents.core.personality_engine import ExperienceSignal, PersonalityEngine


@pytest.mark.unit
def test_personality_engine_produces_deterministic_trajectory() -> None:
    engine = PersonalityEngine()
    sequence = [
        ExperienceSignal(
            social_outcome=0.4,
            conflict_outcome=-0.1,
            task_outcome=0.3,
            mood_trajectory=0.2,
            governance_participation=0.1,
        ),
        ExperienceSignal(
            social_outcome=-0.2,
            conflict_outcome=0.1,
            task_outcome=0.0,
            mood_trajectory=-0.3,
            governance_participation=0.0,
        ),
        ExperienceSignal(
            social_outcome=0.1,
            conflict_outcome=0.0,
            task_outcome=0.2,
            mood_trajectory=0.0,
            governance_participation=0.2,
        ),
    ]

    def run_sequence() -> tuple[float, ...]:
        state = AgentState(
            agent_id="agent",
            name="Agent",
            traits=PersonalityTraits(),
        )
        for signal in sequence:
            engine.update_traits(state, signal)
        return (
            state.traits.trust_baseline,
            state.traits.empathy,
            state.traits.assertiveness,
            state.traits.adaptability,
            state.traits.resilience,
        )

    assert run_sequence() == pytest.approx(run_sequence())


@pytest.mark.unit
def test_personality_engine_appends_trait_audit_records() -> None:
    state = AgentState(agent_id="agent", name="Agent", traits=PersonalityTraits())
    engine = PersonalityEngine()

    records = engine.update_traits(
        state,
        ExperienceSignal(social_outcome=0.5, governance_participation=0.5),
    )

    assert records
    assert state.trait_change_audit
    first = state.trait_change_audit[0]
    assert first["step"] == state.step_counter
    assert first["trait"]
    assert "social_outcome" in first
    assert "governance_participation" in first
