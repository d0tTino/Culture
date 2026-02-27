import numpy as np
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
            engine.apply_experience_drift(state, signal)
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

    records = engine.apply_experience_drift(
        state,
        ExperienceSignal(social_outcome=0.5, governance_participation=0.5),
    )

    assert records
    assert state.trait_change_audit
    first = state.trait_change_audit[0]
    assert first["step"] == state.step_counter
    assert first["trait"]
    assert first["cause"] == "experience_drift"
    assert first["source"] == "simulation.turn"


@pytest.mark.unit
def test_role_transition_blend_uses_engine_audit_and_bounds() -> None:
    state = AgentState(agent_id="agent", name="Agent", traits=PersonalityTraits(openness=0.1))
    engine = PersonalityEngine()

    records = engine.apply_role_transition_blend(
        state,
        target_traits={"openness": 1.0},
        blend_ratio=1.0,
        max_step=0.03,
        source="test.role_transition",
    )

    assert records
    assert state.traits.openness == pytest.approx(0.13)
    assert records[0]["cause"] == "role_transition_blend"
    assert records[0]["source"] == "test.role_transition"


@pytest.mark.unit
def test_exogenous_intervention_uses_same_constraints() -> None:
    state = AgentState(agent_id="agent", name="Agent", traits=PersonalityTraits(assertiveness=0.5))
    engine = PersonalityEngine()

    records = engine.apply_exogenous_trait_intervention(
        state,
        {"assertiveness": 0.8},
        source="admin.console",
        max_step=0.05,
        cause="admin_edit",
    )

    assert state.traits.assertiveness == pytest.approx(0.55)
    assert records[0]["cause"] == "admin_edit"
    assert records[0]["source"] == "admin.console"


@pytest.mark.unit
def test_same_role_different_traits_diverge_action_distribution_over_many_turns() -> None:
    engine = PersonalityEngine()
    available_actions = [
        "propose_idea",
        "ask_clarification",
        "continue_collaboration",
        "idle",
    ]
    high_empathy = AgentState(
        agent_id="a",
        name="A",
        current_role="Facilitator",
        traits=PersonalityTraits(empathy=0.9, assertiveness=0.2),
    )
    low_empathy = AgentState(
        agent_id="b",
        name="B",
        current_role="Facilitator",
        traits=PersonalityTraits(empathy=0.2, assertiveness=0.9),
    )

    rng = np.random.default_rng(7)

    def sample_distribution(state: AgentState) -> dict[str, float]:
        counts = {action: 0 for action in available_actions}
        for _ in range(2000):
            bias = engine.action_biases(state, available_actions)
            logits = np.array([bias[action] for action in available_actions], dtype=float)
            probs = np.exp(logits) / np.exp(logits).sum()
            picked = rng.choice(available_actions, p=probs)
            counts[str(picked)] += 1
        return {action: count / 2000.0 for action, count in counts.items()}

    high_empathy_dist = sample_distribution(high_empathy)
    low_empathy_dist = sample_distribution(low_empathy)

    assert high_empathy_dist["continue_collaboration"] > low_empathy_dist[
        "continue_collaboration"
    ]
    assert low_empathy_dist["propose_idea"] > high_empathy_dist["propose_idea"]
