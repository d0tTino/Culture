import random

import pytest

from src.agents.core.agent_state import AgentState, PersonalityTraits
from src.agents.core.personality_engine import ExperienceSignal, PersonalityEngine


@pytest.mark.unit
def test_experience_drift_persists_transition_events() -> None:
    state = AgentState(agent_id="agent", name="Agent", traits=PersonalityTraits())
    engine = PersonalityEngine()

    engine.apply_experience_drift(
        state,
        ExperienceSignal(social_outcome=0.8, conflict_outcome=0.1, mood_trajectory=0.2),
        max_step=0.01,
        source="test.turn",
    )

    assert state.personality_transition_events
    event = state.personality_transition_events[-1]
    assert event["cause"] == "experience_drift"
    assert event["source"] == "test.turn"
    assert event["resulting_traits"]["openness"] == pytest.approx(state.traits.openness)
    assert state.trait_transition_log.schema_version == 1
    assert state.trait_transition_log.verify_hash_chain()


@pytest.mark.unit
def test_replay_transitions_is_deterministic_for_identical_stream() -> None:
    rng = random.Random(17)
    signals = [
        ExperienceSignal(
            social_outcome=rng.uniform(-1.0, 1.0),
            conflict_outcome=rng.uniform(-1.0, 1.0),
            task_outcome=rng.uniform(-1.0, 1.0),
            mood_trajectory=rng.uniform(-1.0, 1.0),
            governance_participation=rng.uniform(-1.0, 1.0),
        )
        for _ in range(25)
    ]

    engine = PersonalityEngine()
    state = AgentState(agent_id="agent", name="Agent", traits=PersonalityTraits())
    initial_projection = engine.trait_projection(state)["raw"]

    for signal in signals:
        engine.apply_experience_drift(state, signal)

    replayed = engine.replay_transitions(initial_projection, state.trait_transition_log)
    final_projection = engine.trait_projection(state)["raw"]

    assert replayed == pytest.approx(final_projection)


@pytest.mark.unit
def test_drift_deltas_are_bounded_by_max_step_property() -> None:
    rng = random.Random(23)
    engine = PersonalityEngine()
    state = AgentState(agent_id="agent", name="Agent", traits=PersonalityTraits())

    for _ in range(50):
        max_step = rng.uniform(0.001, 0.03)
        engine.apply_experience_drift(
            state,
            ExperienceSignal(
                social_outcome=rng.uniform(-1.0, 1.0),
                conflict_outcome=rng.uniform(-1.0, 1.0),
                task_outcome=rng.uniform(-1.0, 1.0),
                mood_trajectory=rng.uniform(-1.0, 1.0),
                governance_participation=rng.uniform(-1.0, 1.0),
            ),
            max_step=max_step,
            source="property.test",
        )

        event = state.personality_transition_events[-1]
        for delta in event["deltas"]:
            assert abs(float(delta["bounded_delta"])) <= max_step + 1e-12


@pytest.mark.unit
def test_replay_log_is_deterministic_for_same_seed_and_events() -> None:
    rng = random.Random(7)
    stream = [
        ExperienceSignal(
            social_outcome=rng.uniform(-1.0, 1.0),
            conflict_outcome=rng.uniform(-1.0, 1.0),
            task_outcome=rng.uniform(-1.0, 1.0),
            mood_trajectory=rng.uniform(-1.0, 1.0),
            governance_participation=rng.uniform(-1.0, 1.0),
        )
        for _ in range(15)
    ]

    engine = PersonalityEngine()
    state_a = AgentState(agent_id="a", name="A", traits=PersonalityTraits())
    state_b = AgentState(agent_id="b", name="B", traits=PersonalityTraits())

    for signal in stream:
        engine.apply_experience_drift(state_a, signal, source="determinism.test")
        engine.apply_experience_drift(state_b, signal, source="determinism.test")

    assert state_a.trait_transition_log.hash_chain == state_b.trait_transition_log.hash_chain
    assert state_a.trait_transition_log.transitions == state_b.trait_transition_log.transitions
