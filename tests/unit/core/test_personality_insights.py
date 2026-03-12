import pytest

from src.agents.core.agent_state import AgentState, PersonalityTraits
from src.agents.core.personality_engine import ExperienceSignal, PersonalityEngine
from src.agents.core.personality_insights import (
    build_character_arc_summaries,
    personality_timeline,
    smooth_trait_deltas,
)


@pytest.mark.unit
def test_personality_timeline_merges_trait_and_lifecycle_events() -> None:
    engine = PersonalityEngine()
    state = AgentState(agent_id="a", name="Agent", traits=PersonalityTraits())
    engine.apply_experience_drift(state, ExperienceSignal(social_outcome=0.4), source="test")
    state.lifecycle_history.append({"step": 0, "from": "active", "to": "retired", "reason": "age"})

    timeline = personality_timeline(state.trait_transition_log, state.lifecycle_history)

    kinds = {entry["kind"] for entry in timeline}
    assert "personality_transition" in kinds
    assert "lifecycle_transition" in kinds


@pytest.mark.unit
def test_character_arc_summaries_capture_social_events() -> None:
    engine = PersonalityEngine()
    state = AgentState(agent_id="a", name="Agent", traits=PersonalityTraits())

    for _ in range(4):
        engine.apply_experience_drift(
            state,
            ExperienceSignal(social_outcome=0.7, conflict_outcome=-0.3),
            source="test",
        )

    summaries = build_character_arc_summaries(state.trait_transition_log, window_size=2)

    assert len(summaries) == 2
    assert summaries[-1]["social_event_counts"]["positive_social_outcomes"] >= 1
    assert summaries[-1]["top_trait_changes"]


@pytest.mark.unit
def test_hysteresis_suppresses_short_sign_flip_oscillation() -> None:
    engine = PersonalityEngine()
    state = AgentState(agent_id="a", name="Agent", traits=PersonalityTraits())

    for _ in range(6):
        engine.apply_experience_drift(
            state,
            ExperienceSignal(social_outcome=0.8),
            source="warmup",
        )

    smoothed = smooth_trait_deltas(
        {"trust_baseline": -0.001},
        recent_transitions=state.trait_transition_log.transitions,
        window_size=5,
        alpha=0.5,
        hysteresis_threshold=0.0025,
    )
    assert smoothed["trust_baseline"] == pytest.approx(0.0)


@pytest.mark.unit
def test_long_run_smoothing_keeps_oscillation_bounded() -> None:
    engine = PersonalityEngine()
    state = AgentState(agent_id="a", name="Agent", traits=PersonalityTraits())

    history: list[float] = []
    for idx in range(120):
        social = 0.9 if idx % 2 == 0 else -0.9
        engine.apply_experience_drift(
            state,
            ExperienceSignal(social_outcome=social, mood_trajectory=social * 0.2),
            source="long_run",
        )
        history.append(state.traits.trust_baseline)

    # The smoother should dampen alternating pressure into a bounded band.
    assert max(history) - min(history) < 0.2
