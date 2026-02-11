import pytest

from src.agents.core.agent_state import PersonalityTraits
from src.agents.core.trait_policy import (
    action_intent_biasing,
    mood_update_multiplier,
    relationship_update_sensitivity,
    trait_drift_from_experience,
)


@pytest.mark.unit
def test_action_intent_biasing_returns_known_action_biases() -> None:
    traits = PersonalityTraits(
        openness=0.9,
        analytical_focus=0.8,
        empathy=0.7,
        assertiveness=0.8,
        adaptability=0.9,
    )
    biases = action_intent_biasing(
        traits,
        ["propose_idea", "perform_deep_analysis", "idle"],
    )

    assert set(biases.keys()) == {"propose_idea", "perform_deep_analysis", "idle"}
    assert biases["propose_idea"] > biases["idle"]
    assert all(-1.0 <= score <= 1.0 for score in biases.values())


@pytest.mark.unit
def test_mood_update_multiplier_reacts_to_traits() -> None:
    calm = PersonalityTraits(emotional_sensitivity=0.2, resilience=0.9)
    reactive = PersonalityTraits(emotional_sensitivity=0.9, resilience=0.2)

    assert mood_update_multiplier(reactive) > mood_update_multiplier(calm)


@pytest.mark.unit
def test_relationship_update_sensitivity_scales_targeted_and_trust() -> None:
    low_trust = PersonalityTraits(trust_baseline=0.2)
    high_trust = PersonalityTraits(trust_baseline=0.9)

    assert relationship_update_sensitivity(high_trust, is_targeted=False) > relationship_update_sensitivity(
        low_trust,
        is_targeted=False,
    )
    assert relationship_update_sensitivity(high_trust, is_targeted=True) >= relationship_update_sensitivity(
        high_trust,
        is_targeted=False,
    )


@pytest.mark.unit
def test_trait_drift_from_experience_emits_expected_keys() -> None:
    drift = trait_drift_from_experience(
        {
            "social_outcome": -0.8,
            "mood_level": 0.2,
            "adaptability": 0.4,
        }
    )

    assert set(drift) == {
        "trust_baseline",
        "empathy",
        "assertiveness",
        "adaptability",
        "resilience",
    }
    assert drift["trust_baseline"] < 0
