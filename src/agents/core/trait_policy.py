from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .agent_state import PersonalityTraits

DEFAULT_TRAIT_POLICY_COEFFICIENTS: dict[str, float] = {
    "action_bias.openness": 0.35,
    "action_bias.analytical_focus": 0.30,
    "action_bias.empathy": 0.30,
    "action_bias.assertiveness": 0.25,
    "action_bias.adaptability": 0.20,
    "mood.sensitivity_weight": 0.60,
    "mood.resilience_weight": 0.40,
    "relationship.base_multiplier": 0.75,
    "relationship.trust_weight": 0.50,
    "relationship.targeted_weight": 1.00,
    "drift.social.trust_baseline": 0.01,
    "drift.social.empathy": 0.006,
    "drift.social.assertiveness_positive": 0.002,
    "drift.social.assertiveness_negative": 0.004,
    "drift.reflection.adaptability": 0.004,
    "drift.reflection.resilience": 0.003,
}


def _coefficient(
    coefficients: Mapping[str, float] | None,
    key: str,
) -> float:
    if coefficients is None:
        return DEFAULT_TRAIT_POLICY_COEFFICIENTS[key]
    return float(coefficients.get(key, DEFAULT_TRAIT_POLICY_COEFFICIENTS[key]))


def action_intent_biasing(
    traits: PersonalityTraits,
    available_actions: Sequence[str],
    coefficients: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return deterministic per-action trait bias scores in ``[-1, 1]``."""

    openness_w = _coefficient(coefficients, "action_bias.openness")
    analytical_w = _coefficient(coefficients, "action_bias.analytical_focus")
    empathy_w = _coefficient(coefficients, "action_bias.empathy")
    assertive_w = _coefficient(coefficients, "action_bias.assertiveness")
    adaptability_w = _coefficient(coefficients, "action_bias.adaptability")

    centered = {
        "openness": traits.openness - 0.5,
        "analytical_focus": traits.analytical_focus - 0.5,
        "empathy": traits.empathy - 0.5,
        "assertiveness": traits.assertiveness - 0.5,
        "adaptability": traits.adaptability - 0.5,
    }

    templates = {
        "propose_idea": (1.0, 0.2, 0.2, 0.8, 0.4),
        "perform_deep_analysis": (0.1, 1.0, 0.0, 0.1, 0.1),
        "ask_clarification": (0.2, 0.4, 0.7, 0.2, 0.2),
        "continue_collaboration": (0.3, 0.2, 0.8, 0.3, 0.4),
        "request_role_change": (0.5, 0.1, 0.1, 0.6, 1.0),
        "send_direct_message": (0.2, 0.0, 0.5, 0.4, 0.0),
        "idle": (-0.3, -0.2, -0.2, -0.4, -0.3),
    }

    biases: dict[str, float] = {}
    for action in available_actions:
        o, a, e, s, d = templates.get(str(action), (0.0, 0.0, 0.0, 0.0, 0.0))
        raw = (
            openness_w * centered["openness"] * o
            + analytical_w * centered["analytical_focus"] * a
            + empathy_w * centered["empathy"] * e
            + assertive_w * centered["assertiveness"] * s
            + adaptability_w * centered["adaptability"] * d
        )
        biases[str(action)] = max(-1.0, min(1.0, raw))
    return biases


def mood_update_multiplier(
    traits: PersonalityTraits,
    coefficients: Mapping[str, float] | None = None,
) -> float:
    """Return multiplier applied to sentiment before mood update rates."""

    sensitivity_weight = _coefficient(coefficients, "mood.sensitivity_weight")
    resilience_weight = _coefficient(coefficients, "mood.resilience_weight")
    multiplier = 1.0 + (
        sensitivity_weight * (traits.emotional_sensitivity - 0.5)
    ) - (resilience_weight * (traits.resilience - 0.5))
    return max(0.1, multiplier)


def relationship_update_sensitivity(
    traits: PersonalityTraits,
    *,
    is_targeted: bool,
    coefficients: Mapping[str, float] | None = None,
) -> float:
    """Return sentiment scaling factor used by relationship updates."""

    base = _coefficient(coefficients, "relationship.base_multiplier")
    trust_weight = _coefficient(coefficients, "relationship.trust_weight")
    targeted_weight = _coefficient(coefficients, "relationship.targeted_weight")
    targeted_factor = targeted_weight if is_targeted else 1.0
    return targeted_factor * (base + trust_weight * traits.trust_baseline)


def trait_drift_from_experience(
    experience_signals: Mapping[str, float],
    coefficients: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Map high-level experience signals to concrete trait drift deltas."""

    social = float(experience_signals.get("social_outcome", 0.0))
    mood_level = float(experience_signals.get("mood_level", 0.0))
    adaptability = float(experience_signals.get("adaptability", 0.5))

    social_assertiveness_coeff = _coefficient(
        coefficients,
        "drift.social.assertiveness_positive" if social >= 0 else "drift.social.assertiveness_negative",
    )
    assertiveness_drift = social_assertiveness_coeff * social

    return {
        "trust_baseline": _coefficient(coefficients, "drift.social.trust_baseline") * social,
        "empathy": _coefficient(coefficients, "drift.social.empathy") * social,
        "assertiveness": assertiveness_drift,
        "adaptability": _coefficient(coefficients, "drift.reflection.adaptability")
        * (1.0 - adaptability),
        "resilience": _coefficient(coefficients, "drift.reflection.resilience")
        * (0.5 - abs(mood_level)),
    }


def merge_trait_policy_coefficients(overrides: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Return a full coefficient map using defaults with optional overrides."""

    merged = dict(DEFAULT_TRAIT_POLICY_COEFFICIENTS)
    if overrides:
        for key, value in overrides.items():
            merged[str(key)] = float(value)
    return merged
