from __future__ import annotations

import pytest

from src.sim.analytics import (
    USER_VALUE_KPI_PAYLOAD_VERSION,
    SimulationStagnationThresholds,
    compute_user_value_kpis,
)

pytestmark = pytest.mark.unit


def test_compute_user_value_kpis_from_events_and_board_entries() -> None:
    events = [
        {"type": "agent_action", "step": 1, "agent_id": "a", "action_intent": "propose", "target_agent_id": "b"},
        {"type": "agent_action", "step": 2, "agent_id": "b", "action_intent": "respond", "target_agent_id": "a"},
        {"type": "human_command", "step": 3},
        {"type": "snapshot", "step": 4},
        {"type": "snapshot", "step": 8},
    ]
    entries = [
        {"entry_id": "c1", "step": 1, "entry_type": "conflict", "content_summary": "Conflict started", "tags": ["conflict"], "parent_entry_id": None},
        {"entry_id": "r1", "step": 2, "entry_type": "resolution", "content_summary": "resolved", "tags": ["resolution"], "parent_entry_id": "c1"},
    ]

    report = compute_user_value_kpis(events=events, knowledge_entries=entries)

    assert report.payload_version == USER_VALUE_KPI_PAYLOAD_VERSION
    assert report.thresholds == SimulationStagnationThresholds()
    assert report.narrative_continuity_score >= 0
    assert report.unresolved_conflict_count == 0
    assert report.cross_agent_interaction_diversity > 0
    assert report.user_intervention_rate > 0
    assert report.return_session_continuity == 1.0


@pytest.mark.parametrize(
    ("thresholds", "events", "expected_alert", "unexpected_alerts"),
    [
        (
            SimulationStagnationThresholds(min_novelty_score=0.6, min_interaction_diversity=0.0, max_repetitive_intents_ratio=1.0, min_social_graph_change_count=0),
            [
                {"type": "agent_action", "step": 1, "agent_id": "a", "action_intent": "idle", "target_agent_id": "b"},
                {"type": "agent_action", "step": 2, "agent_id": "b", "action_intent": "idle", "target_agent_id": "a"},
                {"type": "agent_action", "step": 3, "agent_id": "a", "action_intent": "explore", "target_agent_id": "b"},
                {"type": "agent_action", "step": 4, "agent_id": "b", "action_intent": "explore", "target_agent_id": "a", "content": "relationship formed"},
            ],
            "low_novelty",
            {"low_interaction_diversity", "repetitive_intents", "no_social_graph_change"},
        ),
        (
            SimulationStagnationThresholds(min_novelty_score=0.0, min_interaction_diversity=0.6, max_repetitive_intents_ratio=1.0, min_social_graph_change_count=0),
            [
                {"type": "agent_action", "step": 1, "agent_id": "a", "action_intent": "propose", "target_agent_id": "b"},
                {"type": "agent_action", "step": 2, "agent_id": "a", "action_intent": "explore", "target_agent_id": "b"},
                {"type": "agent_action", "step": 3, "agent_id": "c", "action_intent": "respond", "target_agent_id": "a", "content": "ally shift"},
                {"type": "agent_action", "step": 4, "agent_id": "b", "action_intent": "reflect", "target_agent_id": "a"},
            ],
            "low_interaction_diversity",
            {"low_novelty", "repetitive_intents", "no_social_graph_change"},
        ),
        (
            SimulationStagnationThresholds(min_novelty_score=0.0, min_interaction_diversity=0.0, max_repetitive_intents_ratio=0.6, min_social_graph_change_count=0),
            [
                {"type": "agent_action", "step": 1, "agent_id": "a", "action_intent": "idle", "target_agent_id": "b"},
                {"type": "agent_action", "step": 2, "agent_id": "b", "action_intent": "idle", "target_agent_id": "a"},
                {"type": "agent_action", "step": 3, "agent_id": "a", "action_intent": "idle", "target_agent_id": "b", "content": "coalition updated"},
                {"type": "agent_action", "step": 4, "agent_id": "b", "action_intent": "explore", "target_agent_id": "a"},
            ],
            "repetitive_intents",
            {"low_novelty", "low_interaction_diversity", "no_social_graph_change"},
        ),
        (
            SimulationStagnationThresholds(min_novelty_score=0.0, min_interaction_diversity=0.0, max_repetitive_intents_ratio=1.0, min_social_graph_change_count=1),
            [
                {"type": "agent_action", "step": 1, "agent_id": "a", "action_intent": "propose", "target_agent_id": "b"},
                {"type": "agent_action", "step": 2, "agent_id": "b", "action_intent": "respond", "target_agent_id": "a"},
            ],
            "no_social_graph_change",
            {"low_novelty", "low_interaction_diversity", "repetitive_intents"},
        ),
    ],
)
def test_compute_user_value_kpis_threshold_crossings_are_independent(
    thresholds: SimulationStagnationThresholds,
    events: list[dict[str, object]],
    expected_alert: str,
    unexpected_alerts: set[str],
) -> None:
    report = compute_user_value_kpis(events=events, knowledge_entries=[], thresholds=thresholds)

    assert expected_alert in report.stagnation_alerts
    assert unexpected_alerts.isdisjoint(report.stagnation_alerts)


def test_compute_user_value_kpis_serializes_payload_version_and_thresholds() -> None:
    report = compute_user_value_kpis(events=[], knowledge_entries=[])

    payload = report.as_dict()

    assert payload["payload_version"] == USER_VALUE_KPI_PAYLOAD_VERSION
    assert payload["thresholds"] == {
        "min_novelty_score": 0.2,
        "min_interaction_diversity": 0.2,
        "max_repetitive_intents_ratio": 0.75,
        "min_social_graph_change_count": 1,
    }
