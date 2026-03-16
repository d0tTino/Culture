from __future__ import annotations

import pytest

from src.sim.analytics import compute_user_value_kpis

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

    assert report.narrative_continuity_score >= 0
    assert report.unresolved_conflict_count == 0
    assert report.cross_agent_interaction_diversity > 0
    assert report.user_intervention_rate > 0
    assert report.return_session_continuity == 1.0


def test_compute_user_value_kpis_emits_stagnation_alerts() -> None:
    events = [
        {"type": "agent_action", "step": 1, "agent_id": "a", "action_intent": "idle"},
        {"type": "agent_action", "step": 2, "agent_id": "a", "action_intent": "idle"},
        {"type": "agent_action", "step": 3, "agent_id": "a", "action_intent": "idle"},
    ]

    report = compute_user_value_kpis(events=events, knowledge_entries=[])

    assert "repetitive_intents" in report.stagnation_alerts
    assert "no_social_graph_change" in report.stagnation_alerts
