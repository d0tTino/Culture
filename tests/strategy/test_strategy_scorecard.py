from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.sim.analytics import (
    STRATEGIC_ADHERENCE_PAYLOAD_VERSION,
    StrategicAdherenceThresholds,
    compute_strategic_adherence_scorecard,
    evaluate_release_gate,
    strategic_adherence_scorecard_from_dict,
)

pytestmark = pytest.mark.unit


CANONICAL_SCENARIOS: dict[str, dict[str, list[dict[str, object]]]] = {
    "social_emergence": {
        "events": [
            {"type": "agent_action", "step": 1, "agent_id": "a", "action_intent": "propose", "target_agent_id": "b", "content": "relationship formed", "latency_ms": 180},
            {"type": "agent_action", "step": 2, "agent_id": "b", "action_intent": "respond", "target_agent_id": "c", "content": "ally pact", "latency_ms": 200},
            {"type": "agent_action", "step": 3, "agent_id": "c", "action_intent": "coordinate", "target_agent_id": "a", "content": "coalition updated", "latency_ms": 220},
            {"type": "agent_action", "step": 4, "agent_id": "a", "action_intent": "reflect", "target_agent_id": "c", "latency_ms": 210},
        ],
        "knowledge_entries": [
            {"entry_id": "s1", "step": 1, "entry_type": "social_update", "content_summary": "New alliance formed", "tags": ["ally"], "memory_recall_quality": 0.84},
        ],
    },
    "user_intervention": {
        "events": [
            {"type": "human_command", "step": 1, "content": "pause and regroup", "latency_ms": 150},
            {"type": "command_ack", "step": 1, "agent_id": "a", "latency_ms": 140},
            {"type": "agent_action", "step": 2, "agent_id": "a", "action_intent": "replan", "target_agent_id": "b", "latency_ms": 190},
            {"type": "human_command", "step": 3, "content": "resume with safer routing", "latency_ms": 160},
            {"type": "command_ack", "step": 4, "agent_id": "b", "latency_ms": 170},
            {"type": "agent_action", "step": 4, "agent_id": "b", "action_intent": "reroute", "target_agent_id": "c", "latency_ms": 200},
        ],
        "knowledge_entries": [
            {"entry_id": "u1", "step": 4, "entry_type": "operator_note", "content_summary": "Operator instructions applied", "tags": ["command"], "memory_recall_quality": 0.8},
        ],
    },
    "governance": {
        "events": [
            {"type": "human_command", "step": 1, "content": "open vote", "latency_ms": 120},
            {"type": "governance_vote", "step": 2, "agent_id": "a", "action_intent": "vote_yes", "target_agent_id": "proposal-1", "recovery_success": True, "latency_ms": 180},
            {"type": "governance_vote", "step": 2, "agent_id": "b", "action_intent": "vote_yes", "target_agent_id": "proposal-1", "recovery_success": True, "latency_ms": 200},
            {"type": "agent_action", "step": 3, "agent_id": "c", "action_intent": "implement_policy", "target_agent_id": "a", "latency_ms": 210},
        ],
        "knowledge_entries": [
            {"entry_id": "g1", "step": 1, "entry_type": "conflict", "content_summary": "Conflict about resource usage", "tags": ["conflict"]},
            {"entry_id": "g2", "step": 3, "entry_type": "resolution", "content_summary": "Resolved through governance vote", "tags": ["resolution"], "parent_entry_id": "g1", "memory_recall_quality": 0.82},
        ],
    },
    "long_run_persistence": {
        "events": [
            {"type": "agent_action", "step": 1, "agent_id": "a", "action_intent": "document", "target_agent_id": "b", "latency_ms": 230, "memory_recall_quality": 0.86},
            {"type": "snapshot", "step": 2, "latency_ms": 110},
            {"type": "agent_action", "step": 3, "agent_id": "b", "action_intent": "retrieve", "target_agent_id": "a", "latency_ms": 210, "memory_recall_quality": 0.88},
            {"type": "snapshot", "step": 5, "latency_ms": 100},
            {"type": "agent_action", "step": 6, "agent_id": "c", "action_intent": "continue_plan", "target_agent_id": "a", "latency_ms": 220, "memory_recall_quality": 0.9},
        ],
        "knowledge_entries": [
            {"entry_id": "p1", "step": 1, "entry_type": "memory", "content_summary": "Initial plan recorded", "tags": ["memory"], "memory_recall_quality": 0.91},
            {"entry_id": "p2", "step": 6, "entry_type": "memory", "content_summary": "Plan continued after restore", "tags": ["memory"], "parent_entry_id": "p1", "memory_recall_quality": 0.92},
        ],
    },
}


def _combined_inputs() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    events: list[dict[str, object]] = []
    knowledge_entries: list[dict[str, object]] = []
    for scenario in CANONICAL_SCENARIOS.values():
        events.extend(scenario["events"])
        knowledge_entries.extend(scenario["knowledge_entries"])
    return events, knowledge_entries


@pytest.mark.parametrize("scenario_name", sorted(CANONICAL_SCENARIOS))
def test_canonical_strategy_scenarios_produce_expected_pillar_signal(scenario_name: str) -> None:
    scenario = CANONICAL_SCENARIOS[scenario_name]
    scorecard = compute_strategic_adherence_scorecard(
        events=scenario["events"],
        knowledge_entries=scenario["knowledge_entries"],
    )

    assert scorecard.payload_version == STRATEGIC_ADHERENCE_PAYLOAD_VERSION
    assert scorecard.aggregate_adherence_score > 0.0
    targeted_pillar = next(
        pillar for pillar in scorecard.pillar_scores if pillar.name == scenario_name
    )
    assert targeted_pillar.status in {"pass", "warning"}
    assert targeted_pillar.score >= targeted_pillar.threshold.warning_floor


def test_strategy_scorecard_writes_dashboard_json_artifact(tmp_path: Path) -> None:
    events, knowledge_entries = _combined_inputs()

    scorecard = compute_strategic_adherence_scorecard(
        events=events,
        knowledge_entries=knowledge_entries,
    )
    output_path = tmp_path / "strategic_adherence_scorecard.json"
    scorecard.write_json(output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["aggregate_adherence_score"] == scorecard.aggregate_adherence_score
    assert payload["aggregate_status"] == scorecard.aggregate_status
    assert {pillar["name"] for pillar in payload["pillar_scores"]} == {
        "social_emergence",
        "user_intervention",
        "governance",
        "long_run_persistence",
    }
    assert set(payload["additional_metrics"]) == {
        "latency_score",
        "recovery_success",
        "command_responsiveness",
        "memory_recall_quality",
    }


def test_release_gate_blocks_aggregate_regression_and_critical_pillar_drop() -> None:
    events, knowledge_entries = _combined_inputs()
    baseline = compute_strategic_adherence_scorecard(
        events=events,
        knowledge_entries=knowledge_entries,
    )

    degraded_events = list(events)
    degraded_events.extend(
        [
            {"type": "agent_action", "step": 20, "agent_id": "a", "action_intent": "idle", "target_agent_id": "b", "latency_ms": 950, "memory_recall_quality": 0.2},
            {"type": "agent_action", "step": 21, "agent_id": "a", "action_intent": "idle", "target_agent_id": "b", "latency_ms": 980, "memory_recall_quality": 0.2},
            {"type": "agent_action", "step": 22, "agent_id": "a", "action_intent": "idle", "target_agent_id": "b", "latency_ms": 990, "memory_recall_quality": 0.2, "recovery_success": False},
        ]
    )
    degraded_knowledge = list(knowledge_entries)
    degraded_knowledge.append(
        {"entry_id": "g3", "step": 22, "entry_type": "conflict", "content_summary": "Conflict reopened", "tags": ["conflict"], "parent_entry_id": None}
    )
    current = compute_strategic_adherence_scorecard(
        events=degraded_events,
        knowledge_entries=degraded_knowledge,
    )

    gate = evaluate_release_gate(current, baseline)

    assert not gate.passed
    assert gate.aggregate_delta < 0
    assert "aggregate_score_regressed" in gate.reasons
    assert "critical_pillar_drop" in gate.reasons
    assert {"governance", "long_run_persistence"}.intersection(gate.critical_pillar_regressions)


def test_release_gate_accepts_current_scorecard_against_committed_baseline_fixture(tmp_path: Path) -> None:
    fixture_path = Path("tests/strategy/fixtures/strategic_adherence_baseline.json")
    baseline_payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    baseline = strategic_adherence_scorecard_from_dict(baseline_payload)
    events, knowledge_entries = _combined_inputs()

    current = compute_strategic_adherence_scorecard(
        events=events,
        knowledge_entries=knowledge_entries,
        thresholds=StrategicAdherenceThresholds(),
    )
    output_path = tmp_path / "strategic_adherence_scorecard.json"
    current.write_json(output_path)

    gate = evaluate_release_gate(current, baseline)

    assert gate.passed
    assert gate.aggregate_delta >= 0
    assert gate.critical_pillar_regressions == ()
