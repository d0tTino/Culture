import json
import sys
import types

import pytest

from tests.unit.interfaces.test_dashboard_backend_control import load_dashboard_backend


def _prepare_dashboard_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    if "src.sim.resource_manager" not in sys.modules:
        resource_manager = types.ModuleType("src.sim.resource_manager")
        resource_manager.BudgetCharger = object
        resource_manager.BudgetChecker = object
        resource_manager.HasResources = object
        resource_manager.ResourceManager = object
        resource_manager.TickCapper = object
        resource_manager.get_budget_charger = lambda: None
        resource_manager.get_budget_checker = lambda: None
        resource_manager.get_resource_manager = lambda: None
        resource_manager.get_tick_capper = lambda: None
        monkeypatch.setitem(sys.modules, "src.sim.resource_manager", resource_manager)


def _patch_metrics(db, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db.metrics, "get_du_per_1k_tokens", lambda: 1.5)
    monkeypatch.setattr(db.metrics, "get_llm_latency_p95", lambda: 450.0)
    monkeypatch.setattr(db.metrics, "get_coalition_count", lambda: 3)
    monkeypatch.setattr(db.metrics, "get_average_sentiment", lambda: 0.42)
    monkeypatch.setattr(db.metrics, "get_rag_hit_rate", lambda: 0.85)
    monkeypatch.setattr(db.metrics, "get_memory_retrievals", lambda: 80)
    monkeypatch.setattr(db.metrics, "get_memory_retrieval_errors", lambda: 20)
    monkeypatch.setattr(db.metrics, "get_llm_errors_total", lambda: 5)
    monkeypatch.setattr(db.metrics, "get_llm_calls_total", lambda: 50)


@pytest.mark.unit
def test_cost_metrics_data_includes_observability_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_dashboard_backend(monkeypatch)
    db = load_dashboard_backend()
    _patch_metrics(db, monkeypatch)

    payload = db._cost_metrics_data()

    assert payload["du_per_1k_tokens"] == 1.5
    assert payload["llm_latency_p95_ms"] == 450.0
    assert payload["coalition_count"] == 3
    assert payload["average_sentiment"] == 0.42
    assert payload["rag_hit_rate"] == 0.85
    assert payload["memory_retrievals_total"] == 80
    assert payload["memory_retrieval_errors_total"] == 20
    assert pytest.approx(payload["memory_retrieval_success_rate"]) == 0.8
    assert pytest.approx(payload["memory_retrieval_error_rate"]) == 0.2
    assert payload["llm_errors_total"] == 5
    assert pytest.approx(payload["llm_error_rate"]) == 0.1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_observability_metrics_serializes_extended_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_dashboard_backend(monkeypatch)
    db = load_dashboard_backend()
    _patch_metrics(db, monkeypatch)

    resp = await db.api_observability_metrics()
    payload = json.loads(resp.body)

    expected_keys = {
        "du_per_1k_tokens",
        "llm_latency_p95_ms",
        "coalition_count",
        "average_sentiment",
        "rag_hit_rate",
        "memory_retrievals_total",
        "memory_retrieval_errors_total",
        "memory_retrieval_success_rate",
        "memory_retrieval_error_rate",
        "llm_errors_total",
        "llm_error_rate",
    }

    assert expected_keys.issubset(payload.keys())
    assert payload["memory_retrieval_errors_total"] == 20
    assert payload["llm_errors_total"] == 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_character_arcs_returns_identity_and_personality_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_dashboard_backend(monkeypatch)
    db = load_dashboard_backend()

    state = types.SimpleNamespace(
        personality_transition_events=[{"cause": "experience_drift"}],
        identity_events=[{"to_state": "retired"}],
        trait_transition_log={
            "schema_version": 1,
            "seed_traits": {"trust_baseline": 0.5},
            "transitions": [
                {
                    "step": 2,
                    "cause": "experience_drift",
                    "source": "simulation.turn",
                    "max_step": 0.01,
                    "input_signals": {"social_outcome": 0.4},
                    "deltas": [
                        {
                            "trait": "trust_baseline",
                            "before": 0.5,
                            "proposed_delta": 0.01,
                            "bounded_delta": 0.01,
                            "after": 0.51,
                        }
                    ],
                    "resulting_traits": {"trust_baseline": 0.51},
                }
            ],
            "hash_chain": ["abc"],
        },
        lifecycle_history=[{"step": 3, "from": "active", "to": "retired", "reason": "test"}],
    )
    sim = types.SimpleNamespace(agents=[types.SimpleNamespace(agent_id="agent-1", state=state)])
    monkeypatch.setitem(db.DEFAULT_CONTEXT.sim_state, "simulation", sim)

    resp = await db.api_character_arcs()
    payload = json.loads(resp.body)

    assert payload["arcs"]["agent-1"]["personality"][0]["cause"] == "experience_drift"
    assert payload["arcs"]["agent-1"]["identity"][0]["to_state"] == "retired"
    assert payload["arcs"]["agent-1"]["timeline"]
    assert payload["arcs"]["agent-1"]["summaries"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_state_and_timeline_include_trait_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_dashboard_backend(monkeypatch)
    db = load_dashboard_backend()

    state = types.SimpleNamespace(
        model_dump=lambda: {"name": "Agent"},
        trait_transition_log={
            "schema_version": 1,
            "seed_traits": {"trust_baseline": 0.5},
            "transitions": [
                {
                    "step": 2,
                    "cause": "experience_drift",
                    "source": "simulation.turn",
                    "max_step": 0.01,
                    "input_signals": {"social_outcome": 0.4},
                    "deltas": [
                        {
                            "trait": "trust_baseline",
                            "before": 0.5,
                            "proposed_delta": 0.01,
                            "bounded_delta": 0.01,
                            "after": 0.51,
                        }
                    ],
                    "resulting_traits": {"trust_baseline": 0.51},
                }
            ],
            "hash_chain": ["abc"],
        },
        lifecycle_history=[{"step": 3, "from": "active", "to": "retired", "reason": "test"}],
    )
    sim = types.SimpleNamespace(agents=[types.SimpleNamespace(agent_id="agent-1", state=state)])
    monkeypatch.setitem(db.DEFAULT_CONTEXT.sim_state, "simulation", sim)

    state_payload = json.loads((await db.get_agent_state("agent-1")).body)
    assert state_payload["state"]["top_trait_changes"]
    assert state_payload["state"]["top_trait_changes"][0]["timeline_link"].startswith(
        "/api/agents/agent-1/personality_timeline"
    )

    timeline_payload = json.loads((await db.get_agent_personality_timeline("agent-1")).body)
    assert timeline_payload["timeline"]
    assert {item["kind"] for item in timeline_payload["timeline"]} == {
        "personality_transition",
        "lifecycle_transition",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_user_value_metrics_includes_kpis_and_periodic_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_dashboard_backend(monkeypatch)
    db = load_dashboard_backend()

    sim = types.SimpleNamespace(
        current_step=12,
        metrics=[{"_target_summary": {"novelty": {"status": "pass"}}, "_target_alerts": ["none"]}],
        knowledge_board=types.SimpleNamespace(
            to_snapshot=lambda: {
                "entries": [
                    {
                        "entry_id": "e1",
                        "step": 1,
                        "entry_type": "conflict",
                        "content_summary": "conflict",
                        "tags": ["conflict"],
                    }
                ]
            }
        ),
    )
    monkeypatch.setitem(db.DEFAULT_CONTEXT.sim_state, "simulation", sim)
    monkeypatch.setattr(
        db.event_log,
        "fetch_events",
        lambda after_step=0: [
            {"type": "agent_action", "step": 1, "agent_id": "a", "action_intent": "idle"},
            {"type": "human_command", "step": 2},
            {"type": "snapshot", "step": 3},
        ],
    )

    resp = await db.api_user_value_metrics()
    payload = json.loads(resp.body)

    assert payload["payload_version"] == 2
    assert payload["thresholds"] == {
        "min_novelty_score": 0.2,
        "min_interaction_diversity": 0.2,
        "max_repetitive_intents_ratio": 0.75,
        "min_social_graph_change_count": 1,
    }
    assert "narrative_continuity_score" in payload
    assert "unresolved_conflict_count" in payload
    assert "stagnation_alerts" in payload
    assert payload["periodic_summary"]["step"] == 12
