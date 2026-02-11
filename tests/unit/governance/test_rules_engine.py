import pytest

from src.governance.rules_engine import RulesEngine

pytestmark = pytest.mark.unit


def test_materialize_rejected_proposal() -> None:
    engine = RulesEngine()
    result = engine.materialize_from_proposal(
        "no move",
        proposer_id="a1",
        approved=False,
    )
    assert result["decision"] == "rejected"
    assert result["rule_ids"] == []


def test_materialize_and_enforce_deny_and_penalty() -> None:
    engine = RulesEngine()
    deny = engine.materialize_from_proposal("no move", proposer_id="a1", approved=True)
    penalize = engine.materialize_from_proposal(
        "penalize talk by 2 ip", proposer_id="a1", approved=True
    )

    assert deny["decision"] == "accepted"
    assert penalize["decision"] == "accepted"

    denied = engine.evaluate_action("move")
    assert denied.allowed is False
    assert denied.decision == "rejected"
    assert denied.violated_rules

    overridden = engine.evaluate_action("talk")
    assert overridden.allowed is True
    assert overridden.decision == "overridden"
    assert overridden.penalties and overridden.penalties[0]["ip"] == 2.0
