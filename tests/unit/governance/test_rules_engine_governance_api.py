from src.governance.rules_engine import RulesEngine


def test_pre_action_check_maps_blocked_and_cost_outcomes() -> None:
    engine = RulesEngine()
    engine.materialize_from_proposal("ban move", proposer_id="a1", approved=True)
    engine.materialize_from_proposal("penalize gather by 2 ip", proposer_id="a1", approved=True)

    blocked = engine.pre_action_check("move")
    assert blocked.outcome == "blocked"
    assert blocked.allowed is False

    with_cost = engine.pre_action_check("gather")
    assert with_cost.outcome == "allowed_with_cost"
    assert with_cost.allowed is True
    assert with_cost.penalties


def test_pre_action_check_maps_vote_required_outcome() -> None:
    engine = RulesEngine()
    engine.materialize_from_proposal("require vote for build", proposer_id="a1", approved=True)

    outcome = engine.pre_action_check("build")
    assert outcome.outcome == "needs_vote"
    assert outcome.allowed is False


def test_proposal_workflow_and_read_apis() -> None:
    engine = RulesEngine()
    engine.proposal_workflow("ban dance", proposer_id="a1", approved=True)

    assert engine.current_rules()
    assert engine.passed_laws() == ["ban dance"]
    assert engine.pending_votes() == []

    record = engine.post_action_enforcement(
        engine.pre_action_check("dance"),
        agent_id="a2",
        step=3,
    )
    assert record["outcome"] == "blocked"
    assert engine.sanctions()
