import pytest
from pytest import MonkeyPatch

pytest.importorskip("dspy")

from src.agents.dspy_programs.relationship_updater import (
    FailsafeRelationshipUpdater,
    get_failsafe_output,
    get_relationship_updater,
    update_relationship,
)


@pytest.mark.unit
def test_failsafe_relationship_updater_returns_expected() -> None:
    updater = FailsafeRelationshipUpdater()
    result = updater(0.5, "summary", "persona1", "persona2", 0.1)
    assert hasattr(result, "new_relationship_score")
    assert hasattr(result, "relationship_change_rationale")
    assert result.new_relationship_score == 0.5
    assert "Failsafe" in result.relationship_change_rationale


@pytest.mark.unit
def test_get_failsafe_output_direct() -> None:
    result = get_failsafe_output(current_relationship_score=0.2)
    assert hasattr(result, "new_relationship_score")
    assert getattr(result, "new_relationship_score", None) == 0.2
    assert "Failsafe" in getattr(result, "relationship_change_rationale", "")


@pytest.mark.unit
def test_get_relationship_updater_fallback(monkeypatch: MonkeyPatch) -> None:
    # Simulate dspy import error to force fallback
    import sys

    monkeypatch.setitem(sys.modules, "dspy", None)
    # Should return FailsafeRelationshipUpdater
    updater = get_relationship_updater()
    assert isinstance(updater, FailsafeRelationshipUpdater)
    result = updater(0.1, "summary", "persona1", "persona2", 0.0)
    assert hasattr(result, "new_relationship_score")
    assert hasattr(result, "relationship_change_rationale")
    assert result.new_relationship_score == 0.1
    assert "Failsafe" in result.relationship_change_rationale


@pytest.mark.unit
def test_update_relationship_adjusts_strength() -> None:
    from src.agents.dspy_programs.relationship_updater import _RELATIONSHIPS

    _RELATIONSHIPS.clear()

    msg = update_relationship("a", "b", "friendship", 0.3)
    assert _RELATIONSHIPS["a"]["b"]["friendship"] == pytest.approx(0.3)
    assert "a to b" in msg

    msg = update_relationship("a", "b", "friendship", 0.8)
    # Should clamp to 1.0
    assert _RELATIONSHIPS["a"]["b"]["friendship"] == pytest.approx(1.0)
