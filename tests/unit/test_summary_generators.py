import pytest
from pytest import MonkeyPatch

from src.agents.dspy_programs.l1_summary_generator import L1SummaryGenerator
from src.agents.dspy_programs.l2_summary_generator import L2SummaryGenerator


@pytest.mark.unit
def test_l1_summary_generator_failsafe(monkeypatch: MonkeyPatch) -> None:
    gen = L1SummaryGenerator()
    # Patch l1_predictor to None to trigger fallback
    gen.l1_predictor = None
    result = gen.generate_summary(agent_role="Test", recent_events="events", current_mood="happy")
    # Accept either the fallback summary or the failsafe string
    assert "Failsafe" in result or result.startswith("As a Test with happy mood, I processed")


@pytest.mark.unit
def test_l2_summary_generator_failsafe(monkeypatch: MonkeyPatch) -> None:
    gen = L2SummaryGenerator()
    # Patch l2_predictor to None to trigger fallback/error
    gen.l2_predictor = None
    result = gen.generate_summary(
        agent_role="Test",
        l1_summaries_context="context",
        overall_mood_trend="trend",
        agent_goals="goals",
    )
    # L2 fallback returns empty string if DSPy not available, so patch dspy to None as well
    assert result == "" or "Failsafe" in result
