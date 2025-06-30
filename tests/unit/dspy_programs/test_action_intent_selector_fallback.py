import pytest
from typing_extensions import Self

dspy = pytest.importorskip("dspy")

from src.agents.dspy_programs import action_intent_selector


def test_action_intent_selector_failsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch load_optimized_program to always raise
    monkeypatch.setattr(
        action_intent_selector,
        "load_optimized_program",
        lambda: (_ for _ in ()).throw(Exception("Simulated load failure")),
    )
    program = action_intent_selector.ActionIntentSelector()
    result = program(context="A test context.", agent_role="Analyzer")
    assert "Failsafe" in result.intent
