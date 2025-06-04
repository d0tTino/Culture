import pytest
from pytest import MonkeyPatch
from typing_extensions import Self

pytest.importorskip("dspy")

from src.agents.dspy_programs import action_intent_selector


def test_action_intent_selector_failsafe(monkeypatch: MonkeyPatch) -> None:
    # Patch load_optimized_program to always raise
    monkeypatch.setattr(
        action_intent_selector,
        "load_optimized_program",
        lambda: (_ for _ in ()).throw(Exception("Simulated load failure")),
    )

    # Patch select_action_intent_module to a dummy callable that also raises
    class Dummy:
        def __call__(self: Self, *a: object, **k: object) -> None:
            raise Exception("Simulated base failure")

    monkeypatch.setattr(action_intent_selector, "select_action_intent_module", Dummy())
    # Patch configure_dspy_with_ollama to raise
    monkeypatch.setattr(
        action_intent_selector,
        "configure_dspy_with_ollama",
        lambda: (_ for _ in ()).throw(Exception("Simulated DSPy config failure")),
    )
    # Call get_optimized_action_selector and verify it returns the failsafe
    selector = action_intent_selector.get_optimized_action_selector()
    result = selector(
        agent_role="Facilitator",
        current_situation="Test",
        agent_goal="Test goal",
        available_actions=["idle"],
    )
    assert hasattr(result, "chosen_action_intent")
    assert result.chosen_action_intent == "idle"
    assert "Failsafe" in result.justification_thought
