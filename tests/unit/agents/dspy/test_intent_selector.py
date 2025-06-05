import pytest

from src.agents.dspy_programs.intent_selector import IntentSelectorProgram

dspy = pytest.importorskip("dspy")
if not hasattr(dspy, "Predict"):
    pytest.skip("dspy Predict not available", allow_module_level=True)


@pytest.mark.unit
def test_intent_selector_returns_static() -> None:
    program = IntentSelectorProgram()
    intent = program.run()
    assert intent == "PROPOSE_IDEA"
