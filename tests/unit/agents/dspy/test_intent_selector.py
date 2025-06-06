import pytest

try:
    from src.agents.dspy_programs.intent_selector import IntentSelectorProgram
except IndentationError:  # pragma: no cover - bad integration
    pytest.skip("dspy integration module invalid", allow_module_level=True)

dspy = pytest.importorskip("dspy")
if not hasattr(dspy, "Predict"):
    pytest.skip("dspy Predict not available", allow_module_level=True)


@pytest.mark.unit
def test_intent_selector_returns_static() -> None:
    program = IntentSelectorProgram()
    intent = program.run()
    assert intent == "PROPOSE_IDEA"
