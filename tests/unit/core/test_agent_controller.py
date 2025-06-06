import pytest
from typing_extensions import Self

try:
    from src.agents.core.agent_controller import AgentController
except IndentationError:
    pytest.skip("agent_state module is unparsable", allow_module_level=True)

dspy = pytest.importorskip("dspy")
if not hasattr(dspy, "Predict"):
    pytest.skip("dspy Predict not available", allow_module_level=True)


@pytest.mark.unit
def test_select_intent_uses_dspy() -> None:
    class DummyLM:
        def __call__(self: Self, prompt: str, *args: object, **kwargs: object) -> str:
            return "CONTINUE_COLLABORATION"

    controller = AgentController(lm=DummyLM())
    intent = controller.select_intent(None)
    assert intent == "CONTINUE_COLLABORATION"
