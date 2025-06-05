from __future__ import annotations

# mypy: ignore-errors
from typing_extensions import Self

from src.infra.dspy_ollama_integration import dspy


class _StubLM(dspy.LM):
    """Deterministic LM returning a fixed intent for tests."""

    def __init__(self: Self) -> None:
        super().__init__(model="stub-model")

    def __call__(
        self: Self, prompt: str | None = None, *args: object, **kwargs: object
    ) -> str:
        return "PROPOSE_IDEA"


INTENTS = ["PROPOSE_IDEA", "CONTINUE_COLLABORATION"]


class IntentPrompt(dspy.Signature):
    question = dspy.InputField()
    intent = dspy.OutputField()


class IntentSelectorProgram:
    """Minimal DSPy program that selects an intent."""

    def __init__(self: Self, lm: dspy.LM | None = None) -> None:
        self.lm = lm or _StubLM()
        dspy.settings.configure(lm=self.lm)
        self._predict = dspy.Predict(IntentPrompt)

    def run(self: Self) -> str:
        result = self._predict(question="What proposal should the agent make?")
        return getattr(result, "intent", "PROPOSE_IDEA")
