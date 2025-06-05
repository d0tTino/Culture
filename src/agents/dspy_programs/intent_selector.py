from __future__ import annotations

# mypy: ignore-errors
from collections.abc import Callable

from typing_extensions import Self

from src.infra.dspy_ollama_integration import dspy


class _StubLM(dspy.LM):
    """Deterministic LM returning a fixed intent for tests."""

    def __init__(self: Self) -> None:
        super().__init__(model="stub-model")

    def __call__(
        self: Self, prompt: str | None = None, *args: object, **kwargs: object
    ) -> list[str]:
        # DSPy expects a list of completion strings
        return ['{"intent": "PROPOSE_IDEA"}']


class _CallableLM(dspy.LM):
    """Wrap a simple callable so it can be used as a DSPy LM."""

    def __init__(self: Self, fn: Callable[[str | None], object]) -> None:
        super().__init__(model="callable-lm")
        self.fn = fn

    def __call__(
        self: Self, prompt: str | None = None, *args: object, **kwargs: object
    ) -> list[str]:
        result = self.fn(prompt or "")
        if isinstance(result, list):
            return [str(r) for r in result]
        return [str(result)]


INTENTS = ["PROPOSE_IDEA", "CONTINUE_COLLABORATION"]


class IntentPrompt(dspy.Signature):
    question = dspy.InputField()
    intent = dspy.OutputField()


class IntentSelectorProgram:
    """Minimal DSPy program that selects an intent."""

    def __init__(self: Self, lm: dspy.LM | Callable[[str | None], object] | None = None) -> None:
        if lm is None:
            self.lm = _StubLM()
        elif isinstance(lm, dspy.LM):  # type: ignore[misc]
            self.lm = lm
        else:
            self.lm = _CallableLM(lm)

        dspy.settings.configure(lm=self.lm)
        self._predict = dspy.Predict(IntentPrompt)

    def run(self: Self) -> str:
        result = self._predict(question="What proposal should the agent make?")
        return getattr(result, "intent", "PROPOSE_IDEA")
