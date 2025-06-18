from __future__ import annotations

from collections.abc import Callable
from typing import Any

from typing_extensions import Self

import dspy_ai as dspy  # type: ignore


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

    def __init__(self: Self, fn: Callable[[str | None], str | list[str]]) -> None:
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


# dspy lacks type hints, so Signature resolves to Any
class IntentPrompt(dspy.Signature):
    question = dspy.InputField()
    intent = dspy.OutputField()


class IntentSelectorProgram:
    """Minimal DSPy program that selects an intent."""

    def __init__(
        self: Self,
        lm: Any | Callable[[str | None], str | list[str]] | None = None,
    ) -> None:
        if lm is None:
            self.lm = _StubLM()
        elif isinstance(lm, dspy.LM):
            self.lm = lm
        elif callable(lm):
            self.lm = _CallableLM(lm)
        else:
            raise TypeError("lm must be a dspy.LM instance or callable")

        dspy.settings.configure(lm=self.lm)
        self._predict = dspy.Predict(IntentPrompt)

    def run(self: Self) -> str:
        """Return the chosen intent, falling back to raw LM output on errors."""
        try:
            result = self._predict(question="What proposal should the agent make?")
        except Exception as e:  # pragma: no cover - DSPy parsing errors
            raw: Any = self.lm("What proposal should the agent make?")
            if isinstance(raw, list):
                raw = raw[0]
            return str(raw).strip()

        return getattr(result, "intent", str(result).strip())
