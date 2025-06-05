from __future__ import annotations

from typing_extensions import Self

from src.agents.dspy_programs.intent_selector import IntentSelectorProgram


class AgentController:
    """Minimal controller that delegates intent selection to DSPy."""

    def __init__(self: Self, lm: object | None = None) -> None:
        self.intent_selector = IntentSelectorProgram(lm=lm)

    def select_intent(self: Self, state: object | None = None) -> str:
        """Return the chosen intent from the DSPy program."""
        return self.intent_selector.run()
