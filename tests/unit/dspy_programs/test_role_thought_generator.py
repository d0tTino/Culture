import pytest

pytestmark = pytest.mark.unit

from src.agents.dspy_programs import role_thought_generator as rtg


def test_get_failsafe_output_positional() -> None:
    assert rtg.get_failsafe_output("Analyst") == (
        "Failsafe: Unable to generate thought for role Analyst."
    )


def test_failsafe_role_thought_generator() -> None:
    generator = rtg.FailsafeRoleThoughtGenerator()
    result = generator("Facilitator", "context")
    assert getattr(result, "thought") == (
        "Failsafe: Unable to generate thought for role Facilitator."
    )


def test_generate_role_prefixed_thought_failsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rtg, "get_role_thought_generator", lambda: rtg.FailsafeRoleThoughtGenerator()
    )
    thought = rtg.generate_role_prefixed_thought("Researcher", "test situation")
    assert thought == "Failsafe: Unable to generate thought for role Researcher."
