import pytest
from pytest import MonkeyPatch

pytest.importorskip("dspy")

import dsp

from src.agents.dspy_programs.role_thought_generator import (
    FailsafeRoleThoughtGenerator,
    generate_role_prefixed_thought,
)


@pytest.mark.unit
@pytest.mark.dspy
def test_failsafe_role_thought_generator_returns_expected() -> None:
    generator = FailsafeRoleThoughtGenerator()
    result = generator("Leader", "context")
    assert hasattr(result, "thought")
    assert "Failsafe" in result.thought


@pytest.mark.unit
@pytest.mark.dspy
def test_generate_role_prefixed_thought_uses_generator(monkeypatch: MonkeyPatch) -> None:
    def dummy(role_name: str, context: str) -> object:
        return type(
            "Dummy",
            (),
            {"thought": f"As a {role_name}, thinking about {context}"},
        )()

    monkeypatch.setattr(
        "src.agents.dspy_programs.role_thought_generator.get_role_thought_generator",
        lambda: dummy,
    )

    result = generate_role_prefixed_thought("Builder", "blueprint")
    assert result == "As a Builder, thinking about blueprint"


@pytest.mark.unit
@pytest.mark.dspy
def test_generate_role_prefixed_thought_noncallable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.agents.dspy_programs.role_thought_generator.get_role_thought_generator",
        lambda: object(),
    )

    result = generate_role_prefixed_thought("Artist", "painting")
    assert result == "Failsafe: Unable to generate thought (generator not callable)."
