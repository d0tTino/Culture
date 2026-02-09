import pytest

from src.agents.dspy_programs.role_thought_generator import generate_role_prefixed_thought


@pytest.mark.unit
def test_role_thought_passes_traits_to_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def dummy(*, role_name: str, context: str, traits_summary: str) -> object:
        captured["traits"] = traits_summary
        return type("Dummy", (), {"thought": f"As a {role_name}, {traits_summary}"})()

    monkeypatch.setattr(
        "src.agents.dspy_programs.role_thought_generator.get_role_thought_generator",
        lambda: dummy,
    )

    out = generate_role_prefixed_thought("Innovator", "ctx", "openness=0.9")
    assert "openness=0.9" in out
    assert captured["traits"] == "openness=0.9"


@pytest.mark.unit
def test_trait_differences_change_output_style(monkeypatch: pytest.MonkeyPatch) -> None:
    def style_gen(*, role_name: str, context: str, traits_summary: str) -> object:
        style = "collaborative" if "empathy=0.90" in traits_summary else "direct"
        return type("Dummy", (), {"thought": f"As a {role_name}, I respond in a {style} style."})()

    monkeypatch.setattr(
        "src.agents.dspy_programs.role_thought_generator.get_role_thought_generator",
        lambda: style_gen,
    )

    high_empathy = generate_role_prefixed_thought("Facilitator", "ctx", "empathy=0.90")
    low_empathy = generate_role_prefixed_thought("Facilitator", "ctx", "empathy=0.20")

    assert high_empathy != low_empathy
    assert "collaborative" in high_empathy
    assert "direct" in low_empathy
