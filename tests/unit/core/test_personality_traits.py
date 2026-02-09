import pytest

from src.agents.core.agent_controller import AgentController
from src.agents.core.agent_state import AgentState, PersonalityTraits
from src.agents.core.roles import ROLE_ANALYZER, ROLE_FACILITATOR, get_role_trait_template


@pytest.mark.unit
def test_role_trait_templates_seed_state() -> None:
    facilitator = AgentState(
        agent_id="a1",
        name="fac",
        current_role=ROLE_FACILITATOR,
    )
    analyzer = AgentState(agent_id="a2", name="ana", current_role=ROLE_ANALYZER)

    assert facilitator.traits.empathy == pytest.approx(
        get_role_trait_template(ROLE_FACILITATOR)["empathy"]
    )
    assert analyzer.traits.analytical_focus == pytest.approx(
        get_role_trait_template(ROLE_ANALYZER)["analytical_focus"]
    )


@pytest.mark.unit
def test_traits_modulate_mood_response() -> None:
    sensitive = AgentState(
        agent_id="s1",
        name="sensitive",
        traits=PersonalityTraits(emotional_sensitivity=0.95, resilience=0.2),
    )
    resilient = AgentState(
        agent_id="r1",
        name="resilient",
        traits=PersonalityTraits(emotional_sensitivity=0.2, resilience=0.95),
    )

    AgentController(sensitive).update_mood(-1.0)
    AgentController(resilient).update_mood(-1.0)

    assert sensitive.mood_level < resilient.mood_level


@pytest.mark.unit
def test_trait_drift_is_bounded() -> None:
    state = AgentState(agent_id="a3", name="drift")
    before = state.traits.trust_baseline
    state.apply_trait_drift({"trust_baseline": 1.0}, max_step=0.01)
    assert state.traits.trust_baseline == pytest.approx(before + 0.01)
