import pytest

from src.agents.core.agent_state import AgentLifecycleState, AgentState, PersonalityTraits
from src.agents.core.personality_engine import ExperienceSignal, PersonalityEngine
from src.sim.population_service import PopulationService

pytestmark = pytest.mark.unit


class _Agent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = AgentState(agent_id=agent_id, name=agent_id, traits=PersonalityTraits())


class _Sim:
    def __init__(self, agents: list[_Agent]) -> None:
        self.current_step = 4
        self.projects = {}
        self.agents = agents
        self.event_kernel = type("Kernel", (), {"emit_environment_event": _noop_async})()
        self.knowledge_board = None
        self.knowledge_board_service = None


async def _noop_async(*args, **kwargs):
    return None


def _clone_state(state: AgentState) -> AgentState:
    dumped = state.model_dump(mode="python") if hasattr(state, "model_dump") else state.dict()
    if hasattr(AgentState, "model_validate"):
        return AgentState.model_validate(dumped)
    return AgentState.parse_obj(dumped)


def test_trait_event_replay_reaches_equal_agent_state() -> None:
    engine = PersonalityEngine()
    baseline = AgentState(agent_id="a", name="A", traits=PersonalityTraits())
    direct = _clone_state(baseline)
    replayed = _clone_state(baseline)
    signal = ExperienceSignal(social_outcome=0.7, conflict_outcome=-0.2, mood_trajectory=0.3)

    engine.apply_experience_drift(direct, signal, max_step=0.01, source="test")
    event = engine.build_experience_drift_event(replayed, signal, max_step=0.01, source="test")
    engine.reduce_trait_drift_event(replayed, event)

    assert replayed.traits.model_dump(mode="python") == direct.traits.model_dump(mode="python")
    assert replayed.personality_transition_events == direct.personality_transition_events
    assert replayed.trait_change_audit == direct.trait_change_audit


@pytest.mark.asyncio
async def test_lifecycle_event_replay_reaches_equal_agent_state() -> None:
    svc = PopulationService()
    a = _Agent("A")
    b = _Agent("B")
    sim = _Sim([a, b])

    await svc.transition_lifecycle(
        simulation=sim,
        agent=a,
        to_state=AgentLifecycleState.RETIRED,
        reason="test_transition",
    )
    transition_event = a.state.identity_events[-1]

    replay_agent = _Agent("A")
    svc.apply_lifecycle_transition_event(agent=replay_agent, event=transition_event)

    assert replay_agent.state.lifecycle_state == a.state.lifecycle_state
    assert replay_agent.state.legacy_artifacts == a.state.legacy_artifacts
    assert replay_agent.state.memory_archival_policy == a.state.memory_archival_policy
