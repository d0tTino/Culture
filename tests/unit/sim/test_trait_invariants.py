from types import SimpleNamespace

import pytest

from src.agents.core.agent_state import PersonalityTraits
from src.sim.simulation import Simulation


class SeedAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._state = SimpleNamespace(ip=0.0, du=0.0)

    def get_id(self) -> str:
        return self.agent_id

    @property
    def state(self) -> SimpleNamespace:
        return self._state


@pytest.mark.unit
def test_trait_update_invariants_accept_valid_records() -> None:
    sim = Simulation([SeedAgent("seed")])
    state = SimpleNamespace(traits=PersonalityTraits())
    sim._assert_trait_update_invariants(
        state,
        [{"delta": 0.01, "cause": "experience_drift", "source": "simulation.turn"}],
        max_step=0.01,
    )
    sim.close()


@pytest.mark.unit
def test_trait_update_invariants_reject_missing_audit_source() -> None:
    sim = Simulation([SeedAgent("seed")])
    state = SimpleNamespace(traits=PersonalityTraits())

    with pytest.raises(AssertionError, match="cause/source"):
        sim._assert_trait_update_invariants(
            state,
            [{"delta": 0.01, "cause": "experience_drift"}],
            max_step=0.01,
        )
    sim.close()
