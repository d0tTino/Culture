from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.infra import config
from src.sim.simulation import Simulation


class DummyAgent:
    def __init__(self, ip_gain: float, du_gain: float) -> None:
        self.agent_id = "dummy"
        self.state = SimpleNamespace(
            ip=0.0,
            du=0.0,
            short_term_memory=[],
            messages_sent_count=0,
            last_message_step=0,
            relationships={},
            role="dummy",
            steps_in_current_role=0,
            update_collective_metrics=lambda ip, du: None,
        )
        self.ip_gain = ip_gain
        self.du_gain = du_gain

    def get_id(self) -> str:  # pragma: no cover - compatibility
        return self.agent_id

    def update_state(self, new_state: SimpleNamespace) -> None:  # pragma: no cover
        self.state = new_state

    async def run_turn(self, **_: object) -> dict[str, object]:
        self.state.ip += self.ip_gain
        self.state.du += self.du_gain
        return {}


@given(ip_gain=st.floats(min_value=0, max_value=50), du_gain=st.floats(min_value=0, max_value=50))
@pytest.mark.asyncio
async def test_per_tick_throttling(
    monkeypatch: pytest.MonkeyPatch, ip_gain: float, du_gain: float
) -> None:
    monkeypatch.setitem(config.CONFIG_OVERRIDES, "MAX_IP_PER_TICK", 5.0)
    monkeypatch.setitem(config.CONFIG_OVERRIDES, "MAX_DU_PER_TICK", 7.0)

    agent = DummyAgent(ip_gain, du_gain)
    sim = Simulation(agents=[agent])

    await sim.run_step()

    assert agent.state.ip <= 5.0
    assert agent.state.du <= 7.0
