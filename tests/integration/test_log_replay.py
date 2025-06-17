import pytest

from src.app import create_simulation
from src.infra import event_log
from src.infra.checkpoint import load_checkpoint, save_checkpoint
from tests.utils.mock_llm import MockLLM


@pytest.mark.integration
def test_replay_from_event_log(monkeypatch, tmp_path):
    monkeypatch.setattr(event_log, "log_event", lambda e: None)
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)

    with MockLLM():
        sim = create_simulation(num_agents=1, steps=1, scenario="log")
        chk = tmp_path / "sim.pkl"
        save_checkpoint(sim, chk)

    event = {
        "type": "agent_action",
        "agent_id": sim.agents[0].agent_id,
        "step": 3,
        "ip": 12.34,
        "du": 56.78,
    }

    expected_step = event["step"]
    expected_ip = event["ip"]
    expected_du = event["du"]

    def fake_fetch(after_step=0):
        return [e for e in [event] if e.get("step", 0) > after_step]

    monkeypatch.setattr(event_log, "fetch_events", fake_fetch)

    loaded, _ = load_checkpoint(chk, replay=True)

    assert loaded.current_step == expected_step
    assert loaded.agents[0].state.ip == expected_ip
    assert loaded.agents[0].state.du == expected_du
