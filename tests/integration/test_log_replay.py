import asyncio

import pytest

from src.app import create_simulation
from src.infra import event_log
from src.infra.checkpoint import load_checkpoint, save_checkpoint
from tests.utils.mock_llm import MockLLM


@pytest.mark.integration
def test_replay_from_event_log(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(event_log, "log_event", lambda e: events.append(e))
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)

    with MockLLM():
        sim = create_simulation(num_agents=1, steps=1, scenario="log")
        chk = tmp_path / "sim.pkl"
        save_checkpoint(sim, chk)
        asyncio.run(sim.async_run(2))
        expected_ip = sim.agents[0].state.ip
        expected_step = sim.current_step

    def fake_fetch(after_step=0):
        return [ev for ev in events if ev.get("step", 0) > after_step]

    monkeypatch.setattr(event_log, "fetch_events", fake_fetch)

    loaded, _ = load_checkpoint(chk, replay=True)

    assert loaded.current_step == expected_step
    assert loaded.agents[0].state.ip == expected_ip
