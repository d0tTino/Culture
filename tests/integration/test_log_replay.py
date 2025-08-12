import pytest

from src.app import create_simulation
from src.infra import event_log
from src.infra.checkpoint import load_checkpoint, save_checkpoint
from tests.utils.mock_llm import MockLLM


@pytest.mark.integration
def test_replay_from_event_log(monkeypatch, tmp_path):
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)
    monkeypatch.setattr(event_log, "_producer", None, raising=False)

    with MockLLM():
        sim = create_simulation(num_agents=1, steps=1, scenario="log")
        chk = tmp_path / "sim.pkl"
        save_checkpoint(sim, chk)

    base_event = {
        "type": "agent_action",
        "agent_id": sim.agents[0].agent_id,
        "step": 3,
        "ip": 12.34,
        "du": 56.78,
    }

    valid_event = event_log.log_event(base_event)
    out_of_order = event_log.log_event({**base_event, "step": 2})
    tampered = {**valid_event, "step": 4, "du": 0.0, "trace_hash": valid_event["trace_hash"], "prev_hash": valid_event.get("trace_hash")}

    baseline, _ = load_checkpoint(chk, replay=False)
    baseline.apply_event(valid_event)

    def fake_fetch(after_step=0):
        raw = [valid_event, out_of_order, tampered]
        return event_log._filter_events(raw, after_step=after_step)

    monkeypatch.setattr(event_log, "fetch_events", fake_fetch)

    loaded, _ = load_checkpoint(chk, replay=True)

    assert loaded.current_step == baseline.current_step
    assert loaded.agents[0].state.ip == baseline.agents[0].state.ip
    assert loaded.agents[0].state.du == baseline.agents[0].state.du
