import pytest

from src.app import create_simulation
from src.infra import event_log
from src.infra.checkpoint import load_checkpoint, save_checkpoint
from tests.utils.mock_llm import MockLLM


class DummyProducer:
    def __init__(self, store):
        self.store = store

    def produce(self, topic: str, payload: bytes) -> None:
        self.store.append(payload)

    def poll(self, timeout: float) -> None:
        pass


class DummyConsumer:
    def __init__(self, _conf, store):
        self.store = store
        self.index = 0

    def subscribe(self, _topics):
        pass

    def poll(self, _timeout):
        if self.index >= len(self.store):
            return None
        payload = self.store[self.index]
        self.index += 1

        class Message:
            def __init__(self, value: bytes) -> None:
                self._value = value

            def error(self):
                return None

            def value(self) -> bytes:
                return self._value

        return Message(payload)

    def close(self) -> None:
        pass


@pytest.mark.integration
def test_stream_replay(monkeypatch, tmp_path):
    messages = []
    monkeypatch.setenv("ENABLE_REDPANDA", "1")
    monkeypatch.setattr(event_log, "KafkaProducer", lambda conf: DummyProducer(messages))
    monkeypatch.setattr(event_log, "KafkaConsumer", lambda conf: DummyConsumer(conf, messages))
    monkeypatch.setattr(event_log, "_producer", None, raising=False)

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
    event = event_log.log_event(event)

    events = list(event_log.stream_events(after_step=0, timeout=0.1))
    assert events == [event]

    loaded, _ = load_checkpoint(chk, replay=False)
    for ev in events:
        loaded.apply_event(ev)

    assert loaded.current_step == 3
    assert loaded.agents[0].state.ip == 12.34
    assert loaded.agents[0].state.du == 56.78
