import json
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from src.app import create_simulation
from src.interfaces import dashboard_backend as db
from src.sim.simulation import Simulation
from tests.utils.mock_llm import MockLLM


class ReplayState(SimpleNamespace):
    ip: float = 0.0
    du: float = 0.0
    short_term_memory: ClassVar[list[Any]] = []
    messages_sent_count: int = 0
    last_message_step: int = 0
    relationships: ClassVar[dict[str, Any]] = {}
    steps_in_current_role: int = 0
    mood_level: float = 0.0

    def update_collective_metrics(self, ip: float, du: float) -> None:  # pragma: no cover - noop
        self.ip = ip
        self.du = du


class ReplayAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = ReplayState()

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, new_state: ReplayState) -> None:
        self.state = new_state


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_action_explain_why_logged_and_served(
    monkeypatch: pytest.MonkeyPatch, tmp_path: str
) -> None:
    from src.infra import event_log as event_log_module

    log_path = tmp_path / "event_log.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(log_path))
    monkeypatch.setattr(event_log_module, "_last_hash", None, raising=False)
    monkeypatch.setattr(event_log_module, "_seed_cache", {}, raising=False)
    monkeypatch.setattr(event_log_module, "_header_written", set(), raising=False)

    captured_events: list[dict[str, Any]] = []
    real_log_event = event_log_module.log_event

    def capture(event: dict[str, Any]) -> dict[str, Any]:
        persisted = real_log_event(event)
        captured_events.append(persisted)
        return persisted

    monkeypatch.setattr("src.sim.simulation.log_event", capture)

    with MockLLM():
        sim = create_simulation(num_agents=1, steps=1, scenario="explain why")
        sim.knowledge_board.add_entry(
            "Shared context", sim.agents[0].agent_id, 0, sim.vector.to_dict()
        )
        await sim.run_step(max_turns=1)
        await sim.stop_event_listener()

    agent_events = [evt for evt in captured_events if evt.get("type") == "agent_action"]
    assert agent_events, "Expected at least one agent_action event"
    event = agent_events[0]

    explain = event.get("explain_why")
    assert isinstance(explain, dict)
    assert isinstance(explain.get("memories"), list)
    assert isinstance(explain.get("tool_calls"), list)
    kb_entries = explain.get("knowledge_board_entries", [])
    assert isinstance(kb_entries, list)
    assert any("Shared context" in entry for entry in kb_entries)

    replay_sim = Simulation(agents=[ReplayAgent(str(event.get("agent_id", "")))])  # type: ignore[list-item]
    replay_sim.apply_event(event)

    monkeypatch.setattr(db.event_log, "fetch_events", lambda *a, **k: [event])

    resp = await db.api_agent_action_explain_why(limit=5)
    body = resp.body.decode("utf-8") if isinstance(resp.body, (bytes, bytearray)) else resp.body
    data = json.loads(body)

    returned = data.get("events", [])
    assert returned, "Expected agent action data in dashboard response"
    explain_resp = returned[0]["explain_why"]
    assert any("Shared context" in entry for entry in explain_resp["knowledge_board_entries"])
    assert isinstance(explain_resp["memories"], list)
    assert isinstance(explain_resp["tool_calls"], list)
