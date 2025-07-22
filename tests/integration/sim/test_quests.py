import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from src.infra import config
from src.infra.ledger import Ledger
from src.sim import quests
from src.sim.simulation import Simulation
from tests.integration.interfaces.test_dashboard_backend_api import load_dashboard_backend
from tests.utils.mock_llm import MockLLM


class DummyState(SimpleNamespace):
    ip: float = 0.0
    du: float = 0.0
    short_term_memory: ClassVar[list] = []
    messages_sent_count: int = 0
    last_message_step: int = 0
    relationships: ClassVar[dict] = {}
    role: str = "dummy"
    steps_in_current_role: int = 0

    def update_collective_metrics(self, ip: float, du: float) -> None:
        pass


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = DummyState()

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, new_state: DummyState) -> None:
        self.state = new_state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        memory_service: object | None = None,
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
    ) -> dict:
        return {}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_periodic_quest_generation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    quests.QUESTS.clear()
    monkeypatch.setitem(config.CONFIG_OVERRIDES, "QUEST_GENERATION_INTERVAL_STEPS", 1)
    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr(quests, "ledger", ledger)
    responses = {"structured_output": {"id": 1, "title": "Q1", "description": "desc"}}
    with MockLLM(responses):

        async def _allow(_: str) -> bool:
            return True

        monkeypatch.setattr("src.governance.evaluate_policy", _allow)
        monkeypatch.setattr("src.sim.simulation.evaluate_policy", _allow)
        agent = DummyAgent("a1")
        sim = Simulation(agents=[agent])
        await sim.run_step(max_turns=3)
        await sim.stop_event_listener()

    assert len(ledger.get_quests()) == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_quests_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    quests.QUESTS.clear()
    monkeypatch.setitem(config.CONFIG_OVERRIDES, "QUEST_GENERATION_INTERVAL_STEPS", 1)
    db = load_dashboard_backend()
    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr(db, "ledger", ledger)
    monkeypatch.setattr(quests, "ledger", ledger)
    responses = {"structured_output": {"id": 2, "title": "Q2", "description": "desc"}}
    with MockLLM(responses):

        async def _allow(_: str) -> bool:
            return True

        monkeypatch.setattr("src.governance.evaluate_policy", _allow)
        monkeypatch.setattr("src.sim.simulation.evaluate_policy", _allow)
        agent = DummyAgent("a1")
        sim = Simulation(agents=[agent])
        await sim.run_step(max_turns=3)
        await sim.stop_event_listener()

    resp = await db.get_quests_api()
    data = json.loads(resp.body)
    assert data["quests"][0]["title"] == "Q2"
