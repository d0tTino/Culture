from types import SimpleNamespace

import pytest

from src.app import create_simulation
from src.infra import config
from tests.unit.memory.test_semantic_memory_manager import DummyDriver


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = SimpleNamespace(ip=0.0, du=0.0, mood_level=0.0)

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, new_state: SimpleNamespace) -> None:
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


def setup_semantic_manager(tmp_path):
    return DummyDriver()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simulation_schedules_semantic_job(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from src.sim import simulation as sim_module

    async def _allow(action):
        return True

    monkeypatch.setattr(sim_module, "evaluate_policy", _allow)
    monkeypatch.setitem(config.CONFIG_OVERRIDES, "SEMANTIC_MEMORY_CONSOLIDATION_INTERVAL_STEPS", 1)
    driver = setup_semantic_manager(tmp_path)
    monkeypatch.setattr("neo4j.GraphDatabase.driver", lambda *a, **k: driver, raising=False)
    agent = DummyAgent("a1")
    sim = create_simulation(
        num_agents=1,
        steps=1,
        scenario="semantic",
        use_vector_store=True,
        vector_store_dir=tmp_path,
        use_semantic_memory=True,
        semantic_db_uri="bolt://dummy",
    )

    sim.memory_service.vector_store.add_memory("a1", 0, "thought", "hello")

    await sim.run_step()
    assert driver.store  # summary written
