from types import SimpleNamespace

import pytest

from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.infra import config
from src.sim.simulation import Simulation
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
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
    ) -> dict:
        return {}


def setup_semantic_manager(tmp_path):
    vector = ChromaVectorStoreManager(
        persist_directory=tmp_path, embedding_function=lambda t: [[0.0] for _ in t]
    )
    driver = DummyDriver()
    manager = SemanticMemoryManager(vector, driver)
    return manager, vector, driver


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simulation_schedules_semantic_job(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setitem(config.CONFIG_OVERRIDES, "SEMANTIC_MEMORY_CONSOLIDATION_INTERVAL_STEPS", 1)
    manager, vector, driver = setup_semantic_manager(tmp_path)
    agent = DummyAgent("a1")
    sim = Simulation([agent], vector_store_manager=vector, semantic_manager=manager)

    vector.add_memory("a1", 0, "thought", "hello")

    await sim.run_step()
    assert driver.store  # summary written
