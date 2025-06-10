import time
from types import SimpleNamespace
from typing import ClassVar


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


import pytest

from src.infra import config
from src.shared.memory_store import ChromaMemoryStore
from src.sim.simulation import Simulation


class DummyAgent:
    def __init__(self) -> None:
        self.agent_id = "dummy"
        self.state = DummyState()

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, new_state: DummyState) -> None:
        self.state = new_state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager: ChromaMemoryStore | None = None,
        knowledge_board: object | None = None,
    ) -> dict:
        if vector_store_manager is not None:
            vector_store_manager.add_documents(
                [f"memory {simulation_step}"],
                [{"timestamp": time.time()}],
            )
        return {}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_memory_store_ttl_pruning(monkeypatch: pytest.MonkeyPatch) -> None:
    store = ChromaMemoryStore()
    agent = DummyAgent()
    sim = Simulation(agents=[agent], vector_store_manager=store)

    monkeypatch.setitem(config.CONFIG_OVERRIDES, "MEMORY_STORE_TTL_SECONDS", 7 * 86400)
    monkeypatch.setitem(config.CONFIG_OVERRIDES, "MEMORY_STORE_PRUNE_INTERVAL_STEPS", 1)

    current = 0.0

    def fake_time() -> float:
        return current

    monkeypatch.setattr(time, "time", fake_time)

    for day in range(9):
        current = day * 86400
        await sim.run_step()

    assert len(store._store) == 8
