import asyncio
import sys
import types
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

# Stub heavy dependencies
sys.modules.setdefault("weaviate", SimpleNamespace(classes=SimpleNamespace()))
sys.modules.setdefault("weaviate.classes", SimpleNamespace())

class DummyNeo4j:
    Driver = object
    GraphDatabase = object

sys.modules.setdefault("neo4j", DummyNeo4j())

dashboard_stub = types.ModuleType("src.interfaces.dashboard_backend")
class SimulationEvent(SimpleNamespace):
    event_type: str = ""
    data: dict[str, Any] | None = None

dashboard_stub.SimulationEvent = SimulationEvent

async def _noop(*args: Any, **kwargs: Any) -> None:
    return None

dashboard_stub.emit_event = _noop
dashboard_stub.emit_map_action_event = _noop
dashboard_stub.event_queue = asyncio.Queue()

sys.modules.setdefault("src.interfaces.dashboard_backend", dashboard_stub)

import pydantic
from pydantic_settings import BaseSettings as PydanticBaseSettings

pydantic.BaseSettings = PydanticBaseSettings

import src.infra.config as config
from src.sim.simulation import Simulation


async def _allow(_: str) -> bool:
    return True


class DummyState(SimpleNamespace):
    ip: float = 0.0
    du: float = 0.0
    short_term_memory: ClassVar[list[Any]] = []
    messages_sent_count: int = 0
    last_message_step: int = 0
    relationships: ClassVar[dict[str, Any]] = {}
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
        environment_perception: dict[str, Any] | None = None,
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
    ) -> dict[str, Any]:
        return {"message_content": f"hello from {self.agent_id}"}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_message_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.sim.simulation.evaluate_policy", _allow)
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setattr(config, "REQUIRED_CONFIG_KEYS", [])

    sim = Simulation(agents=[DummyAgent("a1"), DummyAgent("a2")])  # type: ignore[arg-type]

    await asyncio.gather(sim._run_agent_turn(0), sim._run_agent_turn(1))

    async with sim._msg_lock:
        assert len(sim.pending_messages_for_next_round) == 2
        assert len(sim.messages_to_perceive_this_round) == 2
