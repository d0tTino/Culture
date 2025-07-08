import asyncio
import json
from types import SimpleNamespace
from typing import Callable

import httpx
import pytest
from starlette.responses import Response

# Skip this test if FastAPI is not available.
pytest.importorskip("fastapi")

from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.interfaces import dashboard_backend as db
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


async def _clear_event_queue() -> None:
    queue = db.get_event_queue()
    while not queue.empty():
        await queue.get()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simulation_emits_sse(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=tmp_path, embedding_function=lambda t: [[0.0] for _ in t]
    )
    driver = DummyDriver()
    manager = SemanticMemoryManager(vector, driver)
    agent = DummyAgent("a1")
    sim = Simulation([agent], vector_store_manager=vector, semantic_manager=manager)

    class CaptureESR:
        def __init__(self, gen: object) -> None:
            self.gen = gen

    monkeypatch.setattr(http_app, "EventSourceResponse", CaptureESR)
    await _clear_event_queue()
    await sim.run_step()

    queue = db.get_event_queue()
    await queue.put(None)
    resp = await http_app.stream_events(DummyRequest())
    event = await resp.gen.__anext__()
    payload = json.loads(event["data"])
    assert payload["event_type"] == "agent_action"
    with pytest.raises(StopAsyncIteration):
        await resp.gen.__anext__()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_messages_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    queue: asyncio.Queue[db.AgentMessage] = asyncio.Queue()
    monkeypatch.setattr(db, "message_sse_queue", queue)

    msg = db.AgentMessage(agent_id="agent-1", content="hello", step=1)
    await db.enqueue_message(msg)

    async with httpx.AsyncClient(app=http_app.app, base_url="http://test") as client:
        async with client.stream("GET", "/stream/messages") as resp:
            data_line = None
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_line = line.removeprefix("data: ")
                    break
    assert data_line is not None
    payload = json.loads(data_line)
    assert payload["content"] == "hello"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_semantic_summaries_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    vector = ChromaVectorStoreManager(
        persist_directory=tmp_path, embedding_function=lambda t: [[0.0] for _ in t]
    )
    driver = DummyDriver()
    manager = SemanticMemoryManager(vector, driver)
    agent = DummyAgent("agent-1")
    sim = Simulation([agent], vector_store_manager=vector, semantic_manager=manager)

    monkeypatch.setitem(db.SIM_STATE, "semantic_manager", manager)
    monkeypatch.setitem(db.SIM_STATE, "simulation", sim)

    await _clear_event_queue()
    await sim.run_step()

    async with httpx.AsyncClient(app=http_app.app, base_url="http://test") as client:
        resp = await client.get("/api/agents/agent-1/semantic_summaries")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("summaries"), list)
