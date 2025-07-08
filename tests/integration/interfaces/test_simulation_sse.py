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
def test_simulation_emits_sse(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    async def run_test() -> None:
        class TestESR(Response):
            def __init__(self, gen: object) -> None:
                super().__init__("", media_type="text/event-stream")
                self.gen = gen

            async def __call__(self, scope: dict, receive: object, send: Callable) -> None:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"text/event-stream")],
                    }
                )
                async for event in self.gen:
                    body = f"data: {event['data']}\n\n".encode()
                    await send({"type": "http.response.body", "body": body, "more_body": True})
                    break
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                if hasattr(self.gen, "aclose"):
                    await self.gen.aclose()

        monkeypatch.setattr(db, "EventSourceResponse", TestESR)

        import importlib
        import sys

        if "src.http_app" in sys.modules:
            http_app = importlib.reload(sys.modules["src.http_app"])
        else:
            http_app = importlib.import_module("src.http_app")
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

        # Store a memory and create a semantic summary
        vector.add_memory("agent-1", 1, "thought", "hello")
        await manager.run_nightly_job("agent-1")

        msg = db.AgentMessage(agent_id="agent-1", content="hello", step=1)
        await db.enqueue_message(msg)

        transport = httpx.ASGITransport(app=http_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream("GET", "/stream/messages") as resp:
                payload = None
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        payload = json.loads(line.split("data: ", 1)[1])
                        break

            assert payload is not None
            assert payload["content"] == "hello"

            resp2 = await client.get("/api/agents/agent-1/semantic_summaries")
            assert resp2.status_code == 200
            data = resp2.json()
            assert "summaries" in data
            assert isinstance(data["summaries"], list)

    asyncio.run(run_test())
