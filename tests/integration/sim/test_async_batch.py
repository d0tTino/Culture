import asyncio
import sys
import time
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from src.interfaces.dashboard_backend import SimulationEvent, get_event_queue


class DummyNeo4j:
    Driver = object
    GraphDatabase = object


sys.modules.setdefault("neo4j", DummyNeo4j())

from src.sim.simulation import Simulation


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
        await asyncio.sleep(0.1)
        return {"step": simulation_step}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_agent_turns() -> None:
    agents = [DummyAgent(str(i)) for i in range(5)]
    sim = Simulation(agents=agents)  # type: ignore[arg-type]

    start = time.perf_counter()
    results = await sim.run_turns_concurrent(agents)
    elapsed = time.perf_counter() - start

    assert len(results) == 5
    assert elapsed < 0.5
    assert sim.current_step == 5


async def _clear_event_queue() -> None:
    queue = get_event_queue()
    while not queue.empty():
        _ = await queue.get()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_events_enqueued_during_run_step() -> None:
    await _clear_event_queue()
    agent = DummyAgent("agent1")
    sim = Simulation(agents=[agent])  # type: ignore[list-item]

    await sim.run_step()

    queue = get_event_queue()
    evt = await asyncio.wait_for(queue.get(), 0.1)
    assert isinstance(evt, SimulationEvent)
    assert evt.type == "agent_action"
    assert evt.data is not None
    assert evt.data["agent_id"] == "agent1"
    assert evt.data["step"] == 1
    await _clear_event_queue()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_async_batch_stress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure vLLM async batching handles many concurrent requests."""
    from unittest.mock import AsyncMock, MagicMock

    from src.infra import config, llm_client

    captured_urls: list[str] = []

    async def fake_post(url: str, json: dict[str, Any], timeout: float | None = None) -> MagicMock:
        captured_urls.append(url)
        responses = [
            {
                "choices": [{"message": {"content": f"resp{i}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            for i, _ in enumerate(json["requests"])
        ]
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.text = json_module.dumps({"responses": responses})
        return resp

    # Configure vLLM client
    monkeypatch.setattr(llm_client, "LLM_API_BASE", "http://vllm:8001")
    monkeypatch.setattr(llm_client, "VLLM_API_BASE", "http://vllm:8001")
    monkeypatch.setattr(llm_client, "USE_VLLM", True)
    monkeypatch.setattr(config.settings, "LLM_API_BASE", "http://vllm:8001")  # type: ignore[attr-defined]
    monkeypatch.setattr(config.settings, "VLLM_API_BASE", "http://vllm:8001")  # type: ignore[attr-defined]
    monkeypatch.setitem(config._CONFIG, "LLM_API_BASE", "http://vllm:8001")
    monkeypatch.setitem(config._CONFIG, "VLLM_API_BASE", "http://vllm:8001")
    monkeypatch.setattr(llm_client, "client", llm_client._create_vllm_client())  # type: ignore[attr-defined]

    # Patch DU accounting
    monkeypatch.setattr(llm_client.ledger, "calculate_gas_price", lambda agent_id: (0.1, 0.0))
    monkeypatch.setattr(llm_client.ledger, "log_change", lambda *a, **k: None)

    class DummyRM:
        def __init__(self) -> None:
            self.ensure_calls: list[tuple[str, float]] = []

        def ensure_du_budget(self, agent_id: str, amt: float) -> None:
            self.ensure_calls.append((agent_id, amt))

        def charge_du(self, agent_id: str, amt: float) -> None:
            pass

    dummy_rm = DummyRM()
    monkeypatch.setattr("src.sim.resource_manager.get_resource_manager", lambda: dummy_rm)

    import json as json_module

    monkeypatch.setattr(
        "src.infra.llm_client.httpx.AsyncClient.post",
        AsyncMock(side_effect=fake_post),
    )

    llm = llm_client.LLMClient(
        llm_client.LLMClientConfig(batch_size=5, batch_timeout=0.01, model_name="m")
    )

    async def _call(i: int) -> str:
        state = SimpleNamespace(agent_id=f"a{i}", du=10.0)
        resp = await llm.chat(model="m", messages=[], agent_state=state)
        return resp["message"]["content"]

    results = await asyncio.gather(*[_call(i) for i in range(20)])

    assert len(results) == 20
    assert all(r.startswith("resp") for r in results)
    assert all(url.endswith("/v1/async_batch") for url in captured_urls)
    assert len(dummy_rm.ensure_calls) == 20
