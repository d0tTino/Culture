import asyncio
import time

import pytest

from src.interfaces.dashboard_backend import SimulationEvent, get_event_queue
from src.sim.simulation import Simulation


class DummyAgent:
    def __init__(self) -> None:
        self.agent_id = "A"
        self.state = type(
            "S",
            (),
            {
                "ip": 0.0,
                "du": 0.0,
                "short_term_memory": [],
                "messages_sent_count": 0,
                "last_message_step": None,
                "collective_ip": 0.0,
                "collective_du": 0.0,
            },
        )()

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, state: object) -> None:
        self.state = state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager=None,
        knowledge_board=None,
    ) -> dict:
        return {}


@pytest.mark.unit
def test_event_task_runs_without_running_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[str] = []

    async def handler(self: Simulation, text: str) -> None:
        received.append(text)

    monkeypatch.setattr(Simulation, "_handle_human_command", handler)
    sim = Simulation([DummyAgent()])

    async def send() -> None:
        queue = get_event_queue()
        await queue.put(
            SimulationEvent(event_type="broadcast", data={"content": "hi"})
        )

    fut = asyncio.run_coroutine_threadsafe(send(), sim._event_loop)
    fut.result()
    time.sleep(0.05)
    sim.close()

    assert received == ["hi"]
