import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.interfaces import dashboard_backend as db


@pytest.fixture
def anyio_backend() -> str:
    """Limit AnyIO to the asyncio backend."""
    return "asyncio"


class DummyDiscordClient:
    """Minimal stand-in for ``discord.Client``."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.event = lambda f: f
        self.user = "dummy"

    async def start(self, token: str) -> None:  # pragma: no cover - unused
        self.token = token

    async def close(self) -> None:  # pragma: no cover - unused
        pass


class DummyNeo4j:
    Driver = object
    GraphDatabase = object


class DummyState:
    def __init__(self, ip: float = 2.0, du: float = 2.0) -> None:
        self.ip = ip
        self.du = du
        self.short_term_memory = []
        self.messages_sent_count = 0
        self.last_message_step = None
        self.collective_ip = 0.0
        self.collective_du = 0.0


class DummyAgent:
    def __init__(self, agent_id: str, ip: float = 2.0, du: float = 2.0) -> None:
        self.agent_id = agent_id
        self._state = DummyState(ip, du)
        from src.infra.ledger import ledger as _ledger

        _ledger.log_change(agent_id, ip, du, "init")

    def get_id(self) -> str:
        return self.agent_id

    @property
    def state(self) -> DummyState:
        return self._state

    def update_state(self, state: DummyState) -> None:
        self._state = state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager=None,
        knowledge_board=None,
    ) -> dict:
        return {}


@pytest.mark.integration
@pytest.mark.anyio("asyncio")
async def test_event_queue_broadcast_cost(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr("src.sim.simulation.ledger", ledger)

    agent = DummyAgent("A")
    with patch("src.interfaces.discord_bot.discord.Client", DummyDiscordClient):
        sim = Simulation([agent])

    queue = db.get_event_queue()
    await queue.put(
        db.SimulationEvent(
            event_type="broadcast", data={"author": "human", "content": "/broadcast hello"}
        )
    )
    await asyncio.sleep(0.1)

    assert agent.state.ip == pytest.approx(1.0)
    assert agent.state.du == pytest.approx(1.0)
    async with sim._msg_lock:
        recipients = {m["recipient_id"] for m in sim.pending_messages_for_next_round}
    assert recipients == {"A"}

    sim.close()
    await queue.put(None)


@pytest.mark.integration
@pytest.mark.anyio("asyncio")
async def test_event_queue_kb_post(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr("src.sim.simulation.ledger", ledger)

    agent = DummyAgent("A")
    with patch("src.interfaces.discord_bot.discord.Client", DummyDiscordClient):
        sim = Simulation([agent])

    queue = db.get_event_queue()
    await queue.put(
        db.SimulationEvent(
            event_type="broadcast", data={"author": "human", "content": "/kb important fact"}
        )
    )
    await asyncio.sleep(0.1)

    entries = sim.knowledge_board.get_state(max_entries=1)
    assert "important fact" in entries[0]

    sim.close()
    await queue.put(None)
