import sys
from pathlib import Path

import pytest

from src.interfaces.interaction_commands import InteractionContext
from src.sim.commands.domain_commands import HumanMessageCommand

pytestmark = pytest.mark.unit


class DummyNeo4j:
    Driver = object
    GraphDatabase = object


class DummyState:
    def __init__(self, ip: float = 10.0, du: float = 10.0) -> None:
        self.ip = ip
        self.du = du


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._state = DummyState()

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, state: DummyState) -> None:
        self._state = state

    @property
    def state(self) -> DummyState:
        return self._state

    async def run_turn(self, *args, **kwargs):
        return {}


def _snapshot(sim) -> tuple[list[tuple[str, str]], float, float]:
    pending = [(m["sender_id"], str(m["recipient_id"])) for m in sim.pending_messages_for_next_round]
    return pending, sim.agents[0].state.ip, sim.agents[0].state.du


async def _run_path(path: str, sim, content: str):
    if path == "discord":
        result = await sim.command_bus.dispatch(
            HumanMessageCommand(content=content),
            context=InteractionContext(sender_id="user-1", source="discord"),
        )
    elif path == "dashboard":
        result = await sim.command_bus.dispatch_payload(
            {
                "command_type": "human_message",
                "content": content,
                "sender_id": "user-1",
                "source": "dashboard",
            },
            context=InteractionContext(sender_id="user-1", source="dashboard"),
        )
    else:
        result = await sim.interaction_service.execute_from_payload(
            {
                "command_type": "human_message",
                "content": content,
                "sender_id": "user-1",
                "source": "queue",
            },
            context=InteractionContext(sender_id="user-1", source="queue"),
        )
    return result.model_dump(), _snapshot(sim)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["hello", "/broadcast hello all"])
async def test_interaction_contract_across_transport_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())

    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    outcomes = []
    for path in ("discord", "dashboard", "queue"):
        ledger = Ledger(tmp_path / f"{path}.sqlite")
        monkeypatch.setattr("src.infra.ledger.ledger", ledger)
        monkeypatch.setattr("src.sim.simulation.ledger", ledger)
        sim = Simulation([DummyAgent("alpha"), DummyAgent("beta")])
        try:
            outcomes.append(await _run_path(path, sim, content))
        finally:
            sim.close()

    base_result, base_snapshot = outcomes[0]
    for result, snapshot in outcomes[1:]:
        assert result == base_result
        assert snapshot == base_snapshot
