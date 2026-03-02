import sys
from pathlib import Path

import pytest

from src.interfaces.domain_command_adapters import (
    command_from_discord_message,
    command_from_payload,
)
from src.interfaces.interaction_schema import InteractionContext
from src.interfaces.transport_adapters import parse_discord_message_routing
from src.sim.commands.domain_commands import ControlCommand, DirectMessageCommand

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


def _snapshot_messages(sim) -> list[tuple[str, str]]:
    return [(m["sender_id"], str(m["recipient_id"])) for m in sim.pending_messages_for_next_round]


@pytest.mark.asyncio
async def test_equivalent_direct_message_paths_produce_identical_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    outcomes = []
    for path in ("discord", "event_bus", "dashboard", "internal"):
        ledger = Ledger(tmp_path / f"cmd-{path}.sqlite")
        monkeypatch.setattr("src.infra.ledger.ledger", ledger)
        monkeypatch.setattr("src.sim.simulation.ledger", ledger)
        sim = Simulation([DummyAgent("alpha"), DummyAgent("beta")])
        try:
            if path == "discord":
                recipient, is_broadcast, content, err = parse_discord_message_routing("/dm beta hello")
                assert err is None and not is_broadcast
                command = command_from_discord_message(
                    content=content,
                    recipient_id=recipient,
                    is_broadcast=is_broadcast,
                    target_agent_id=recipient,
                )
                result = await sim.command_dispatcher.dispatch(
                    command,
                    context=InteractionContext(sender_id="user-1", source="discord"),
                )
            elif path == "event_bus":
                command = command_from_payload(
                    {
                        "command_type": "direct_message",
                        "content": "hello",
                        "recipient_id": "beta",
                        "target_agent_id": "beta",
                    }
                )
                result = await sim.command_dispatcher.dispatch(
                    command,
                    context=InteractionContext(sender_id="user-1", source="event_bus"),
                )
            elif path == "dashboard":
                command = command_from_payload(
                    {
                        "command_type": "direct_message",
                        "content": "hello",
                        "recipient_id": "beta",
                        "target_agent_id": "beta",
                    }
                )
                result = await sim.command_dispatcher.dispatch(
                    command,
                    context=InteractionContext(sender_id="user-1", source="dashboard"),
                )
            else:
                result = await sim.command_dispatcher.dispatch(
                    DirectMessageCommand(content="hello", recipient_id="beta", target_agent_id="beta"),
                    context=InteractionContext(sender_id="user-1", source="internal"),
                )
            outcomes.append((result.model_dump(), _snapshot_messages(sim)))
        finally:
            sim.close()

    assert outcomes[0] == outcomes[1] == outcomes[2] == outcomes[3]


@pytest.mark.asyncio
async def test_equivalent_control_paths_mutate_state_identically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    states = []
    for path in ("discord", "dashboard", "internal"):
        ledger = Ledger(tmp_path / f"control-{path}.sqlite")
        monkeypatch.setattr("src.infra.ledger.ledger", ledger)
        monkeypatch.setattr("src.sim.simulation.ledger", ledger)
        sim = Simulation([DummyAgent("alpha"), DummyAgent("beta")])
        try:
            if path == "discord":
                command = command_from_payload({"command": "set_speed", "value": 2.5})
                result = await sim.command_dispatcher.dispatch(
                    command,
                    context=InteractionContext(sender_id="user-1", source="discord", permissions={"admin"}),
                )
            elif path == "dashboard":
                command = command_from_payload({"command": "set_speed", "value": 2.5})
                result = await sim.command_dispatcher.dispatch(
                    command,
                    context=InteractionContext(sender_id="dashboard", source="dashboard", permissions={"admin"}),
                )
            else:
                result = await sim.command_dispatcher.dispatch(
                    ControlCommand(action="set_speed", value=2.5),
                    context=InteractionContext(sender_id="internal", source="internal", permissions={"admin"}),
                )
            states.append((result.status, sim.speed, result.reason_code))
        finally:
            sim.close()

    assert states[0] == states[1] == states[2] == ("ok", 2.5, "moderation_applied")
