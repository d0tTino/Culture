from __future__ import annotations

from src.interfaces.domain_command_adapters import parse_bus_command
from src.interfaces.interaction_schema import ControlEnvelope


def test_checkpoint_and_replay_to_step_parse_as_control() -> None:
    checkpoint = parse_bus_command({"command": "checkpoint"})
    replay = parse_bus_command({"command": "replay_to_step", "value": 25})

    assert isinstance(checkpoint, ControlEnvelope)
    assert checkpoint.action == "checkpoint"
    assert isinstance(replay, ControlEnvelope)
    assert replay.action == "replay_to_step"
