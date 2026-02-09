import asyncio

import pytest

from src.app import spawn_agent_command
from src.interfaces.dashboard_backend import SimulationEvent
from src.sim.context import SimulationContext


@pytest.mark.unit
@pytest.mark.asyncio
async def test_spawn_agent_command_includes_optional_profile_fields() -> None:
    queue: asyncio.Queue[SimulationEvent] = asyncio.Queue()
    ctx = SimulationContext()
    ctx.get_event_queue = lambda: queue  # type: ignore[method-assign]

    await spawn_agent_command(
        "agent-x",
        ctx,
        role="Analyzer",
        persona="Careful planner",
        backstory="Raised in a lab",
        traits={"openness": 0.8},
    )

    event = await queue.get()
    assert event.type == "control"
    assert event.data == {
        "command": "spawn",
        "agent_id": "agent-x",
        "role": "Analyzer",
        "persona": "Careful planner",
        "backstory": "Raised in a lab",
        "traits": {"openness": 0.8},
    }
