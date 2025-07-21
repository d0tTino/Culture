import asyncio

import pytest

from src.sim.world_map import WorldMap

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_concurrent_moves_multiple_agents() -> None:
    m = WorldMap(width=5, height=5)
    await m.add_agent("A", x=0, y=0)
    await m.add_agent("B", x=0, y=1)

    async def mover(agent_id: str) -> None:
        for _ in range(50):
            await m.move(agent_id, 1, 0)

    await asyncio.gather(*(mover(a) for a in ("A", "B")))

    assert m.agent_positions["A"] == (4, 0)
    assert m.agent_positions["B"] == (4, 1)
