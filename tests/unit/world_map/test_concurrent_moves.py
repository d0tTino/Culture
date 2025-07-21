import asyncio

import pytest

from src.sim.world_map import WorldMap

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_concurrent_moves() -> None:
    m = WorldMap(width=5, height=5)
    await m.add_agent("A")

    async def mover() -> None:
        for _ in range(50):
            await m.move("A", 1, 0)

    await asyncio.gather(*(mover() for _ in range(5)))

    assert m.agent_positions["A"] == (4, 0)
