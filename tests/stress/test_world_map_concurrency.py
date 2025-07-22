import asyncio

import pytest

from src.sim.world_map import ResourceToken, WorldMap

pytestmark = pytest.mark.stress


@pytest.mark.asyncio
async def test_move_and_gather_concurrently() -> None:
    m = WorldMap(width=5, height=5)
    await m.add_agent("A", x=0, y=0)
    await m.add_resource(4, 0, ResourceToken.WOOD, 500)

    async def worker() -> None:
        for _ in range(50):
            await m.move("A", 1, 0)
            await m.gather("A", ResourceToken.WOOD)

    await asyncio.gather(*(worker() for _ in range(5)))

    assert m.agent_positions["A"] == (4, 0)
    bag = m.agent_resources["A"].get("wood", 0)
    remaining = m.resources.get((4, 0), {}).get("wood", 0)
    assert bag + remaining == 500
