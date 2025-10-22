import asyncio

import pytest

from src.sim.knowledge_board import BoardEntry, KnowledgeBoard


@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_writes() -> None:
    board = KnowledgeBoard()

    async def writer(i: int) -> None:
        async with board.lock:
            board.add_entry(
                BoardEntry(content_full=f"entry-{i}", entry_type="concurrency_test"),
                "agent",
                i,
            )

    await asyncio.gather(*(writer(i) for i in range(20)))

    assert len(board.get_full_entries()) == 20
