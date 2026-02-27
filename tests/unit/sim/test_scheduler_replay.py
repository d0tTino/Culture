import hashlib
import random
from typing import Any

import pytest

from src.sim.event_kernel import EventKernel

pytestmark = pytest.mark.unit


async def _noop() -> None:
    return None


def _command_stream(seed: int, size: int = 30) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    commands: list[dict[str, Any]] = []
    for idx in range(size):
        commands.append(
            {
                "step": rng.randint(0, 6),
                "agent_id": f"agent-{rng.randint(1, 3)}",
                "tokens": rng.randint(1, 4),
                "count": idx,
            }
        )
    return commands


async def _run_stream(seed: int) -> tuple[list[str], bytes]:
    kernel = EventKernel()
    stream = _command_stream(seed)
    for command in stream:
        kernel.set_budget(command["agent_id"], 1000)
    for command in stream:
        kernel.schedule_at_nowait(
            command["step"],
            _noop,
            agent_id=command["agent_id"],
            tokens=command["tokens"],
        )
    events = await kernel.dispatch(10_000)
    trace_hashes = [kernel.event_metadata(event)["trace_hash"] for event in events]
    trace_bytes = "\n".join(trace_hashes).encode("utf-8")
    return trace_hashes, trace_bytes


@pytest.mark.asyncio
async def test_scheduler_replay_hashes_are_byte_identical() -> None:
    left_hashes, left_bytes = await _run_stream(seed=1337)
    right_hashes, right_bytes = await _run_stream(seed=1337)

    assert left_hashes == right_hashes
    assert left_bytes == right_bytes
    assert hashlib.sha256(left_bytes).digest() == hashlib.sha256(right_bytes).digest()


@pytest.mark.asyncio
async def test_scheduler_tie_break_uses_insertion_count() -> None:
    kernel = EventKernel()

    kernel.schedule_at_nowait(2, _noop)
    kernel.schedule_at_nowait(2, _noop)
    kernel.schedule_at_nowait(2, _noop)

    events = await kernel.dispatch(10)
    metadata = [kernel.event_metadata(event) for event in events]

    assert [item["step"] for item in metadata] == [2, 2, 2]
    assert [item["count"] for item in metadata] == [0, 1, 2]
