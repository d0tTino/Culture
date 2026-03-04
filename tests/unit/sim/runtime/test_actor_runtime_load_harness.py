from __future__ import annotations

import asyncio

import pytest

from src.sim.runtime.actor_runtime import EventEnvelope, Mailbox
from src.sim.runtime.load_test_harness import run_default_load_suite

pytestmark = pytest.mark.unit


def test_mailbox_ack_and_retry() -> None:
    mailbox = Mailbox("world", retry_timeout_s=0.0)
    mailbox.push(
        EventEnvelope(
            event_type="agent_step",
            from_actor="agent-1",
            to_actor="world",
            payload={"agent_id": "agent-1"},
            step=0,
            event_index=1,
        )
    )

    first = mailbox.receive()
    assert first is not None
    # No ACK; a second read should re-queue with incremented attempt.
    second = mailbox.receive()
    assert second is not None
    assert second.envelope.attempt == 2
    mailbox.ack(second.delivery_id)
    assert mailbox.depth() == 0


def test_default_load_suite_metrics_and_replay_determinism() -> None:
    results = asyncio.run(run_default_load_suite())
    assert [result.agent_count for result in results] == [20, 50, 100]
    for result in results:
        assert result.throughput_events_per_sec > 0
        assert result.p95_step_latency_ms >= 0
        assert result.deterministic_replay
