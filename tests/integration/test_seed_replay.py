import random
from typing import Any

import pytest

from src.app import create_simulation
from src.infra import event_log
from tests.utils.mock_llm import MockLLM

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_seed_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running simulations with the same seed should log identical events."""
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)
    monkeypatch.setattr(event_log, "_producer", None, raising=False)

    async def run() -> list[dict[str, Any]]:
        random.seed(1234)
        if np is not None:
            np.random.seed(1234)
        event_log._last_hash = None
        event_log._seed = None
        events: list[dict[str, Any]] = []
        real_log_event = event_log.log_event

        def capture(ev: dict[str, Any]) -> dict[str, Any]:
            out = real_log_event(ev)
            events.append(out)
            return out

        with monkeypatch.context() as m:
            m.setattr(event_log, "log_event", capture)
            with MockLLM():
                sim = create_simulation(num_agents=1, steps=1, scenario="test")
                await sim.run_step()
        return events

    first = await run()
    second = await run()

    assert first == second
