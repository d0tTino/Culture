from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.sim.engines.domain_events import DomainEvent
from src.sim.engines.reducers import DomainEventReducer
from src.sim.runtime.step_context import StepContext


@pytest.mark.unit
def test_reducer_applies_committed_turn_side_effects() -> None:
    simulation = SimpleNamespace(
        agents=[SimpleNamespace(), SimpleNamespace()],
        current_step=4,
        current_agent_index=1,
        total_turns_executed=10,
        vector=SimpleNamespace(increment=lambda _agent_id: None),
        event_kernel=SimpleNamespace(schedule_immediate_nowait=lambda *_args, **_kwargs: None),
        _create_agent_event=lambda _index: None,
        _set_labeled_gauge=lambda *_args, **_kwargs: None,
    )
    reducer = DomainEventReducer()
    context = StepContext(max_turns=2)

    reducer.apply(
        simulation,
        context,
        [
            DomainEvent(
                domain="turn",
                name="planned_turns_committed",
                payload={
                    "committed_outputs": [{"merge_outcome": "accepted"}, {"merge_outcome": "rejected"}],
                    "turn_count": 2,
                    "accepted_turn_count": 1,
                },
            )
        ],
    )

    assert simulation.current_step == 6
    assert simulation.current_agent_index == 1
    assert simulation.total_turns_executed == 11


@pytest.mark.unit
def test_reducer_counts_scheduler_events_as_executed_turns() -> None:
    simulation = SimpleNamespace(
        agents=[SimpleNamespace()],
        current_step=2,
        current_agent_index=0,
        total_turns_executed=3,
        vector=SimpleNamespace(increment=lambda _agent_id: None),
        event_kernel=SimpleNamespace(schedule_immediate_nowait=lambda *_args, **_kwargs: None),
        _create_agent_event=lambda _index: None,
        _set_labeled_gauge=lambda *_args, **_kwargs: None,
    )
    reducer = DomainEventReducer()
    context = StepContext(max_turns=2)

    reducer.apply(
        simulation,
        context,
        [
            DomainEvent(
                domain="turn",
                name="scheduler_events_ready",
                payload={"events": [{"type": "agent_action"}, {"type": "agent_action"}]},
            )
        ],
    )

    assert simulation.total_turns_executed == 5
