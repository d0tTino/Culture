from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.sim.simulation import Simulation


@dataclass
class DummyState:
    ip: float = 0.0
    du: float = 0.0
    mood_level: float = 0.0
    short_term_memory: list[dict[str, Any]] = field(default_factory=list)
    messages_sent_count: int = 0
    last_message_step: int | None = None
    collective_ip: float = 0.0
    collective_du: float = 0.0
    is_alive: bool = True
    inheritance: float = 0.0
    parent_id: str | None = None
    genes: dict[str, float] = field(default_factory=dict)


class DeterministicAgent:
    def __init__(self, agent_id: str, *, mutates_snapshot: bool = False, recipient: str | None = None) -> None:
        self.agent_id = agent_id
        self._state = DummyState()
        self._mutates_snapshot = mutates_snapshot
        self._recipient = recipient

    def get_id(self) -> str:
        return self.agent_id

    @property
    def state(self) -> DummyState:
        return self._state

    def update_state(self, state: DummyState) -> None:
        self._state = state

    async def run_turn(self, **kwargs: Any) -> dict[str, Any]:
        env = kwargs["environment_perception"]
        perceived = env.setdefault("perceived_messages", [])
        if self._mutates_snapshot:
            perceived.append({"sender_id": self.agent_id, "content": "mutated"})
        return {
            "action_intent": "send_direct_message",
            "message_content": f"hello:{self.agent_id}:{len(perceived)}",
            "message_recipient_id": self._recipient,
            "map_action": {"action": "gather", "x": 1, "y": 1},
        }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tick_pipeline_parallel_equals_sequential_for_same_seed() -> None:
    agents_parallel = [DeterministicAgent("A", recipient="R"), DeterministicAgent("B", recipient="R")]
    agents_sequential = [DeterministicAgent("A", recipient="R"), DeterministicAgent("B", recipient="R")]

    sim_parallel = Simulation(agents_parallel, seed=123)
    sim_sequential = Simulation(agents_sequential, seed=123)

    try:
        out_parallel = await sim_parallel._run_step_pipeline(max_turns=2, parallel_decision=True)
        out_sequential = await sim_sequential._run_step_pipeline(max_turns=2, parallel_decision=False)

        normalized_parallel = [{k: v for k, v in entry.items() if k != "simulation_step"} for entry in out_parallel]
        normalized_sequential = [{k: v for k, v in entry.items() if k != "simulation_step"} for entry in out_sequential]

        assert normalized_parallel == normalized_sequential
    finally:
        await sim_parallel.stop_event_listener()
        sim_parallel.close()
        await sim_sequential.stop_event_listener()
        sim_sequential.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tick_pipeline_snapshot_isolation_prevents_mid_tick_bleed() -> None:
    agents = [
        DeterministicAgent("A", mutates_snapshot=True, recipient="R"),
        DeterministicAgent("B", recipient="R"),
    ]
    sim = Simulation(agents, seed=11)

    try:
        committed = await sim._run_step_pipeline(max_turns=2, parallel_decision=True)

        accepted = [entry for entry in committed if entry["merge_outcome"] == "accepted"]
        assert accepted
        # B should not observe A's local mutation because each worker gets a copied read snapshot.
        assert any(item["agent_id"] == "B" and item["message_content"].endswith(":0") for item in committed)
    finally:
        await sim.stop_event_listener()
        sim.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tick_pipeline_conflict_resolution_rules_are_deterministic() -> None:
    agents = [
        DeterministicAgent("A", recipient="same-target"),
        DeterministicAgent("B", recipient="same-target"),
        DeterministicAgent("C", recipient="same-target"),
    ]
    sim = Simulation(agents, seed=7)

    try:
        committed = await sim._run_step_pipeline(max_turns=3, parallel_decision=True)

        accepted = [entry for entry in committed if entry["merge_outcome"] == "accepted"]
        rejected = [entry for entry in committed if entry["merge_outcome"] != "accepted"]

        assert len(accepted) == 1
        assert all(item["merge_outcome"] == "rejected_resource_conflict" for item in rejected)
        assert accepted[0]["agent_id"] == "A"
    finally:
        await sim.stop_event_listener()
        sim.close()
