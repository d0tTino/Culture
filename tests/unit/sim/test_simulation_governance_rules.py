from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from src.agents.core.agent_state import AgentActionIntent
from src.governance.rules_engine import governance_rules_engine
from src.sim.simulation import Simulation


class DummyState(SimpleNamespace):
    ip: float = 5.0
    du: float = 5.0
    age: int = 0
    is_alive: bool = True
    inheritance: float = 0.0
    short_term_memory: ClassVar[list] = []
    messages_sent_count: int = 0
    last_message_step: int = 0
    relationships: ClassVar[dict] = {}
    current_role: str = "dummy"
    steps_in_current_role: int = 0

    def update_collective_metrics(self, ip: float, du: float) -> None:
        return None


class MoveAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = DummyState()

    def get_id(self) -> str:
        return self.agent_id

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager: object | None = None,
        memory_service: object | None = None,
        knowledge_board: object | None = None,
    ) -> dict:
        return {
            "action_intent": AgentActionIntent.MOVE.value,
            "map_action": {"action": "move", "dx": 1, "dy": 0},
        }

    def update_state(self, new_state: DummyState) -> None:
        self.state = new_state


@pytest.mark.unit
@pytest.mark.asyncio
async def test_governance_rule_blocks_side_effects() -> None:
    governance_rules_engine._active_rules.clear()
    governance_rules_engine.materialize_from_proposal("no move", proposer_id="a1", approved=True)
    agent = MoveAgent("a1")
    sim = Simulation(agents=[agent])

    await sim.run_step(max_turns=1)

    assert sim.world_map.agent_positions.get("a1") == (0, 0)
    entries = sim.knowledge_board.get_full_entries()
    decision_entries = [e for e in entries if e.get("entry_type") == "governance_decision"]
    assert decision_entries
    metadata = decision_entries[-1].get("reference_metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("decision") == "rejected"
