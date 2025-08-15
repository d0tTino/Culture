import random

import pytest

from src.app import create_simulation
from src.infra.checkpoint import capture_rng_state
from src.sim.simulation import Simulation
from tests.utils.mock_llm import MockLLM


@pytest.mark.asyncio
@pytest.mark.integration
async def test_snapshot_restores_rng_state() -> None:
    with MockLLM():
        sim = create_simulation(num_agents=1, steps=1, scenario="test", seed=123)
        await sim.run_step()
        snapshot = {
            "step": sim.current_step,
            "collective_ip": sim.collective_ip,
            "collective_du": sim.collective_du,
            "knowledge_board": sim.knowledge_board.to_dict(),
            "world_map": sim.world_map.to_dict(),
            "agents": [
                {
                    "agent_id": ag.agent_id,
                    "ip": ag.state.ip,
                    "du": ag.state.du,
                    "mood": ag.state.mood_level,
                }
                for ag in sim.agents
            ],
            "seed": sim.seed,
            "rng_state": capture_rng_state(),
        }
        expected = random.random()
        Simulation.from_snapshot(snapshot)
        actual = random.random()
        assert expected == actual
