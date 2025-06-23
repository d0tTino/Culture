import sys
import types

import pytest

# Stub optional heavy dependencies before importing simulation modules
neo4j_stub = types.ModuleType("neo4j")
neo4j_stub.GraphDatabase = object  # type: ignore[attr-defined]
neo4j_stub.Driver = object  # type: ignore[attr-defined]
sys.modules.setdefault("neo4j", neo4j_stub)
sys.modules.setdefault("neo4j.exceptions", types.ModuleType("neo4j.exceptions"))

from src.agents.core.agent_state import AgentActionIntent
from src.agents.core.base_agent import Agent, AgentActionOutput
from src.agents.graphs.interaction_handlers import (
    handle_create_project_node,
    handle_join_project_node,
)
from src.infra import config
from src.sim.simulation import Simulation


@pytest.mark.integration
def test_project_handlers_modify_state() -> None:
    # Create simple agents and simulation
    creator = Agent(
        agent_id="creator",
        initial_state={"name": "Creator", "influence_points": 20, "data_units": 20},
    )
    joiner = Agent(
        agent_id="joiner",
        initial_state={"name": "Joiner", "influence_points": 10, "data_units": 10},
    )
    sim = Simulation(agents=[creator, joiner])

    create_action = AgentActionOutput(
        thought="create",
        message_content=None,
        message_recipient_id=None,
        action_intent=AgentActionIntent.CREATE_PROJECT.value,
        requested_role_change=None,
        project_name_to_create="Proj",
        project_description_for_creation="desc",
        project_id_to_join_or_leave=None,
    )
    state = {
        "state": creator.state,
        "structured_output": create_action,
        "environment_perception": {"simulation": sim},
    }
    handle_create_project_node(state)

    assert len(sim.projects) == 1
    project_id = next(iter(sim.projects))
    assert creator.state.current_project_id == project_id
    assert creator.state.ip == pytest.approx(20 - config.IP_COST_CREATE_PROJECT)
    assert creator.state.du == pytest.approx(20 - config.DU_COST_CREATE_PROJECT)

    join_action = AgentActionOutput(
        thought="join",
        message_content=None,
        message_recipient_id=None,
        action_intent=AgentActionIntent.JOIN_PROJECT.value,
        requested_role_change=None,
        project_name_to_create=None,
        project_description_for_creation=None,
        project_id_to_join_or_leave=project_id,
    )
    join_state = {
        "state": joiner.state,
        "structured_output": join_action,
        "environment_perception": {"simulation": sim},
    }
    handle_join_project_node(join_state)

    assert joiner.state.current_project_id == project_id
    assert joiner.state.ip == pytest.approx(10 - config.IP_COST_JOIN_PROJECT)
    assert joiner.state.du == pytest.approx(10 - config.DU_COST_JOIN_PROJECT)
    assert joiner.agent_id in sim.projects[project_id]["members"]
