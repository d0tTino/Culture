from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("chromadb")

from src.agents.core.base_agent import Agent
from src.agents.graphs.basic_agent_types import AgentActionOutput
from src.infra import config
from src.sim.simulation import Simulation


@pytest.mark.asyncio
async def test_create_project_via_graph() -> None:
    agent = Agent(
        agent_id="creator",
        initial_state={
            "name": "Creator",
            "current_role": "Innovator",
            "influence_points": 30,
            "data_units": 30,
        },
    )
    sim = Simulation(agents=[agent])

    action_output = AgentActionOutput(
        thought="create project",
        message_content=None,
        message_recipient_id=None,
        action_intent="create_project",
        requested_role_change=None,
        project_name_to_create="TestProj",
        project_description_for_creation="desc",
        project_id_to_join_or_leave=None,
    )

    with (
        patch.object(agent, "async_select_action_intent", AsyncMock(return_value=action_output)),
        patch(
            "src.agents.graphs.graph_nodes.generate_structured_output",
            return_value=action_output,
        ),
    ):
        await sim.run_step()

    assert len(sim.projects) == 1
    project_id = next(iter(sim.projects))
    assert agent.state.current_project_id == project_id
    assert agent.state.current_project_affiliation == "TestProj"
    assert agent.state.ip == 30 - config.IP_COST_CREATE_PROJECT
    assert agent.state.du == 30 - config.DU_COST_CREATE_PROJECT


@pytest.mark.asyncio
async def test_join_project_via_graph() -> None:
    creator = Agent(
        agent_id="creator",
        initial_state={
            "name": "Creator",
            "current_role": "Innovator",
            "influence_points": 30,
            "data_units": 30,
        },
    )
    joiner = Agent(
        agent_id="joiner",
        initial_state={
            "name": "Joiner",
            "current_role": "Analyzer",
            "influence_points": 10,
            "data_units": 10,
        },
    )
    sim = Simulation(agents=[creator, joiner])

    create_output = AgentActionOutput(
        thought="create",
        message_content=None,
        message_recipient_id=None,
        action_intent="create_project",
        requested_role_change=None,
        project_name_to_create="Joinable",
        project_description_for_creation="desc",
        project_id_to_join_or_leave=None,
    )

    with (
        patch.object(creator, "async_select_action_intent", AsyncMock(return_value=create_output)),
        patch(
            "src.agents.graphs.graph_nodes.generate_structured_output",
            return_value=create_output,
        ),
    ):
        await sim.run_step()

    project_id = next(iter(sim.projects))

    join_output = AgentActionOutput(
        thought="join",
        message_content=None,
        message_recipient_id=None,
        action_intent="join_project",
        requested_role_change=None,
        project_name_to_create=None,
        project_description_for_creation=None,
        project_id_to_join_or_leave=project_id,
    )

    with (
        patch.object(joiner, "async_select_action_intent", AsyncMock(return_value=join_output)),
        patch(
            "src.agents.graphs.graph_nodes.generate_structured_output",
            return_value=join_output,
        ),
    ):
        await sim.run_step()

    assert joiner.state.current_project_id == project_id
    assert joiner.agent_id in sim.projects[project_id]["members"]
    assert joiner.state.ip == 10 - config.IP_COST_JOIN_PROJECT
    assert joiner.state.du == 10 - config.DU_COST_JOIN_PROJECT


@pytest.mark.asyncio
async def test_leave_project_via_graph() -> None:
    agent = Agent(
        agent_id="leaver",
        initial_state={
            "name": "Leaver",
            "current_role": "Innovator",
            "influence_points": 20,
            "data_units": 20,
        },
    )
    sim = Simulation(agents=[agent])

    create_output = AgentActionOutput(
        thought="create",
        message_content=None,
        message_recipient_id=None,
        action_intent="create_project",
        requested_role_change=None,
        project_name_to_create="TempProj",
        project_description_for_creation="desc",
        project_id_to_join_or_leave=None,
    )

    with (
        patch.object(agent, "async_select_action_intent", AsyncMock(return_value=create_output)),
        patch(
            "src.agents.graphs.graph_nodes.generate_structured_output",
            return_value=create_output,
        ),
    ):
        await sim.run_step()

    project_id = next(iter(sim.projects))

    leave_output = AgentActionOutput(
        thought="leave",
        message_content=None,
        message_recipient_id=None,
        action_intent="leave_project",
        requested_role_change=None,
        project_name_to_create=None,
        project_description_for_creation=None,
        project_id_to_join_or_leave=project_id,
    )

    with (
        patch.object(agent, "async_select_action_intent", AsyncMock(return_value=leave_output)),
        patch(
            "src.agents.graphs.graph_nodes.generate_structured_output",
            return_value=leave_output,
        ),
    ):
        await sim.run_step()

    assert agent.state.current_project_id is None
    assert agent.agent_id not in sim.projects[project_id]["members"]
