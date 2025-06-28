from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .simulation import Simulation

logger = logging.getLogger(__name__)


def create_project(
    sim: Simulation,
    project_name: str,
    creator_agent_id: str,
    project_description: str | None = None,
) -> str | None:
    """Create a new project within ``sim``."""
    if not project_name or not creator_agent_id:
        logger.warning("Cannot create project: missing name or creator ID")
        return None

    project_id = f"proj_{len(sim.projects) + 1}_{int(time.time())}"
    for existing_id, existing_proj in sim.projects.items():
        if existing_proj.get("name") == project_name:
            logger.warning(
                "Cannot create project: a project named '%s' already exists (ID: %s)",
                project_name,
                existing_id,
            )
            return None

    creator_agent = next((a for a in sim.agents if a.agent_id == creator_agent_id), None)
    if not creator_agent:
        logger.warning("Cannot create project: creator agent '%s' not found", creator_agent_id)
        return None

    sim.projects[project_id] = {
        "id": project_id,
        "name": project_name,
        "description": project_description or f"Project created by {creator_agent_id}",
        "creator_id": creator_agent_id,
        "created_step": sim.current_step,
        "members": [creator_agent_id],
        "status": "active",
    }

    logger.info(
        "Project '%s' (ID: %s) created by Agent %s",
        project_name,
        project_id,
        creator_agent_id,
    )

    if sim.knowledge_board:
        project_info = (
            f"New Project Created: {project_name}\nID: {project_id}\nCreator: {creator_agent_id}"
        )
        if project_description:
            project_info += f"\nDescription: {project_description}"
        sim.knowledge_board.add_entry(
            project_info,
            creator_agent_id,
            sim.current_step,
            sim.vector.to_dict(),
        )

    if sim.discord_bot:
        embed = sim.discord_bot.create_project_embed(
            action="create",
            project_name=project_name,
            project_id=project_id,
            agent_id=creator_agent_id,
            step=sim.current_step,
        )
        task = asyncio.create_task(
            sim.discord_bot.send_simulation_update(embed=embed, agent_id=creator_agent_id)
        )
        _ = task

    return project_id


def join_project(sim: Simulation, project_id: str, agent_id: str) -> bool:
    """Add ``agent_id`` as a member of ``project_id``."""
    if project_id not in sim.projects:
        logger.warning("Cannot join project: project ID '%s' does not exist", project_id)
        return False

    project = sim.projects[project_id]
    if agent_id in project["members"]:
        logger.warning("Agent %s is already a member of project '%s'", agent_id, project["name"])
        return False

    agent_obj = next((a for a in sim.agents if a.agent_id == agent_id), None)
    if not agent_obj:
        logger.warning("Cannot join project: agent '%s' not found", agent_id)
        return False

    project["members"].append(agent_id)
    logger.info("Agent %s joined project '%s' (ID: %s)", agent_id, project["name"], project_id)

    if sim.knowledge_board:
        join_info = f"Agent {agent_id} joined Project: {project['name']} (ID: {project_id})"
        sim.knowledge_board.add_entry(
            join_info,
            agent_id,
            sim.current_step,
            sim.vector.to_dict(),
        )

    if sim.discord_bot:
        embed = sim.discord_bot.create_project_embed(
            action="join",
            project_name=project["name"],
            project_id=project_id,
            agent_id=agent_id,
            step=sim.current_step,
        )
        task = asyncio.create_task(
            sim.discord_bot.send_simulation_update(embed=embed, agent_id=agent_id)
        )
        _ = task

    return True


def leave_project(sim: Simulation, project_id: str, agent_id: str) -> bool:
    """Remove ``agent_id`` from ``project_id``."""
    if project_id not in sim.projects:
        logger.warning("Cannot leave project: project ID '%s' does not exist", project_id)
        return False

    project = sim.projects[project_id]
    if agent_id not in project["members"]:
        logger.warning("Agent %s is not a member of project '%s'", agent_id, project["name"])
        return False

    agent_obj = next((a for a in sim.agents if a.agent_id == agent_id), None)
    if not agent_obj:
        logger.warning("Cannot leave project: agent '%s' not found", agent_id)
        return False

    project["members"].remove(agent_id)
    logger.info("Agent %s left project '%s' (ID: %s)", agent_id, project["name"], project_id)

    if sim.knowledge_board:
        leave_info = f"Agent {agent_id} left Project: {project['name']} (ID: {project_id})"
        sim.knowledge_board.add_entry(
            leave_info,
            agent_id,
            sim.current_step,
            sim.vector.to_dict(),
        )

    if sim.discord_bot:
        embed = sim.discord_bot.create_project_embed(
            action="leave",
            project_name=project["name"],
            project_id=project_id,
            agent_id=agent_id,
            step=sim.current_step,
        )
        task = asyncio.create_task(
            sim.discord_bot.send_simulation_update(embed=embed, agent_id=agent_id)
        )
        _ = task

    return True


def get_project_details(sim: Simulation) -> dict[str, dict[str, Any]]:
    """Return a copy of the project's details for agent perception."""
    return sim.projects.copy()


__all__ = [
    "create_project",
    "get_project_details",
    "join_project",
    "leave_project",
]
