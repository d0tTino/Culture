#!/usr/bin/env python
# ruff: noqa: RUF006
import argparse
import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Optional, cast

from typing_extensions import Self

from src.agents.core import ResourceManager
from src.agents.core.agent_controller import AgentController
from src.infra import config  # Import to access MAX_PROJECT_MEMBERS
from src.infra.event_log import log_event
from src.infra.logging_config import setup_logging
from src.infra.snapshot import save_snapshot
from src.shared.typing import SimulationMessage
from src.sim.knowledge_board import KnowledgeBoard

# Use TYPE_CHECKING to avoid circular import issues if Agent needs Simulation later
if TYPE_CHECKING:
    from src.agents.core.base_agent import Agent
    from src.agents.memory.vector_store import ChromaVectorStoreManager
    from src.interfaces.discord_bot import SimulationDiscordBot

# Configure the logger for this module
logger = logging.getLogger(__name__)


class Simulation:
    """
    Manages the simulation environment, agents, and time steps.

    Attributes:
        steps_to_run (int): Number of steps the simulation should run (set externally).
    """

    def __init__(
        self: Self,
        agents: list["Agent"],
        vector_store_manager: Optional["ChromaVectorStoreManager"] = None,
        scenario: str = "",
        discord_bot: Optional["SimulationDiscordBot"] = None,
    ) -> None:
        """
        Initializes the Simulation instance.

        Args:
            agents (list[Agent]): A list of Agent instances participating
                                  in the simulation.
            vector_store_manager (Optional[ChromaVectorStoreManager]): Manager for
                                  vector-based agent memory storage and retrieval.
            scenario (str): Description of the simulation scenario that provides
                context for agent interactions.
            discord_bot (Optional[SimulationDiscordBot]): Discord bot for sending
                simulation updates to Discord.
        """
        self.agents: list[Agent] = agents
        self.current_step: int = 0
        self.current_agent_index: int = 0
        self.last_completed_agent_index: int | None = None
        self.steps_to_run: int = 0  # Number of steps to run, set externally
        self.total_turns_executed = 0
        self.resource_manager = ResourceManager(config.MAX_IP_PER_TICK, config.MAX_DU_PER_TICK)
        self.simulation_complete = False
        # Add other simulation-wide state if needed (e.g., environment properties)
        # self.environment_state = {}

        # --- Store the simulation scenario ---
        self.scenario = scenario
        if scenario:
            logger.info(f"Simulation initialized with scenario: {scenario}")
        else:
            logger.warning("Simulation initialized without a scenario description.")

        # --- NEW: Initialize Knowledge Board ---
        self.knowledge_board = KnowledgeBoard()
        logger.info("Simulation initialized with Knowledge Board.")

        # --- NEW: Initialize Project Tracking ---
        self.projects: dict[
            str, dict[str, Any]
        ] = {}  # Structure: {project_id: {name, creator_id, members}}
        logger.info("Simulation initialized with project tracking system.")

        # --- NEW: Initialize Collective Metrics ---
        self.collective_ip: float = 0.0
        self.collective_du: float = 0.0
        logger.info("Simulation initialized with collective IP/DU tracking.")

        # --- Store the vector store manager ---
        self.vector_store_manager = vector_store_manager
        if vector_store_manager:
            logger.info("Simulation initialized with vector store manager for memory persistence.")
        else:
            logger.warning(
                "Simulation initialized without vector store manager. "
                "Memory will not be persisted."
            )
        self._last_memory_prune_step = 0

        # --- Store Discord bot ---
        self.discord_bot = discord_bot
        if discord_bot:
            logger.info("Simulation initialized with Discord bot for sending updates.")
        else:
            logger.info(
                "Simulation initialized without Discord bot. No Discord updates will be sent."
            )

        # --- Store broadcasts from the previous step ---
        self.last_step_messages: list[SimulationMessage] = []
        logger.info("Initialized storage for last step's messages.")
        # --- End NEW ---

        self.pending_messages_for_next_round: list[SimulationMessage] = []
        # Messages available for agents to perceive in the current round.
        self.messages_to_perceive_this_round: list[
            SimulationMessage
        ] = []  # THIS WILL BE THE ACCUMULATOR FOR THE CURRENT ROUND

        self.track_collective_metrics: bool = True

        if not self.agents:
            logger.warning("Simulation initialized with zero agents.")
        else:
            logger.info(f"Simulation initialized with {len(self.agents)} agents:")
            for agent in self.agents:
                logger.info(f"  - {agent.get_id()}")

            # Initialize collective metrics based on starting agent states
            self._update_collective_metrics()
            logger.info(
                f"Initial collective metrics - IP: {self.collective_ip:.1f}, "
                f"DU: {self.collective_du:.1f}"
            )

        # --- Message Handling ---
        # Messages generated by agents in the current round, to be perceived in the next.
        # self.pending_messages_for_next_round: list[dict[str, Any]] = [] # Already initialized above
        # Messages available for agents to perceive in the current round.
        # self.messages_to_perceive_this_round: list[dict[str, Any]] = [] # Already initialized above

    # Add method to update collective metrics
    def _update_collective_metrics(self: Self) -> None:
        """
        Updates the collective IP and DU metrics by summing across all agents.
        """
        total_ip = 0.0
        total_du = 0.0

        for agent in self.agents:
            agent_state = agent.state
            total_ip += agent_state.ip
            total_du += agent_state.du

        self.collective_ip = total_ip
        self.collective_du = total_du

        if self.track_collective_metrics:
            current_collective_ip = sum(agent.state.ip for agent in self.agents)
            current_collective_du = sum(agent.state.du for agent in self.agents)
            for agent_instance in self.agents:
                AgentController(agent_instance.state).update_collective_metrics(
                    current_collective_ip, current_collective_du
                )

        # current_round = (self.current_step -1) // len(self.agents) # Not clearly used, commenting out

    def get_other_agents_public_state(self: Self, current_agent_id: str) -> list[dict[str, Any]]:
        """
        Returns a list of public state information for all agents other than the current one.

        Args:
            current_agent_id (str): The ID of the agent whose perspective this is from.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, each representing another agent's public state.
        """
        other_agents_info = []
        for agent in self.agents:
            if agent.agent_id != current_agent_id:
                # Ensure agent.state is the Pydantic model AgentState
                if hasattr(agent, "state") and hasattr(agent.state, "model_dump"):
                    public_info = {
                        "agent_id": agent.agent_id,
                        "name": agent.state.name,
                        "role": agent.state.current_role,
                        "mood": agent.state.mood_value,
                        "current_project_id": agent.state.current_project_id,
                        # Add other relevant public fields, avoid sensitive internal state
                    }
                    other_agents_info.append(public_info)
                else:
                    # Fallback or log warning if state structure is not as expected
                    logger.warning(
                        f"Agent {agent.agent_id} state is not a Pydantic model or lacks model_dump, skipping public state."
                    )
        return other_agents_info

    async def send_discord_update(
        self: Self,
        message: Optional[str] = None,
        embed: Optional[object] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """
        Send an update to Discord if the discord_bot is available.

        Args:
            message (Optional[str]): The text message to send to Discord
            embed (Optional[object]): The embed object to send to Discord
        """
        if self.discord_bot:
            # Use asyncio.create_task to avoid blocking the simulation
            _ = asyncio.create_task(
                self.discord_bot.send_simulation_update(
                    content=message, embed=embed, agent_id=agent_id
                )
            )

    async def run_step(self: Self, max_turns: int = 1) -> int:
        """
        Runs the simulation for a specified number of agent turns.
        Each turn increments the global simulation step.

        Args:
            max_turns (int): Maximum number of agent turns to run.

        Returns:
            int: The actual number of agent turns executed.
        """
        turn_counter_this_run_step = 0

        for _ in range(max_turns):
            if not self.agents:
                logger.warning("No agents in simulation to run.")
                break

            # Increment current_step for this agent's turn. current_step becomes 1-indexed.
            self.current_step += 1

            agent_to_run_index = self.current_agent_index
            agent = self.agents[agent_to_run_index]
            agent_id = agent.agent_id
            current_agent_state = agent.state

            # At the start of a new round (first agent), clear messages_to_perceive_this_round
            # and populate it from what was pending for the next round.
            if agent_to_run_index == 0:
                self.messages_to_perceive_this_round = list(self.pending_messages_for_next_round)
                self.pending_messages_for_next_round = (
                    []
                )  # Clear pending for the new round accumulation
                logger.debug(
                    f"Turn {self.current_step} (Agent {agent_id}, Index 0): Initialized messages_to_perceive_this_round "
                    f"with {len(self.messages_to_perceive_this_round)} messages from pending_messages_for_next_round."
                )

            logger.info(
                f"--- Processing for Global Turn {self.current_step} (Agent: {agent_id}) ---"
            )
            logger.debug(
                f"Turn {self.current_step} (Agent {agent_id}): "
                f"Messages available for perception at start of this agent's turn (from messages_to_perceive_this_round): "
                f"{len(self.messages_to_perceive_this_round)} messages."
            )
            # messages_for_this_agent_turn will be used by the agent.run_turn via perception_data
            # It draws from self.messages_to_perceive_this_round

            # Log start of agent's turn more clearly with ROUND (self.current_step)
            logger.info(
                f"Running turn for Agent {agent_id} in Global Turn {self.current_step} (Turn {turn_counter_this_run_step + 1}/{max_turns} of this run_step call)"
            )

            # --- Prepare Perception Data ---
            # Use messages from the current round's perception pool
            messages_for_this_agent_turn: list[SimulationMessage] = list(
                self.messages_to_perceive_this_round
            )  # Make a copy for this agent
            perception_data = {
                "other_agents_state": self.get_other_agents_public_state(agent_id),
                "perceived_messages": [
                    msg
                    for msg in messages_for_this_agent_turn
                    if not msg.get("recipient_id") or msg.get("recipient_id") == agent_id
                ],
                "knowledge_board_content": self.knowledge_board.get_recent_entries_for_prompt(
                    max_entries=config.MAX_KB_ENTRIES_FOR_PERCEPTION
                ),
                "scenario_description": self.scenario,
                "available_projects": self.projects,  # Assuming self.projects is updated
                "simulation": self,
                "collective_ip": self.collective_ip,
                "collective_du": self.collective_du,
            }
            logger.debug(
                f"Agent {agent_id} (Round {self.current_step}): perceived_messages = {perception_data['perceived_messages']}"
            )

            ip_start = current_agent_state.ip
            du_start = current_agent_state.du

            # Run the agent's turn with perception data
            agent_output = await agent.run_turn(
                simulation_step=self.current_step,
                environment_perception=perception_data,
                vector_store_manager=self.vector_store_manager,
                knowledge_board=self.knowledge_board,
            )

            self.resource_manager.cap_tick(
                ip_start=ip_start, du_start=du_start, obj=current_agent_state
            )

            # --- Process Agent Output ---
            this_agent_turn_generated_messages = []  # Local list for this agent's output this turn

            message_content = agent_output.get("message_content")
            message_recipient_id = agent_output.get("message_recipient_id")
            action_intent_str = agent_output.get("action_intent", "idle")

            if message_content:
                message_type = (
                    "broadcast"
                    if message_recipient_id is None
                    else f"targeted to {message_recipient_id}"
                )
                logger.info(
                    f"Agent {agent_id} sent a message ({message_type}) at Global Turn {self.current_step}: "
                    f"\\'{message_content}\\'"
                )

                msg_data = cast(
                    SimulationMessage,
                    {
                        "step": self.current_step,
                        "sender_id": agent_id,
                        "recipient_id": message_recipient_id,
                        "content": message_content,
                        "action_intent": action_intent_str,
                        "sentiment_score": None,
                    },
                )
                this_agent_turn_generated_messages.append(msg_data)
                current_agent_state.messages_sent_count += 1
                current_agent_state.last_message_step = self.current_step

            # Add messages generated by this agent to:
            # 1. pending_messages_for_next_round (for the *next* full round of all agents)
            # 2. messages_to_perceive_this_round (so subsequent agents in *this current* round can see them)
            self.pending_messages_for_next_round.extend(this_agent_turn_generated_messages)
            self.messages_to_perceive_this_round.extend(
                this_agent_turn_generated_messages
            )  # ADDED

            logger.debug(
                f"SIM_DEBUG: After Agent {agent_id}'s turn in Global Turn {self.current_step}: "
                f"pending_messages_for_next_round now has {len(self.pending_messages_for_next_round)} messages. "
                f"messages_to_perceive_this_round now has {len(self.messages_to_perceive_this_round)} messages."
            )

            # Update agent counters and report
            logger.info(f"Agent {agent_id} completed Global Turn {self.current_step}:")
            logger.info(f"  - IP: {current_agent_state.ip:.1f} (from {current_agent_state.ip})")
            logger.info(f"  - DU: {current_agent_state.du:.1f} (from {current_agent_state.du})")

            # Update the agent state in the simulation's list of agents
            self.agents[
                agent_to_run_index
            ] = agent  # Ensure the agent object itself is updated if it was replaced
            self.agents[agent_to_run_index].update_state(current_agent_state)

            # Determine next agent index based on role change event this turn
            # If agent changed role this turn, retain the same index for immediate extra turn
            try:
                # Check for a 'role_change' memory entry at this simulation step
                has_role_change = any(
                    mem.get("type") == "role_change" and mem.get("step") == self.current_step
                    for mem in current_agent_state.short_term_memory
                )
                if has_role_change:
                    logger.debug(
                        f"Agent {agent_id} changed role at step {self.current_step}, retaining turn for index {agent_to_run_index}"
                    )
                    self.current_agent_index = agent_to_run_index
                else:
                    # Normal rotation to next agent
                    self.current_agent_index = (agent_to_run_index + 1) % len(self.agents)
            except Exception:
                # Fallback to normal rotation on any error
                self.current_agent_index = (agent_to_run_index + 1) % len(self.agents)

            # --- Log Agent A's relationship to B if they exist (USER REQUEST) ---
            for ag_check in self.agents:
                if ag_check.agent_id == "agent_a_innovator_conflict":
                    rel_score_a_to_b = ag_check.state.relationships.get(
                        "agent_b_analyzer_conflict", "N/A_IN_SIM_LOG"
                    )
                    logger.debug(
                        f"SIM_DEBUG (End of run_step for {agent.agent_id}, current_step: {self.current_step}): Agent A's ({ag_check.agent_id}) relationship to B is {rel_score_a_to_b} (id(state): {id(ag_check.state)}, id(rels): {id(ag_check.state.relationships)})"
                    )
                    break
            # --- End USER REQUEST ---

            # --- End of Turn Processing ---
            self.agents[agent_to_run_index] = agent  # Update the agent in the list
            self.last_completed_agent_index = agent_to_run_index  # Set this before advancing

            # Update collective metrics based on agent's final state for the turn
            self._update_collective_metrics()
            logger.info(
                f"Collective metrics after Global Turn {self.current_step} - IP: {self.collective_ip:.1f}, "
                f"DU: {self.collective_du:.1f}"
            )
            log_event(
                {
                    "type": "agent_action",
                    "agent_id": agent_id,
                    "step": self.current_step,
                    "action_intent": action_intent_str,
                    "ip": current_agent_state.ip,
                    "du": current_agent_state.du,
                }
            )

            if self.current_step % 100 == 0:
                snapshot = {
                    "step": self.current_step,
                    "collective_ip": self.collective_ip,
                    "collective_du": self.collective_du,
                    "knowledge_board": self.knowledge_board.to_dict(),
                    "agents": [
                        {
                            "agent_id": ag.agent_id,
                            "ip": ag.state.ip,
                            "du": ag.state.du,
                            "mood": ag.state.mood_level,
                        }
                        for ag in self.agents
                    ],
                }
                save_snapshot(self.current_step, snapshot)
                log_event({"type": "snapshot", **snapshot})

            # Advance to the next agent for the next turn
            self.current_agent_index = (agent_to_run_index + 1) % len(self.agents)
            self.total_turns_executed += 1
            turn_counter_this_run_step += 1

        if (
            self.vector_store_manager
            and self.current_step - self._last_memory_prune_step
            >= config.MEMORY_STORE_PRUNE_INTERVAL_STEPS
        ):
            try:
                self.vector_store_manager.prune(int(config.MEMORY_STORE_TTL_SECONDS))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Memory store prune failed: %s", exc)
            self._last_memory_prune_step = self.current_step

        return turn_counter_this_run_step  # Return number of agent turns actually processed

    async def async_run(self: Self, num_steps: int) -> None:
        """
        Runs the simulation for a specified number of steps asynchronously.

        Args:
            num_steps (int): Number of steps to run.
        """
        logger.info(f"Starting simulation run for {num_steps} steps (async)")
        total_steps_executed = 0
        import asyncio

        start_time = time.time()
        for step in range(num_steps):
            await asyncio.sleep(0.1)  # Optional: Add a small delay between steps
            steps = await self.run_step(1)  # If run_step needs to be async, refactor it as well
            total_steps_executed += steps
        elapsed_time = time.time() - start_time
        logger.info(
            "Simulation completed "
            f"{total_steps_executed} steps in {elapsed_time:.2f} seconds (async)"
        )

    def apply_event(self: Self, event: dict[str, Any]) -> None:
        """Apply an event from the Redpanda log to the simulation."""
        if event.get("type") == "agent_action":
            aid = event.get("agent_id")
            for agent in self.agents:
                if agent.agent_id == aid:
                    if "ip" in event:
                        agent.state.ip = float(event["ip"])
                    if "du" in event:
                        agent.state.du = float(event["du"])
                    break
            step = event.get("step")
            if isinstance(step, int) and step > self.current_step:
                self.current_step = step
        elif event.get("type") == "environment_change":
            env = event.get("env", {})
            if isinstance(env, dict):
                from src.infra.checkpoint import restore_environment

                restore_environment(env)

    async def run_turns_concurrent(self: Self, agents: list["Agent"]) -> list[dict[str, Any]]:
        """Run a batch of agent turns concurrently.

        Each agent executes ``run_turn`` simultaneously using :func:`asyncio.gather`.
        This helper is useful for stress testing large simulations where sequential
        execution would be too slow.

        Args:
            agents: The agents whose turns should be executed.

        Returns:
            A list of dictionaries returned by each agent's ``run_turn``.
        """

        start_step = self.current_step + 1
        tasks = [
            agent.run_turn(
                simulation_step=start_step + idx,
                environment_perception={},
                vector_store_manager=self.vector_store_manager,
                knowledge_board=self.knowledge_board,
            )
            for idx, agent in enumerate(agents)
        ]

        results = await asyncio.gather(*tasks)

        self.current_step += len(agents)
        self.total_turns_executed += len(agents)
        self.current_agent_index = (self.current_agent_index + len(agents)) % len(self.agents)

        return list(results)

    def create_project(
        self: Self,
        project_name: str,
        creator_agent_id: str,
        project_description: Optional[str] = None,
    ) -> Optional[str]:
        """
        Allows an agent to create a new project.

        Args:
            project_name (str): The name of the new project
            creator_agent_id (str): The ID of the agent creating the project
            project_description (str, optional): A description of the project's purpose

        Returns:
            Optional[str]: The ID of the newly created project, or None if creation failed
        """
        if not project_name or not creator_agent_id:
            logger.warning("Cannot create project: missing name or creator ID")
            return None

        # Create a unique project ID
        project_id = f"proj_{len(self.projects) + 1}_{int(time.time())}"

        # Check if a project with this name already exists
        for existing_id, existing_proj in self.projects.items():
            if existing_proj.get("name") == project_name:
                logger.warning(
                    f"Cannot create project: a project named '{project_name}' already exists "
                    f"(ID: {existing_id})"
                )
                return None

        # Find the creator agent
        creator_agent = None
        for agent in self.agents:
            if agent.agent_id == creator_agent_id:
                creator_agent = agent
                break

        if not creator_agent:
            logger.warning(f"Cannot create project: creator agent '{creator_agent_id}' not found")
            return None

        # Create the project
        self.projects[project_id] = {
            "id": project_id,
            "name": project_name,
            "description": project_description or f"Project created by {creator_agent_id}",
            "creator_id": creator_agent_id,
            "created_step": self.current_step,
            "members": [creator_agent_id],  # Creator is automatically a member
            "status": "active",
        }

        logger.info(
            f"Project '{project_name}' (ID: {project_id}) created by Agent {creator_agent_id}"
        )

        # Add to Knowledge Board
        if self.knowledge_board:
            project_info = (
                f"New Project Created: {project_name}\nID: {project_id}\n"
                f"Creator: {creator_agent_id}"
            )
            if project_description:
                project_info += f"\nDescription: {project_description}"
            self.knowledge_board.add_entry(project_info, creator_agent_id, self.current_step)

        # Send Discord notification if bot is available
        if self.discord_bot:
            embed = self.discord_bot.create_project_embed(
                action="create",
                project_name=project_name,
                project_id=project_id,
                agent_id=creator_agent_id,
                step=self.current_step,
            )
            _ = asyncio.create_task(
                self.discord_bot.send_simulation_update(embed=embed, agent_id=creator_agent_id)
            )

        return project_id

    def join_project(self: Self, project_id: str, agent_id: str) -> bool:
        """
        Adds an agent to an existing project as a member.

        Args:
            project_id (str): The ID of the project to join
            agent_id (str): The ID of the agent joining the project

        Returns:
            bool: True if successfully joined, False otherwise
        """

        # Ensure the project exists
        if project_id not in self.projects:
            logger.warning(f"Cannot join project: project ID '{project_id}' does not exist")
            return False

        project = self.projects[project_id]

        # Ensure the project is active
        if project.get("status") != "active":
            logger.warning(
                f"Cannot join project: project '{project['name']}' is not active "
                f"(status: {project.get('status')})"
            )
            return False

        # Check if agent is already a member
        if agent_id in project["members"]:
            logger.info(f"Agent {agent_id} is already a member of project '{project['name']}'")
            return True

        # Check if the project has reached the maximum number of members
        if len(project["members"]) >= config.MAX_PROJECT_MEMBERS:
            logger.warning(
                f"Cannot join project '{project['name']}': maximum member limit "
                f"({config.MAX_PROJECT_MEMBERS}) reached"
            )
            return False

        # Find the agent
        agent_obj = None
        for agent in self.agents:
            if agent.agent_id == agent_id:
                agent_obj = agent
                break

        if not agent_obj:
            logger.warning(f"Cannot join project: agent '{agent_id}' not found")
            return False

        # Add the agent to the project
        project["members"].append(agent_id)

        logger.info(f"Agent {agent_id} joined project '{project['name']}' (ID: {project_id})")

        # Record the event in the Knowledge Board
        if self.knowledge_board:
            join_info = f"Agent {agent_id} joined Project: {project['name']} (ID: {project_id})"
            self.knowledge_board.add_entry(join_info, agent_id, self.current_step)

        # Send Discord notification if bot is available
        if self.discord_bot:
            embed = self.discord_bot.create_project_embed(
                action="join",
                project_name=project["name"],
                project_id=project_id,
                agent_id=agent_id,
                step=self.current_step,
            )
            _ = asyncio.create_task(
                self.discord_bot.send_simulation_update(embed=embed, agent_id=agent_id)
            )

        return True

    def leave_project(self: Self, project_id: str, agent_id: str) -> bool:
        """
        Removes an agent from a project they are currently a member of.

        Args:
            project_id (str): The ID of the project to leave
            agent_id (str): The ID of the agent leaving the project

        Returns:
            bool: True if successfully left, False otherwise
        """
        # Ensure the project exists
        if project_id not in self.projects:
            logger.warning(f"Cannot leave project: project ID '{project_id}' does not exist")
            return False

        project = self.projects[project_id]

        # Check if agent is a member
        if agent_id not in project["members"]:
            logger.warning(f"Agent {agent_id} is not a member of project '{project['name']}'")
            return False

        # Find the agent
        agent_obj = None
        for agent in self.agents:
            if agent.agent_id == agent_id:
                agent_obj = agent
                break

        if not agent_obj:
            logger.warning(f"Cannot leave project: agent '{agent_id}' not found")
            return False

        # Remove the agent from the project
        project["members"].remove(agent_id)

        logger.info(f"Agent {agent_id} left project '{project['name']}' (ID: {project_id})")

        # Record the event in the Knowledge Board
        if self.knowledge_board:
            leave_info = f"Agent {agent_id} left Project: {project['name']} (ID: {project_id})"
            self.knowledge_board.add_entry(leave_info, agent_id, self.current_step)

        # Send Discord notification if bot is available
        if self.discord_bot:
            embed = self.discord_bot.create_project_embed(
                action="leave",
                project_name=project["name"],
                project_id=project_id,
                agent_id=agent_id,
                step=self.current_step,
            )
            _ = asyncio.create_task(
                self.discord_bot.send_simulation_update(embed=embed, agent_id=agent_id)
            )

        return True

    def get_project_details(self: Self) -> dict[str, dict[str, Any]]:
        """
        Returns a dictionary containing details of all projects for agent perception.

        Returns:
            dict: Dictionary mapping project IDs to project details
        """
        return self.projects.copy()

    # --- Optional helper methods for future use ---
    # def get_environment_view(self, agent: 'Agent'):
    #     """Provides the agent with its perception of the environment."""
    #     # To be implemented: return relevant state based on agent position, sensors etc.
    #     return {"global_time": self.current_step}

    # def execute_action(self, agent: 'Agent', action: str):
    #     """Handles the execution of an agent's chosen action."""
    #     # To be implemented: update agent state, environment state based on action
    #     logger.info(f"Agent {agent.get_id()} performs action: {action}")

    # def update_environment(self):
    #      """Updates the global environment state after agent actions."""
    #      # To be implemented
    #      pass


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Culture.ai simulation.")
    parser.add_argument(
        "--steps", type=int, default=5, help="Number of steps to run the simulation for."
    )
    parser.add_argument(
        "--agents", type=int, default=3, help="Number of agents to create for the simulation."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="Collaborative problem-solving session",
        help="Scenario description for the simulation.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for running the simulation.
    """
    args = parse_args()

    # Configure logging
    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.verbosity))

    # Test DSPy modules
    logging.info("SIMULATION: Attempting to import DSPy role_thought_generator as a test...")
    try:
        logging.info("SIMULATION: Successfully imported DSPy role_thought_generator!")
    except Exception as e:
        logging.error(f"SIMULATION: Failed to import DSPy role_thought_generator: {e}")

    # Test DSPy action intent selector
    logging.info("SIMULATION: Attempting to import DSPy action_intent_selector as a test...")
    try:
        from src.agents.dspy_programs.action_intent_selector import get_optimized_action_selector

        action_selector = get_optimized_action_selector()
        logging.info(
            "SIMULATION: Successfully imported and initialized DSPy action_intent_selector!"
        )

        # Run a quick test
        test_example = {
            "agent_role": "Facilitator",
            "current_situation": "Starting a new simulation.",
            "agent_goal": "Help the group collaborate effectively.",
            "available_actions": [
                "propose_idea",
                "ask_clarification",
                "continue_collaboration",
                "idle",
            ],
        }

        try:
            prediction = action_selector(**test_example)
            logging.info(
                "SIMULATION: Action selector test successful! Selected action: "
                f"{prediction.chosen_action_intent}"
            )
        except Exception as e:
            logging.error(f"SIMULATION: Action selector test call failed: {e}")
    except Exception as e:
        logging.error(f"SIMULATION: Failed to import/initialize DSPy action_intent_selector: {e}")
        import traceback

        logging.error(f"SIMULATION: {traceback.format_exc()}")

    # Create agents for the simulation
    from src.agents.core.base_agent import Agent

    agents = []
    for i in range(args.agents):
        agent = Agent(agent_id=f"agent_{i + 1}")
        agents.append(agent)

    # Create the simulation with the specified agents
    sim = Simulation(agents=agents, scenario=args.scenario)

    # Run the simulation
    import asyncio

    asyncio.run(sim.async_run(num_steps=args.steps))


# Call the main function when the script is run directly
if __name__ == "__main__":
    main()
