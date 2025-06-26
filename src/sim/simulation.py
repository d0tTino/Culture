#!/usr/bin/env python
# ruff: noqa: RUF006
import argparse
import asyncio
import logging
import time
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

from pydantic import ValidationError
from typing_extensions import Self

from src.agents.core import ResourceManager
from src.agents.core.agent_controller import AgentController
from src.agents.core.agent_state import AgentActionIntent
from src.agents.memory.vector_store import ChromaDBException
from src.governance import evaluate_policy
from src.infra import config  # Import to access MAX_PROJECT_MEMBERS
from src.infra.event_log import log_event
from src.infra.logging_config import setup_logging
from src.infra.snapshot import compute_trace_hash, load_snapshot, save_snapshot
from src.interfaces.dashboard_backend import (
    SimulationEvent,
    emit_event,
    emit_map_action_event,
    event_queue,
)
from src.shared.typing import SimulationMessage
from src.sim.event_kernel import EventKernel
from src.sim.graph_knowledge_board import GraphKnowledgeBoard
from src.sim.knowledge_board import KnowledgeBoard
from src.sim.version_vector import VersionVector
from src.sim.world_map import ResourceToken, StructureType, WorldMap

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
        # Reload configuration to pick up any environment overrides set in tests
        config.load_config()

        self.agents: list[Agent] = agents
        self.current_step: int = 0
        self.current_agent_index: int = 0
        self.last_completed_agent_index: int | None = None
        self.steps_to_run: int = 0  # Number of steps to run, set externally
        self.total_turns_executed = 0
        self.resource_manager = ResourceManager(config.MAX_IP_PER_TICK, config.MAX_DU_PER_TICK)
        self.simulation_complete = False
        self.event_kernel = EventKernel()
        self.vector = VersionVector()
        self.agent_initial_token_budget = int(config.get_config("AGENT_TOKEN_BUDGET"))
        # Add other simulation-wide state if needed (e.g., environment properties)
        # self.environment_state = {}

        # --- Store the simulation scenario ---
        self.scenario = scenario
        if scenario:
            logger.info(f"Simulation initialized with scenario: {scenario}")
        else:
            logger.warning("Simulation initialized without a scenario description.")

        # --- NEW: Initialize Knowledge Board ---
        if config.KNOWLEDGE_BOARD_BACKEND == "graph":
            self.knowledge_board = GraphKnowledgeBoard()
            logger.info("Simulation initialized with Graph Knowledge Board.")
        else:
            self.knowledge_board = KnowledgeBoard()
            logger.info("Simulation initialized with Knowledge Board.")

        # Initialize world map and place agents
        self.world_map = WorldMap()
        for idx, ag in enumerate(self.agents):
            self.world_map.add_agent(ag.agent_id, x=idx, y=0)
        logger.info("Simulation initialized with world map.")

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
        self._last_consolidation_step = 0
        self._last_trace_hash = ""

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
                self.event_kernel.set_budget(agent.get_id(), self.agent_initial_token_budget)

            # Initialize collective metrics based on starting agent states
            self._update_collective_metrics()
            logger.info(
                f"Initial collective metrics - IP: {self.collective_ip:.1f}, "
                f"DU: {self.collective_du:.1f}"
            )
            # Prime the event scheduler with the first agent turn
            first_agent = self.agents[0]
            self.event_kernel.schedule_at_nowait(
                self.current_step,
                self._create_agent_event(0),
                agent_id=first_agent.get_id(),
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

    def spawn_agent(self: Self, agent: "Agent", *, inheritance: float = 0.0) -> None:
        """Add a new agent to the simulation."""
        agent.state.ip += inheritance
        self.agents.append(agent)
        self.world_map.add_agent(agent.agent_id, x=len(self.agents) - 1, y=0)
        self._update_collective_metrics()

    def retire_agent(self: Self, agent: "Agent") -> None:
        """Retire an agent and compute inheritance."""
        state = agent.state
        state.is_alive = False
        state.inheritance = state.ip + state.du
        state.ip = 0.0
        state.du = 0.0
        if self.knowledge_board:
            self.knowledge_board.add_entry(
                f"Agent {agent.agent_id} retired",
                agent.agent_id,
                self.current_step,
                self.vector.to_dict(),
            )
        agent.update_state(state)

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

    async def _run_agent_turn(self: Self, agent_index: int) -> None:
        """Execute a single agent turn and schedule the next."""
        if not self.agents:
            logger.warning("No agents in simulation to run.")
            return

        # Increment step and select agent
        self.current_step += 1
        agent = self.agents[agent_index]
        agent_id = agent.agent_id
        self.vector.increment(agent_id)
        current_agent_state = agent.state

        if hasattr(current_agent_state, "age"):
            current_agent_state.age += 1
            max_age = int(config.get_config("MAX_AGENT_AGE"))
            if current_agent_state.age >= max_age:
                self.retire_agent(agent)
                self.current_agent_index = (agent_index + 1) % len(self.agents)
                return

        if not getattr(current_agent_state, "is_alive", True):
            self.current_agent_index = (agent_index + 1) % len(self.agents)
            return

        perception_data: dict[str, Any] = {}
        has_role_change = False
        turn_counter_this_run_step = 0
        next_agent_index = (agent_index + 1) % len(self.agents)

        if agent_index == 0:
            self.messages_to_perceive_this_round = list(self.pending_messages_for_next_round)
            self.pending_messages_for_next_round = []

            agent_to_run_index = self.current_agent_index
            agent = self.agents[agent_to_run_index]
            agent_id = agent.agent_id
            current_agent_state = agent.state

            if not getattr(current_agent_state, "is_alive", True):
                self.current_agent_index = (agent_to_run_index + 1) % len(self.agents)
                return

            # At the start of a new round (first agent), clear messages_to_perceive_this_round
            # and populate it from what was pending for the next round.
            if agent_to_run_index == 0:
                self.messages_to_perceive_this_round = list(self.pending_messages_for_next_round)
                self.pending_messages_for_next_round = []  # Clear pending for the new round accumulation
                logger.debug(
                    f"Turn {self.current_step} (Agent {agent_id}, Index 0): Initialized messages_to_perceive_this_round "
                    f"with {len(self.messages_to_perceive_this_round)} messages from pending_messages_for_next_round."
                )

        ip_start = current_agent_state.ip
        du_start = current_agent_state.du

        agent_output = await agent.run_turn(
            simulation_step=self.current_step,
            environment_perception=perception_data,
            vector_store_manager=self.vector_store_manager,
            knowledge_board=self.knowledge_board,
        )

        self.resource_manager.cap_tick(
            ip_start=ip_start, du_start=du_start, obj=current_agent_state
        )

        this_agent_turn_generated_messages: list[SimulationMessage] = []

        message_content = agent_output.get("message_content")
        message_recipient_id = agent_output.get("message_recipient_id")
        action_intent_str = agent_output.get("action_intent", "idle")
        map_action = agent_output.get("map_action")

        allowed = await evaluate_policy(action_intent_str)
        if not allowed:
            action_intent_str = AgentActionIntent.IDLE.value
            message_content = None
            message_recipient_id = None
            map_action = None

        if message_content:
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

            # If the message is a proposal to the Knowledge Board, record it
            if (
                self.knowledge_board
                and action_intent_str == AgentActionIntent.PROPOSE_IDEA.value
                and message_recipient_id is None
            ):
                self.knowledge_board.add_entry(
                    message_content,
                    agent_id,
                    self.current_step,
                    self.vector.to_dict(),
                )

        self.agents[agent_index] = agent
        self.agents[agent_index].update_state(current_agent_state)

        # Add messages generated by this agent to:
        # 1. pending_messages_for_next_round (for the *next* full round of all agents)
        # 2. messages_to_perceive_this_round (so subsequent agents in *this current* round can see them)
        self.pending_messages_for_next_round.extend(this_agent_turn_generated_messages)
        self.messages_to_perceive_this_round.extend(this_agent_turn_generated_messages)

        if hasattr(current_agent_state, "short_term_memory"):
            stm = current_agent_state.short_term_memory
            if stm:
                last_mem = stm[-1]
                if (
                    isinstance(last_mem, dict)
                    and last_mem.get("type") == "role_change"
                    and last_mem.get("step") == self.current_step
                ):
                    has_role_change = True

        if isinstance(map_action, dict):
            action_type = map_action.get("action")
            if action_type == "move":
                if "x" in map_action and "y" in map_action:
                    tx = int(map_action.get("x", 0))
                    ty = int(map_action.get("y", 0))
                    pos = self.world_map.move_to(agent_id, tx, ty, vector=self.vector.to_dict())
                else:
                    dx = int(map_action.get("dx", 0))
                    dy = int(map_action.get("dy", 0))
                    pos = self.world_map.move(agent_id, dx, dy, vector=self.vector.to_dict())
                action_details = {"position": pos}
                start_ip = current_agent_state.ip
                if config.MAP_MOVE_DU_COST > 0:
                    try:
                        from src.infra.ledger import ledger

                        aid = ledger.open_auction("move")
                        ledger.place_bid(aid, agent_id, config.MAP_MOVE_DU_COST)
                        ledger.resolve_auction(aid)
                    except Exception:  # pragma: no cover - optional
                        logger.debug("Ledger auction failed", exc_info=True)
                    current_agent_state.du -= config.MAP_MOVE_DU_COST
                start_du = current_agent_state.du
                current_agent_state.ip -= config.MAP_MOVE_IP_COST
                current_agent_state.ip += config.MAP_MOVE_IP_REWARD
                current_agent_state.du += config.MAP_MOVE_DU_REWARD
                try:
                    from src.infra.ledger import ledger

                    ledger.log_change(
                        agent_id,
                        current_agent_state.ip - start_ip,
                        current_agent_state.du - start_du,
                        "move",
                    )
                except Exception:  # pragma: no cover - optional
                    logger.debug("Ledger logging failed", exc_info=True)
            elif action_type == "gather":
                res = map_action.get("resource")
                success = False
                if isinstance(res, str):
                    success = self.world_map.gather(
                        agent_id,
                        ResourceToken(res),
                        vector=self.vector.to_dict(),
                    )
                action_details = {"resource": res, "success": success}
                if success:
                    start_ip = current_agent_state.ip
                    if config.MAP_GATHER_DU_COST > 0:
                        try:
                            from src.infra.ledger import ledger

                            aid = ledger.open_auction("gather")
                            ledger.place_bid(aid, agent_id, config.MAP_GATHER_DU_COST)
                            ledger.resolve_auction(aid)
                        except Exception:  # pragma: no cover - optional
                            logger.debug("Ledger auction failed", exc_info=True)
                        current_agent_state.du -= config.MAP_GATHER_DU_COST
                    start_du = current_agent_state.du
                    current_agent_state.ip -= config.MAP_GATHER_IP_COST
                    current_agent_state.ip += config.MAP_GATHER_IP_REWARD
                    current_agent_state.du += config.MAP_GATHER_DU_REWARD
                    try:
                        from src.infra.ledger import ledger

                        ledger.log_change(
                            agent_id,
                            current_agent_state.ip - start_ip,
                            current_agent_state.du - start_du,
                            "gather",
                        )
                    except Exception:  # pragma: no cover - optional
                        logger.debug("Ledger logging failed", exc_info=True)
            elif action_type == "build":
                struct = map_action.get("structure")
                success = False
                if isinstance(struct, str):
                    success = self.world_map.build(
                        agent_id,
                        StructureType(struct),
                        vector=self.vector.to_dict(),
                    )
                action_details = {"structure": struct, "success": success}
                if success:
                    start_ip = current_agent_state.ip
                    if config.MAP_BUILD_DU_COST > 0:
                        try:
                            from src.infra.ledger import ledger

                            aid = ledger.open_auction("build")
                            ledger.place_bid(aid, agent_id, config.MAP_BUILD_DU_COST)
                            ledger.resolve_auction(aid)
                        except Exception:  # pragma: no cover - optional
                            logger.debug("Ledger auction failed", exc_info=True)
                        current_agent_state.du -= config.MAP_BUILD_DU_COST
                    start_du = current_agent_state.du
                    current_agent_state.ip -= config.MAP_BUILD_IP_COST
                    current_agent_state.ip += config.MAP_BUILD_IP_REWARD
                    current_agent_state.du += config.MAP_BUILD_DU_REWARD
                    try:
                        from src.infra.ledger import ledger

                        ledger.log_change(
                            agent_id,
                            current_agent_state.ip - start_ip,
                            current_agent_state.du - start_du,
                            "build",
                        )
                    except Exception:  # pragma: no cover - optional
                        logger.debug("Ledger logging failed", exc_info=True)
            else:
                action_details = {}

            map_event = log_event(
                {
                    "type": "map_action",
                    "agent_id": agent_id,
                    "step": self.current_step,
                    "action": action_type,
                    **action_details,
                }
            )
            await emit_map_action_event(
                agent_id,
                self.current_step,
                action_type,
                **action_details,
            )
            if self.discord_bot:
                embed = self.discord_bot.create_map_action_embed(
                    agent_id=agent_id,
                    action=action_type,
                    details=action_details,
                    step=self.current_step,
                )
                _ = asyncio.create_task(
                    self.discord_bot.send_simulation_update(embed=embed, agent_id=agent_id)
                )

            self.agents[agent_index].update_state(current_agent_state)

        logger.debug(
            f"SIM_DEBUG: After Agent {agent_id}'s turn in Global Turn {self.current_step}: "
            f"pending_messages_for_next_round now has {len(self.pending_messages_for_next_round)} messages. "
            f"messages_to_perceive_this_round now has {len(self.messages_to_perceive_this_round)} messages."
        )
        if has_role_change:
            next_agent_index = agent_index

        self.current_agent_index = next_agent_index
        self.last_completed_agent_index = agent_index

        self._update_collective_metrics()

        logger.info(f"Agent {agent_id} completed Global Turn {self.current_step}:")
        logger.info(f"  - IP: {current_agent_state.ip:.1f} (from {ip_start})")
        logger.info(f"  - DU: {current_agent_state.du:.1f} (from {du_start})")

        event = log_event(
            {
                "type": "agent_action",
                "agent_id": agent_id,
                "step": self.current_step,
                "action_intent": action_intent_str,
                "ip": current_agent_state.ip,
                "du": current_agent_state.du,
            }
        )
        if event is None:
            event = {
                "type": "agent_action",
                "agent_id": agent_id,
                "step": self.current_step,
                "action_intent": action_intent_str,
                "ip": current_agent_state.ip,
                "du": current_agent_state.du,
            }
            event["trace_hash"] = compute_trace_hash(event)
        trace_hash = event["trace_hash"]
        await emit_event(SimulationEvent(event_type="agent_action", data=event))

        if self.current_step % int(config.SNAPSHOT_INTERVAL_STEPS) == 0:
            snapshot = {
                "step": self.current_step,
                "collective_ip": self.collective_ip,
                "collective_du": self.collective_du,
                "knowledge_board": self.knowledge_board.to_dict(),
                "world_map": self.world_map.to_dict(),
                "agents": [
                    {
                        "agent_id": ag.agent_id,
                        "ip": ag.state.ip,
                        "du": ag.state.du,
                        "mood": ag.state.mood_level,
                    }
                    for ag in self.agents
                ],
                "trace_hash": self._last_trace_hash,
            }
            snapshot_no_vector = {
                **{k: v for k, v in snapshot.items() if k != "trace_hash"},
                "knowledge_board": {
                    k: v for k, v in snapshot["knowledge_board"].items() if k != "vector"
                },
                "world_map": {k: v for k, v in snapshot["world_map"].items() if k != "vector"},
            }
            snapshot["trace_hash"] = compute_trace_hash(snapshot_no_vector)
            self._last_trace_hash = snapshot["trace_hash"]
            save_snapshot(self.current_step, snapshot)
            snapshot_event = log_event({"type": "snapshot", **snapshot})
            await emit_event(SimulationEvent(event_type="snapshot", data=snapshot_event))

        # Advance to the next agent for the next turn
        self.current_agent_index = next_agent_index
        self.total_turns_executed += 1
        turn_counter_this_run_step += 1

        if (
            self.vector_store_manager
            and self.current_step - self._last_memory_prune_step
            >= config.MEMORY_STORE_PRUNE_INTERVAL_STEPS
        ):
            try:
                self.vector_store_manager.prune(int(config.MEMORY_STORE_TTL_SECONDS))
            except (ChromaDBException, ValidationError, OSError) as exc:
                logger.error("Failed to prune memory store: %s", exc)
            self._last_memory_prune_step = self.current_step

        if (
            self.vector_store_manager
            and self.current_step % len(self.agents) == 0
            and self.current_step > self._last_consolidation_step
        ):
            start_step = self.current_step - len(self.agents) + 1
            for ag in self.agents:
                try:
                    await self.vector_store_manager.aconsolidate_daily_memories(
                        ag.agent_id,
                        start_step,
                        self.current_step,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Failed nightly consolidation: %s", exc)
        self._last_consolidation_step = self.current_step

        self.vector.increment(self.agents[next_agent_index].get_id())
        await self.event_kernel.schedule_at(
            self.current_step,
            self._create_agent_event(next_agent_index),
            agent_id=self.agents[next_agent_index].get_id(),
            vector=self.vector,
        )

    def _create_agent_event(self: Self, agent_index: int) -> Callable[[], Awaitable[None]]:
        return lambda: self._run_agent_turn(agent_index)

    async def run_step(self: Self, max_turns: int = 1) -> int:
        """Dispatch up to ``max_turns`` events via the kernel."""
        if not self.agents:
            logger.warning("No agents in simulation to run.")
            return 0

        if self.event_kernel.empty():
            self.vector.increment(self.agents[self.current_agent_index].get_id())
            self.event_kernel.schedule_at_nowait(
                self.current_step,
                self._create_agent_event(self.current_agent_index),
                agent_id=self.agents[self.current_agent_index].get_id(),
                vector=self.vector,
            )

        events = await self.event_kernel.dispatch(max_turns)
        self.vector = self.event_kernel.vector
        return len(events)

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
        try:
            for step in range(num_steps):
                await asyncio.sleep(0.1)  # Optional: Add a small delay between steps
                steps = await self.run_step(
                    1
                )  # If run_step needs to be async, refactor it as well
                total_steps_executed += steps
        finally:
            elapsed_time = time.time() - start_time
            logger.info(
                "Simulation completed "
                f"{total_steps_executed} steps in {elapsed_time:.2f} seconds (async)"
            )
            self.close()

    def apply_event(self: Self, event: dict[str, Any]) -> None:
        """Apply an event from the Redpanda log to the simulation."""
        expected_hash = event.get("trace_hash")
        if expected_hash is not None:
            actual_hash = compute_trace_hash({k: v for k, v in event.items() if k != "trace_hash"})
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Trace hash mismatch for event at step {event.get('step')}:"
                    f" expected {expected_hash}, computed {actual_hash}"
                )
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
        elif event.get("type") == "snapshot":
            step = event.get("step")
            if isinstance(step, int):
                snapshot = load_snapshot(step)
                if snapshot.get("trace_hash") != expected_hash:
                    raise ValueError(
                        f"Snapshot hash mismatch at step {step}:"
                        f" event {expected_hash} != file {snapshot.get('trace_hash')}"
                    )
                self._last_trace_hash = snapshot.get("trace_hash", "")

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

    def close(self: Self) -> None:
        """Release resources held by the simulation."""
        if hasattr(self.knowledge_board, "close"):
            try:
                self.knowledge_board.close()  # type: ignore[attr-defined]
            except (OSError, RuntimeError) as exc:  # pragma: no cover - defensive
                logger.exception("Failed to close knowledge board: %s", exc)
        if self.vector_store_manager and hasattr(self.vector_store_manager, "close"):
            try:
                self.vector_store_manager.close()  # type: ignore[attr-defined]
            except (OSError, RuntimeError) as exc:  # pragma: no cover - defensive
                logger.exception("Failed to close vector store manager: %s", exc)
        try:
            event_queue.put_nowait(None)
        except asyncio.QueueFull:  # pragma: no cover - defensive
            logger.exception("Failed to enqueue shutdown event")

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
            self.knowledge_board.add_entry(
                project_info,
                creator_agent_id,
                self.current_step,
                self.vector.to_dict(),
            )

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
            self.knowledge_board.add_entry(
                join_info,
                agent_id,
                self.current_step,
                self.vector.to_dict(),
            )

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
            self.knowledge_board.add_entry(
                leave_info,
                agent_id,
                self.current_step,
                self.vector.to_dict(),
            )

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

    async def propose_law(self: Self, proposer_id: str, text: str) -> bool:
        """Allow an agent to propose a law and trigger a vote."""
        from src.governance import propose_law as _propose

        proposer = next((a for a in self.agents if a.agent_id == proposer_id), None)
        if proposer is None:
            return False

        if self.knowledge_board:
            self.knowledge_board.add_law_proposal(
                text,
                proposer_id,
                self.current_step,
                self.vector.to_dict(),
            )

        approved = await _propose(proposer, text, self.agents)
        if approved and self.knowledge_board:
            self.knowledge_board.add_entry(
                f"Law approved: {text}",
                proposer_id,
                self.current_step,
                self.vector.to_dict(),
            )

        return approved

    async def forward_proposal(self: Self, proposer_id: str, text: str) -> bool:
        """Forward a proposal to :func:`propose_law`."""
        return await self.propose_law(proposer_id, text)

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
