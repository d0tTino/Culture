# src/app.py
"""Entry point for running a simple multi-agent simulation."""

import argparse
import asyncio
import logging
import sys
from typing import Optional

from src.agents.core.base_agent import Agent
from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.infra.config import get_config
from src.infra.llm_client import get_ollama_client
from src.sim.knowledge_board import KnowledgeBoard
from src.sim.simulation import Simulation

# Try to import discord bot if present and enabled
try:
    from src.interfaces.discord_bot import SimulationDiscordBot

    simulation_discord_bot_class: Optional[type[SimulationDiscordBot]] = SimulationDiscordBot
except ImportError:
    logging.warning("Discord bot module not found, will run without Discord integration.")
    simulation_discord_bot_class = None


# Test scenario for relationship verification
VERIFICATION_SCENARIO = (
    "The team is collaboratively designing a specification for a communication protocol. "
    "Each agent should contribute ideas and feedback while being aware of their relationships "
    "with others."
)

# --- Dark Forest Hypothesis Scenario ---
DARK_FOREST_SCENARIO = (
    "In a vast galaxy filled with unknown civilizations, each agent must decide whether to "
    "broadcast their existence, remain hidden, or preemptively attack others. "
    "Revealing oneself may attract allies or deadly enemies. Hiding may ensure survival but "
    "limit opportunities. Agents have incomplete information about others' intentions and must "
    "weigh the risks of communication, cooperation, and aggression. The simulation explores the "
    "consequences of the 'dark forest' hypothesis: in a universe where any contact could be "
    "fatal, what strategies emerge?"
)


def create_base_simulation(
    scenario: str = VERIFICATION_SCENARIO,
    num_agents: int = 3,
    steps: int = 10,
    use_discord: bool = False,
    use_vector_store: bool = False,
    vector_store_dir: str = "./chroma_db",
) -> Simulation:
    """
    Creates a baseline simulation with the specified number of agents.

    Args:
        scenario: The scenario description
        num_agents: Number of agents in the simulation
        steps: Number of steps to run
        use_discord: Whether to use Discord for output
        use_vector_store: Whether to use vector store for memory
        vector_store_dir: Directory path for ChromaDB persistence (default: ./chroma_db)

    Returns:
        A configured Simulation instance
    """
    # Check Ollama availability
    ollama_client = get_ollama_client()
    if not ollama_client:
        logging.error("Failed to connect to Ollama. Please ensure Ollama is running.")
        sys.exit(1)

    # Initialize Discord bot if requested
    discord_bot = None
    if use_discord and simulation_discord_bot_class:
        bot_token = str(get_config("DISCORD_BOT_TOKEN"))
        channel_id = get_config("DISCORD_CHANNEL_ID")
        if not bot_token or not channel_id:
            logging.warning(
                "Discord bot token or channel ID not configured; running without Discord integration."
            )
        else:
            discord_bot = simulation_discord_bot_class(bot_token, int(channel_id))
            if not discord_bot.is_ready:
                logging.warning("Discord bot not ready, will run without Discord integration.")
                discord_bot = None

    # Create the simulation
    kb = KnowledgeBoard()

    # Create agents first
    agents = []
    for i in range(1, num_agents + 1):
        agent_id = f"agent_{i}"
        agent_name = f"Agent_{i}"
        agent = Agent(agent_id=agent_id, name=agent_name)
        agents.append(agent)

    # Create the simulation with the agents
    sim = Simulation(
        agents=agents,
        vector_store_manager=(
            None
            if not use_vector_store
            else ChromaVectorStoreManager(persist_directory=vector_store_dir)
        ),
        scenario=scenario,
        discord_bot=discord_bot,
    )

    # Set the knowledge board
    sim.knowledge_board = kb

    # Set the number of steps
    sim.steps_to_run = steps

    return sim


async def main() -> None:
    """Run a basic multi-agent simulation."""
    from src.infra.dspy_ollama_integration import configure_dspy_with_ollama
    from src.infra.logging_config import setup_logging

    parser = argparse.ArgumentParser(description="Run the Culture.ai simulation")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--discord", action="store_true", help="Enable Discord integration")
    parser.add_argument(
        "--scenario",
        type=str,
        default=VERIFICATION_SCENARIO,
        help="Scenario description",
    )
    args = parser.parse_args()

    configure_dspy_with_ollama(model_name="mistral:latest", temperature=0.1)
    setup_logging(log_dir="logs")

    sim = create_base_simulation(
        scenario=args.scenario,
        num_agents=args.agents,
        steps=args.steps,
        use_discord=args.discord,
    )

    await sim.async_run(args.steps)


if __name__ == "__main__":
    asyncio.run(main())
