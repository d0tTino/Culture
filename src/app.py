import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from src.agents.core.base_agent import Agent
from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.infra.checkpoint import (
    load_checkpoint,
    restore_environment,
    restore_rng_state,
    save_checkpoint,
)
from src.infra.llm_client import get_ollama_client
from src.infra.logging_config import setup_logging
from src.infra.settings import settings
from src.infra.warning_filters import configure_warning_filters
from src.sim.knowledge_board import KnowledgeBoard
from src.sim.simulation import Simulation
from src.utils.loop_helper import use_uvloop_if_available


def _simple_yaml(path: Path) -> dict[str, object]:
    """Very small YAML parser for simple key/value files."""
    data: dict[str, object] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            value = value.strip().strip('"').strip("'")
            if value.isdigit():
                data[key.strip()] = int(value)
            else:
                data[key.strip()] = value
    return data


def load_scenario(value: str) -> tuple[str, int | None, int | None]:
    """Return scenario description and optional overrides from a file."""
    path = Path(value)
    if path.is_file():
        try:  # Try full YAML parsing if PyYAML is available
            import yaml  # type: ignore

            content = yaml.safe_load(path.read_text())
        except Exception:  # pragma: no cover - fallback
            content = _simple_yaml(path)

        if isinstance(content, dict):
            desc = str(content.get("description") or content.get("scenario") or "")
            steps = content.get("steps")
            agents = content.get("agents")
            return (
                desc,
                int(steps) if steps is not None else None,
                int(agents) if agents is not None else None,
            )
        if isinstance(content, str):
            return content, None, None
        return str(content), None, None
    return value, None, None


use_uvloop_if_available()

try:
    from src.interfaces.discord_bot import SimulationDiscordBot

    simulation_discord_bot_class: Optional[type[SimulationDiscordBot]] = SimulationDiscordBot
except ImportError:  # pragma: no cover - optional dependency
    logging.warning("Discord bot module not found, running without Discord integration.")
    simulation_discord_bot_class = None

DEFAULT_SCENARIO = "Agents collaborate to design a specification for a communication protocol."


def create_simulation(
    num_agents: int = 3,
    steps: int = 10,
    scenario: str = DEFAULT_SCENARIO,
    use_discord: bool = False,
    use_vector_store: bool = False,
    vector_store_dir: str = "./chroma_db",
) -> Simulation:
    """Construct a Simulation instance with basic defaults."""

    ollama_client = get_ollama_client()
    if not ollama_client:
        logging.error("Failed to connect to Ollama. Please ensure Ollama is running.")
        sys.exit(1)

    discord_bot = None
    if use_discord and simulation_discord_bot_class:
        bot_token_raw = str(settings.DISCORD_BOT_TOKEN)
        channel_id = settings.DISCORD_CHANNEL_ID
        if bot_token_raw and channel_id:
            tokens = [tok.strip() for tok in bot_token_raw.split(",") if tok.strip()]
            bot = simulation_discord_bot_class(
                tokens if len(tokens) > 1 else tokens[0], int(channel_id)
            )
            if bot.is_ready:
                discord_bot = bot
            else:
                logging.warning("Discord bot not ready, running without integration.")

    agents = [Agent(agent_id=f"agent_{i + 1}", name=f"Agent_{i + 1}") for i in range(num_agents)]

    sim = Simulation(
        agents=agents,
        vector_store_manager=(
            None
            if not use_vector_store
            else (
                ChromaVectorStoreManager(persist_directory=vector_store_dir)
                if ChromaVectorStoreManager is not None
                else None
            )
        ),
        scenario=scenario,
        discord_bot=discord_bot,
    )
    sim.knowledge_board = KnowledgeBoard()
    sim.steps_to_run = steps
    return sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Culture.ai simulation.")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show the Culture.ai version and exit.",
    )
    parser.add_argument("--agents", type=int, default=3, help="Number of agents.")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run.")
    parser.add_argument(
        "--scenario", type=str, default=DEFAULT_SCENARIO, help="Simulation scenario."
    )
    parser.add_argument("--discord", action="store_true", help="Enable Discord integration.")
    parser.add_argument(
        "--vector-store", action="store_true", help="Use ChromaDB for agent memory."
    )
    parser.add_argument(
        "--vector-dir",
        type=str,
        default="./chroma_db",
        help="Directory for vector store persistence.",
    )
    parser.add_argument(
        "--no-warning-filters",
        action="store_true",
        help="Disable default warning filters.",
    )
    parser.add_argument(
        "--log-suppressed-warnings",
        action="store_true",
        help="Log warnings that would otherwise be suppressed.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a checkpoint file to load and save simulation state.",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help=(
            "Restore RNG and environment state from the checkpoint to reproduce agent decisions"
        ),
    )
    parser.add_argument(
        "--proposal",
        type=str,
        help="Submit a law proposal before running the simulation.",
    )
    parser.add_argument(
        "--proposer-id",
        type=str,
        default="agent_1",
        help="Agent ID submitting the proposal.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    desc, file_steps, file_agents = load_scenario(args.scenario)
    if file_steps is not None:
        args.steps = file_steps
    if file_agents is not None:
        args.agents = file_agents
    args.scenario = desc

    if args.version:
        from src import __version__

        print(__version__)
        return

    configure_warning_filters(
        apply_filters=not args.no_warning_filters,
        log_suppressed=args.log_suppressed_warnings,
    )

    sim: Simulation
    meta: dict[str, object] | None = None
    if args.checkpoint and Path(args.checkpoint).exists():
        logging.info("Loading simulation from checkpoint %s", args.checkpoint)
        sim, meta = load_checkpoint(args.checkpoint, replay=args.replay)
        sim.steps_to_run = args.steps
    else:
        sim = create_simulation(
            num_agents=args.agents,
            steps=args.steps,
            scenario=args.scenario,
            use_discord=args.discord,
            use_vector_store=args.vector_store,
            vector_store_dir=args.vector_dir,
        )

    if args.replay and meta:
        if meta.get("rng_state") is not None:
            restore_rng_state(meta["rng_state"])
        if meta.get("environment") is not None:
            restore_environment(meta["environment"])

    if args.proposal:
        asyncio.run(sim.forward_proposal(args.proposer_id, args.proposal))

    asyncio.run(sim.async_run(args.steps))

    if args.checkpoint:
        save_checkpoint(sim, args.checkpoint)


if __name__ == "__main__":
    main()
