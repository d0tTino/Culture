from __future__ import annotations

# mypy: ignore-errors
import logging
import os
import pickle
import platform
import random
import sys
from pathlib import Path
from typing import Any, cast

from src.agents.core.base_agent import Agent

try:  # pragma: no cover - optional dependency
    from src.agents.memory.vector_store import ChromaVectorStoreManager
except Exception:  # pragma: no cover - fallback when chromadb missing
    ChromaVectorStoreManager = None  # type: ignore[misc, assignment]
from src.sim.knowledge_board import KnowledgeBoard
from src.sim.simulation import Simulation

logger = logging.getLogger(__name__)


def capture_rng_state() -> tuple[Any, ...]:
    """Return the current ``random`` module state."""
    return random.getstate()


def restore_rng_state(state: tuple[Any, ...]) -> None:
    """Restore the ``random`` module to ``state``."""
    random.setstate(state)


def capture_environment() -> dict[str, Any]:
    """Capture relevant environment variables and system info."""
    from src.infra.config import DEFAULT_CONFIG

    env = {key: os.environ.get(key) for key in DEFAULT_CONFIG}
    env["python_version"] = sys.version
    env["platform"] = platform.platform()
    return env


def restore_environment(env: dict[str, Any]) -> None:
    """Restore environment variables from ``env``."""
    from src.infra import config

    for key, value in env.items():
        if key in {"python_version", "platform"}:
            continue
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    config.load_config()


def _serialize_simulation(sim: Simulation) -> dict[str, Any]:
    """Convert a ``Simulation`` instance into a serializable dictionary."""
    return {
        "agents": [cast(Agent, agent).state.to_dict(exclude_none=True) for agent in sim.agents],

        "current_step": sim.current_step,
        "current_agent_index": sim.current_agent_index,
        "scenario": sim.scenario,
        "vector_store_dir": getattr(
            sim.vector_store_manager,
            "persist_directory",
            None,
        ),
        "rng_state": capture_rng_state(),
        "environment": capture_environment(),
    }


def save_checkpoint(sim: Simulation, path: str | Path) -> None:
    """Serialize ``sim`` to ``path`` using pickle."""
    data = _serialize_simulation(sim)
    p = Path(path)
    with p.open("wb") as fh:
        pickle.dump(data, fh)
    logger.info("Checkpoint saved to %s", p)


def load_checkpoint(path: str | Path) -> tuple[Simulation, dict[str, Any]]:
    """Restore a ``Simulation`` instance and metadata from ``path``."""
    p = Path(path)
    with p.open("rb") as fh:
        data = pickle.load(fh)

    vector_store_manager = None
    vector_store_dir = data.get("vector_store_dir")
    if vector_store_dir:
        if ChromaVectorStoreManager is None:
            raise ImportError("chromadb is required to load vector store from checkpoint")
        vector_store_manager = ChromaVectorStoreManager(persist_directory=vector_store_dir)

    agents = [
        Agent(
            agent_id=agent_data.get("agent_id"),
            initial_state=agent_data,
            name=agent_data.get("name"),
            vector_store_manager=vector_store_manager,
        )
        for agent_data in data.get("agents", [])
    ]

    sim = Simulation(
        agents=agents,
        vector_store_manager=vector_store_manager,
        scenario=data.get("scenario", ""),
    )
    sim.knowledge_board = KnowledgeBoard()
    sim.current_step = data.get("current_step", 0)
    sim.current_agent_index = data.get("current_agent_index", 0)

    meta = {
        "rng_state": data.get("rng_state"),
        "environment": data.get("environment"),
    }
    return sim, meta
