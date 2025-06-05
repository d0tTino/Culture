#!/usr/bin/env python

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the Python path so `src` imports work when running
# the script directly via `python examples/walking_vertical_slice.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app import create_simulation


def main() -> None:
    """Run a minimal Culture.ai simulation using a real LLM."""
    logging.basicConfig(level=logging.INFO)

    sim = create_simulation(
        num_agents=2,
        steps=3,
        scenario="Vertical slice demonstration",
        use_vector_store=False,
    )
    asyncio.run(sim.async_run(sim.steps_to_run))


if __name__ == "__main__":
    main()
