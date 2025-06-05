#!/usr/bin/env python
"""Simple analysis of memory usage statistics."""

import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path for direct script execution
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agents.memory.memory_tracking_manager import MemoryTrackingManager
from src.agents.memory.vector_store import ChromaVectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("memory_usage_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze memory usage statistics")
    parser.add_argument("--agent_id", required=True, help="Agent ID to analyze")
    parser.add_argument("--chroma_dir", default="./chroma_db", help="ChromaDB directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vector_store = ChromaVectorStoreManager(persist_directory=args.chroma_dir)
    tracker = MemoryTrackingManager(vector_store)

    results = vector_store.retrieve_filtered_memories(
        agent_id=args.agent_id,
        include_usage_stats=True,
    )

    if not results:
        logger.info("No memories found for agent %s", args.agent_id)
        return

    results.sort(key=lambda m: m.get("retrieval_count", 0), reverse=True)

    for mem in results[:10]:
        mus = tracker.calculate_mus(str(mem.get("memory_id", "")))
        logger.info(
            "MUS=%0.3f | retrievals=%s | content=%s",
            mus,
            mem.get("retrieval_count", 0),
            str(mem.get("content", ""))[:60],
        )


if __name__ == "__main__":
    main()
