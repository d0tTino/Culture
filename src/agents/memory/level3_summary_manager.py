from __future__ import annotations

from datetime import datetime

from typing_extensions import Self

from .vector_store import ChromaVectorStoreManager


class Level3SummaryManager:
    """Consolidate multiple Level 2 summaries into a long term summary."""

    def __init__(self: Self, vector_store: ChromaVectorStoreManager) -> None:
        self.vector_store = vector_store

    def consolidate_summaries(self: Self, agent_id: str, start_step: int, end_step: int) -> str:
        """Create an arc summary from Level 2 summaries in the given step range."""
        l2_summaries = self.vector_store.retrieve_filtered_memories(
            agent_id, filters={"memory_type": "chapter_summary"}, limit=None
        )
        selected = [s for s in l2_summaries if start_step <= int(s.get("step", 0)) <= end_step]
        if not selected:
            return ""
        summary = "\n".join(s.get("content", "") for s in selected)
        self.vector_store.add_memory(
            agent_id,
            end_step,
            "arc_summary",
            summary,
            memory_type="arc_summary",
            metadata={
                "consolidated_step_range": f"{start_step}-{end_step}",
                "simulation_step_end_timestamp": datetime.utcnow().isoformat(),
            },
        )
        return summary

    def get_summaries(self: Self, agent_id: str, limit: int = 5) -> list[str]:
        results = self.vector_store.retrieve_filtered_memories(
            agent_id, filters={"memory_type": "arc_summary"}, limit=limit
        )
        return [r.get("content", "") for r in results]
