#!/usr/bin/env python
"""Unit tests for MemoryTrackingManager."""

import asyncio
import math
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
from typing_extensions import Self

from src.agents.memory.memory_tracking_manager import MemoryTrackingManager
from src.agents.memory.vector_store import ChromaVectorStoreManager

pytest.importorskip("chromadb")


@pytest.mark.unit
@pytest.mark.memory
@pytest.mark.usefixtures("chroma_test_dir")
class TestMemoryTrackingManager(unittest.TestCase):
    """Tests for MemoryTrackingManager."""

    @pytest.fixture(autouse=True)
    def _inject_fixtures(self: Self, chroma_test_dir: Path) -> None:
        self.chroma_test_dir = chroma_test_dir

    def setUp(self: Self) -> None:
        self.vector_store = ChromaVectorStoreManager(
            persist_directory=self.chroma_test_dir,
            embedding_function=lambda texts: [[float(len(t))] for t in texts],
        )
        self.manager = MemoryTrackingManager(self.vector_store)
        self.agent_id = "tracking_test_agent"

    def tearDown(self: Self) -> None:
        if hasattr(self.vector_store, "client") and self.vector_store.client:
            close = getattr(self.vector_store.client, "close", None)
            if callable(close):
                close()

    def test_calculate_mus(self: Self) -> None:
        """Calculate MUS for a retrieved memory."""
        mem_id = self.vector_store.add_memory(
            agent_id=self.agent_id,
            step=1,
            event_type="thought",
            content="Unit test memory",
        )
        self.manager.record_retrieval([mem_id], [0.9])
        mus = self.manager.calculate_mus(mem_id)
        self.assertGreaterEqual(mus, 0.0)

    def test_unified_retrieval_usage_updates(self: Self) -> None:
        """Ensure usage stats are updated for vector and semantic retrieval."""
        from types import SimpleNamespace

        from src.agents.graphs.graph_nodes import retrieve_and_summarize_memories_node
        from src.agents.memory.semantic_memory_manager import SemanticMemoryManager

        semantic_manager = SemanticMemoryManager(self.vector_store, driver=None)
        for i in range(2):
            self.vector_store.add_memory(
                agent_id=self.agent_id,
                step=i,
                event_type="thought",
                content=f"m{i}",
            )

        semantic_manager.group_memories_by_topic(self.agent_id, num_topics=1)

        calls: list[list[str]] = []

        from unittest.mock import patch

        patcher = patch.object(
            self.vector_store.tracking_manager,
            "update_usage_stats",
            lambda ids, relevance_scores=None, increment_count=True: calls.append(list(ids)),
        )

        class DummyAgent:
            async def async_generate_l1_summary(
                self, role_prompt: str, memories: str, context: str
            ) -> SimpleNamespace:
                return SimpleNamespace(summary="S")

        from src.agents.memory.memory_service import MemoryService

        state = {
            "agent_id": self.agent_id,
            "memory_service": MemoryService(self.vector_store, semantic_manager),
            "vector_store_manager": self.vector_store,
            "semantic_manager": semantic_manager,
            "agent_instance": DummyAgent(),
            "state": SimpleNamespace(role_prompt="r"),
        }

        with patcher:
            asyncio.run(retrieve_and_summarize_memories_node(state))

        assert len(calls) == 1
        assert all(calls)

    def test_mus_varies_with_retrievals_and_scores(self: Self) -> None:
        """MUS should reflect retrieval count and relevance scores."""
        mem_id = self.vector_store.add_memory(
            agent_id=self.agent_id,
            step=2,
            event_type="thought",
            content="Another memory",
        )
        self.manager.record_retrieval([mem_id], [0.5])
        self.manager.record_retrieval([mem_id], [0.8])
        metadata = self.vector_store.collection.get(ids=[mem_id], include=["metadatas"])
        meta = metadata["metadatas"][0]
        expected = (
            0.4 * math.log(1 + meta["retrieval_count"])
            + 0.4 * (meta["accumulated_relevance_score"] / meta["retrieval_relevance_count"])
            + 0.2 * 1.0
        )
        mus = self.manager.calculate_mus(mem_id)
        self.assertAlmostEqual(mus, expected, places=5)

    def test_calculate_mus_from_metadata_dict(self: Self) -> None:
        """Direct metadata input should return expected MUS."""
        now = datetime.utcnow()
        metadata = {
            "retrieval_count": 4,
            "accumulated_relevance_score": 2.0,
            "retrieval_relevance_count": 2,
            "last_retrieved_timestamp": (now - timedelta(days=2)).isoformat(),
        }
        expected = (
            0.4 * math.log(1 + metadata["retrieval_count"])
            + 0.4
            * (metadata["accumulated_relevance_score"] / metadata["retrieval_relevance_count"])
            + 0.2 * (1.0 / (1.0 + 2))
        )
        mus = self.manager.calculate_mus(metadata)
        self.assertAlmostEqual(mus, expected, places=5)


if __name__ == "__main__":
    unittest.main()
