#!/usr/bin/env python
"""Unit tests for MemoryTrackingManager."""

import unittest

import pytest
from typing_extensions import Self

pytest.importorskip("chromadb")

from src.agents.memory.memory_tracking_manager import MemoryTrackingManager
from src.shared.memory_store import ChromaMemoryStore


@pytest.mark.unit
@pytest.mark.memory
@pytest.mark.usefixtures("chroma_test_dir")
class TestMemoryTrackingManager(unittest.TestCase):
    """Tests for MemoryTrackingManager."""

    @pytest.fixture(autouse=True)
    def _inject_fixtures(self: Self, chroma_test_dir: str) -> None:
        self.chroma_test_dir = chroma_test_dir

    def setUp(self: Self) -> None:
        self.vector_store = ChromaMemoryStore(persist_directory=self.chroma_test_dir)
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


if __name__ == "__main__":
    unittest.main()
