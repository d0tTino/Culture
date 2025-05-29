#!/usr/bin/env python
"""
Integration test for hierarchical memory persistence.

This test verifies that the hierarchical memory system (L1 and L2 summaries)
correctly persists memories across simulation runs.
"""

import json
import logging
import time
import unittest
from datetime import datetime

import pytest
from typing_extensions import Self

from src.agents.memory.vector_store import ChromaVectorStoreManager


@pytest.mark.integration
@pytest.mark.memory
@pytest.mark.usefixtures("chroma_test_dir")
class TestHierarchicalMemoryPersistence(unittest.TestCase):
    """Tests for hierarchical memory persistence in the agent memory system."""

    @pytest.fixture(autouse=True)
    def _inject_fixtures(self: Self, request: object, chroma_test_dir: str) -> None:
        self.request = request
        self.chroma_test_dir = chroma_test_dir

    def setUp(self: Self) -> None:
        self.vector_store = ChromaVectorStoreManager(persist_directory=self.chroma_test_dir)

    def tearDown(self: Self) -> None:
        try:
            if hasattr(self, "vector_store") and self.vector_store:
                if hasattr(self.vector_store, "client") and self.vector_store.client:
                    pass
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Error during teardown: {e}")

    def test_hierarchical_memory_persistence(self: Self) -> None:
        """Test that hierarchical memories (L1 and L2) persist correctly."""
        # Create an agent ID for testing
        agent_id = "test_hierarchical_agent"

        # Add some L1 summaries
        l1_ids = []
        for i in range(5):
            l1_id = self.vector_store.add_memory(
                agent_id=agent_id,
                step=i * 10,  # Spaced steps
                event_type="consolidation",
                content=f"L1 Summary {i}: This is a test consolidated memory",
                memory_type="consolidated_summary",
                metadata={
                    "simulation_step_timestamp": datetime.now().isoformat(),
                    "consolidation_window_start": i * 10,
                    "consolidation_window_end": (i * 10) + 9,
                },
            )
            l1_ids.append(l1_id)

        # Add an L2 summary that references the L1 summaries
        # Store the L1 summary IDs as a JSON string
        # since ChromaDB doesn't support lists in metadata
        l2_id = self.vector_store.add_memory(
            agent_id=agent_id,
            step=50,  # After the L1 summaries
            event_type="chapter_consolidation",
            content=(
                "L2 Chapter Summary: This is a test chapter summary based on the previous "
                "L1 summaries"
            ),
            memory_type="chapter_summary",
            metadata={
                "simulation_step_start_timestamp": datetime.now().isoformat(),
                "simulation_step_end_timestamp": datetime.now().isoformat(),
                "l1_summary_ids_json": json.dumps(l1_ids),  # Convert list to JSON string
                "chapter_start_step": 0,
                "chapter_end_step": 49,
            },
        )

        # Verify retrieval of L1 summaries
        l1_memories = self.vector_store.retrieve_filtered_memories(
            agent_id=agent_id, filters={"memory_type": "consolidated_summary"}, limit=10
        )

        self.assertEqual(len(l1_memories), 5, "Should retrieve all 5 L1 summaries")

        # Verify retrieval of L2 summary
        l2_memories = self.vector_store.retrieve_filtered_memories(
            agent_id=agent_id, filters={"memory_type": "chapter_summary"}, limit=10
        )

        self.assertEqual(len(l2_memories), 1, "Should retrieve the L2 summary")

        # Verify L2 summary contains reference to L1 summaries
        l2_metadata = next(
            (m for m in l2_memories if "memory_id" in m and m["memory_id"] == l2_id), None
        )
        self.assertIsNotNone(l2_metadata, "Should find the L2 summary metadata")

        if l2_metadata:
            self.assertIn(
                "l1_summary_ids_json",
                l2_metadata,
                "L2 summary should reference L1 summary IDs as JSON",
            )
            # Parse the JSON string back to a list and verify
            stored_l1_ids = json.loads(l2_metadata["l1_summary_ids_json"])
            self.assertEqual(len(stored_l1_ids), 5, "L2 should reference all 5 L1 summaries")
            # Verify the IDs match
            for l1_id in l1_ids:
                self.assertIn(
                    l1_id, stored_l1_ids, f"L1 ID {l1_id} should be in the stored L1 IDs"
                )


if __name__ == "__main__":
    unittest.main()
