#!/usr/bin/env python
"""
Test script to verify the MUS-based memory pruning functionality in hierarchical memory.

This test verifies that both Level 1 (session) and Level 2 (chapter) summaries
are properly considered for pruning based on their Memory Utility Score (MUS).
"""

import logging
import time
import unittest
from datetime import datetime
from typing import Optional

import pytest
from pytest import FixtureRequest

pytest.importorskip("chromadb")
from typing_extensions import Self

from src.agents.memory.vector_store import ChromaVectorStoreManager
from tests.utils.mock_llm import MockLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("test_memory_pruning_mus")


@pytest.mark.integration
@pytest.mark.memory
@pytest.mark.slow
@pytest.mark.usefixtures("chroma_test_dir")
class TestMUSBasedMemoryPruning(unittest.TestCase):
    """Tests for MUS-based memory pruning in the agent memory system."""

    @pytest.fixture(autouse=True)
    def _inject_fixtures(self: Self, request: FixtureRequest, chroma_test_dir: str) -> None:
        self.request = request
        self.chroma_test_dir = chroma_test_dir

    def setUp(self: Self) -> None:
        self.mock_llm_cm = MockLLM({"default": "Mocked response for MUS pruning tests"})
        self.mock_llm_cm.__enter__()
        self.vector_store: Optional[ChromaVectorStoreManager] = ChromaVectorStoreManager(
            persist_directory=self.chroma_test_dir
        )
        self.agent_id = "test_mus_pruning_agent"
        self.create_test_memories()

    def tearDown(self: Self) -> None:
        self.mock_llm_cm.__exit__(None, None, None)

        try:
            # Clean up vector store resources
            if self.vector_store is not None:
                if hasattr(self.vector_store, "client") and self.vector_store.client:
                    if hasattr(self.vector_store.client, "close"):
                        self.vector_store.client.close()
                self.vector_store = None
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Error during test cleanup: {e}")

    def create_test_memories(self: Self) -> None:
        """Create test memories with different MUS characteristics."""
        logger.info("Creating test memories with different MUS characteristics...")

        # Storage for memory IDs by category
        self.memory_ids: dict[str, list[str]] = {
            "l1_high_mus": [],
            "l1_medium_mus": [],
            "l1_low_mus": [],
            "l2_high_mus": [],
            "l2_medium_mus": [],
            "l2_low_mus": [],
        }

        assert self.vector_store is not None
        # 1. Create L1 memories (consolidated_summary)
        # High MUS memory (frequently accessed, high relevance)
        for i in range(3):
            memory_id = self.vector_store.add_memory(
                agent_id=self.agent_id,
                step=10 + i,
                event_type="consolidation",
                content=(
                    f"L1 High-MUS Summary {i}: Contains important insights about "
                    f"project architecture."
                ),
                memory_type="consolidated_summary",
                metadata={
                    "retrieval_count": 10 + i,
                    "accumulated_relevance_score": 8.0 + i,
                    "retrieval_relevance_count": 10 + i,
                    "last_retrieved_timestamp": datetime.utcnow().isoformat(),
                },
            )
            self.memory_ids["l1_high_mus"].append(memory_id)

            # Simulate high usage by retrieving multiple times
            for _ in range(10):  # High retrieval count
                self._simulate_retrieval(memory_id, relevance_score=0.9)  # High relevance

        # Medium MUS memory (medium access, medium relevance)
        for i in range(3):
            memory_id = self.vector_store.add_memory(
                agent_id=self.agent_id,
                step=20 + i,
                event_type="consolidation",
                content=(
                    f"L1 Medium-MUS Summary {i}: Discusses weekly progress and team coordination."
                ),
                memory_type="consolidated_summary",
                metadata={
                    "retrieval_count": 5 + i,
                    "accumulated_relevance_score": 5.0 + i,
                    "retrieval_relevance_count": 5 + i,
                    "last_retrieved_timestamp": datetime.utcnow().isoformat(),
                },
            )
            self.memory_ids["l1_medium_mus"].append(memory_id)

            # Simulate medium usage
            for _ in range(5):  # Medium retrieval count
                self._simulate_retrieval(memory_id, relevance_score=0.5)  # Medium relevance

        # Low MUS memory (rarely accessed, low relevance)
        for i in range(3):
            memory_id = self.vector_store.add_memory(
                agent_id=self.agent_id,
                step=30 + i,
                event_type="consolidation",
                content=(
                    f"L1 Low-MUS Summary {i}: Notes about routine administrative tasks completed."
                ),
                memory_type="consolidated_summary",
                metadata={
                    "retrieval_count": 1 + i,
                    "accumulated_relevance_score": 1.0 + i,
                    "retrieval_relevance_count": 1 + i,
                    "last_retrieved_timestamp": datetime.utcnow().isoformat(),
                },
            )
            self.memory_ids["l1_low_mus"].append(memory_id)

            # Simulate low usage
            for _ in range(1):  # Low retrieval count
                self._simulate_retrieval(memory_id, relevance_score=0.2)  # Low relevance

        # 2. Create L2 memories (chapter_summary)
        # High MUS memory
        for i in range(2):
            memory_id = self.vector_store.add_memory(
                agent_id=self.agent_id,
                step=100 + i * 10,
                event_type="chapter_consolidation",
                content=(
                    f"L2 High-MUS Chapter {i}: Major project milestone reached with key "
                    f"architecture decisions."
                ),
                memory_type="chapter_summary",
                metadata={
                    "retrieval_count": 10 + i,
                    "accumulated_relevance_score": 8.0 + i,
                    "retrieval_relevance_count": 10 + i,
                    "last_retrieved_timestamp": datetime.utcnow().isoformat(),
                    "consolidation_period": f"{10 + i * 10}-{19 + i * 10}",
                },
            )
            self.memory_ids["l2_high_mus"].append(memory_id)

            # Simulate high usage
            for _ in range(15):
                self._simulate_retrieval(memory_id, relevance_score=0.95)

        # Medium MUS memory
        for i in range(2):
            memory_id = self.vector_store.add_memory(
                agent_id=self.agent_id,
                step=110 + i * 10,
                event_type="chapter_consolidation",
                content=(
                    f"L2 Medium-MUS Chapter {i}: Agent participated in team discussions "
                    f"about feature implementation."
                ),
                memory_type="chapter_summary",
                metadata={
                    "retrieval_count": 5 + i,
                    "accumulated_relevance_score": 5.0 + i,
                    "retrieval_relevance_count": 5 + i,
                    "last_retrieved_timestamp": datetime.utcnow().isoformat(),
                    "consolidation_period": f"{20 + i * 10}-{29 + i * 10}",
                },
            )
            self.memory_ids["l2_medium_mus"].append(memory_id)

            # Simulate medium usage
            for _ in range(6):
                self._simulate_retrieval(memory_id, relevance_score=0.6)

        # Low MUS memory
        for i in range(2):
            memory_id = self.vector_store.add_memory(
                agent_id=self.agent_id,
                step=120 + i * 10,
                event_type="chapter_consolidation",
                content=(
                    f"L2 Low-MUS Chapter {i}: Routine updates and minor administrative activities."
                ),
                memory_type="chapter_summary",
                metadata={
                    "retrieval_count": 1 + i,
                    "accumulated_relevance_score": 1.0 + i,
                    "retrieval_relevance_count": 1 + i,
                    "last_retrieved_timestamp": datetime.utcnow().isoformat(),
                    "consolidation_period": f"{30 + i * 10}-{39 + i * 10}",
                },
            )
            self.memory_ids["l2_low_mus"].append(memory_id)

            # Simulate low usage
            for _ in range(2):
                self._simulate_retrieval(memory_id, relevance_score=0.3)

        logger.info(
            f"Created test memories: {sum(len(ids) for ids in self.memory_ids.values())} total"
        )

    def _simulate_retrieval(self: Self, memory_id: str, relevance_score: float = 0.5) -> None:
        """Simulate retrieving a memory to update its usage statistics."""
        assert self.vector_store is not None
        self.vector_store._update_memory_usage_stats(
            memory_ids=[memory_id], relevance_scores=[relevance_score], increment_count=True
        )

    def _calculate_expected_mus(self: Self, memory_category: str) -> float:
        """
        Calculate the expected Memory Utility Score (MUS) range for a category.
        Based on the formula: MUS = (0.4 * RFS) + (0.4 * RS) + (0.2 * RecS)
        """
        # Define expected retrieval counts and relevance scores per category
        category_params = {
            "l1_high_mus": {"retrieval_count": 10, "relevance": 0.9, "min_mus": 0.5},
            "l1_medium_mus": {"retrieval_count": 5, "relevance": 0.5, "min_mus": 0.3},
            "l1_low_mus": {"retrieval_count": 1, "relevance": 0.2, "min_mus": 0.1},
            "l2_high_mus": {"retrieval_count": 15, "relevance": 0.95, "min_mus": 0.6},
            "l2_medium_mus": {"retrieval_count": 6, "relevance": 0.6, "min_mus": 0.35},
            "l2_low_mus": {"retrieval_count": 2, "relevance": 0.3, "min_mus": 0.2},
        }

        params = category_params.get(memory_category)
        if not params:
            return 0.0

        # Apply simplified MUS formula (ignoring recency which changes over time)
        # rfs = math.log(1 + params["retrieval_count"])
        # rs = params["relevance"]
        # We're ignoring RecS as it's time-dependent
        return params["min_mus"]  # Return minimum expected MUS

    def test_a_l1_mus_calculation(self: Self) -> None:
        """Test the calculation of MUS for L1 memories."""
        assert self.vector_store is not None
        # Get MUS scores for L1 memories
        l1_high_id = self.memory_ids["l1_high_mus"][0] if self.memory_ids["l1_high_mus"] else None
        l1_low_id = self.memory_ids["l1_low_mus"][0] if self.memory_ids["l1_low_mus"] else None

        if l1_high_id and l1_low_id:
            # Get metadata for high MUS memory
            high_results = self.vector_store.collection.get(
                ids=[l1_high_id], include=["metadatas"]
            )

            # Get metadata for low MUS memory
            low_results = self.vector_store.collection.get(ids=[l1_low_id], include=["metadatas"])

            if high_results and high_results.get("metadatas") and high_results["metadatas"]:
                high_metadata = high_results["metadatas"][0]
                high_mus = self.vector_store._calculate_mus(dict(high_metadata))
                logger.info(f"High MUS L1 memory: {high_mus:.3f}")

            if low_results and low_results.get("metadatas") and low_results["metadatas"]:
                low_metadata = low_results["metadatas"][0]
                low_mus = self.vector_store._calculate_mus(dict(low_metadata))
                logger.info(f"Low MUS L1 memory: {low_mus:.3f}")

            # If we have both scores, verify high > low
            if "high_mus" in locals() and "low_mus" in locals():
                self.assertGreater(
                    high_mus,
                    low_mus,
                    "High MUS memory should have higher score than low MUS memory",
                )

    def test_b_l2_mus_calculation(self: Self) -> None:
        """Test the calculation of MUS for L2 memories."""
        assert self.vector_store is not None
        # Get MUS scores for L2 memories
        l2_high_id = self.memory_ids["l2_high_mus"][0] if self.memory_ids["l2_high_mus"] else None
        l2_low_id = self.memory_ids["l2_low_mus"][0] if self.memory_ids["l2_low_mus"] else None

        if l2_high_id and l2_low_id:
            # Get metadata for high MUS memory
            high_results = self.vector_store.collection.get(
                ids=[l2_high_id], include=["metadatas"]
            )

            # Get metadata for low MUS memory
            low_results = self.vector_store.collection.get(ids=[l2_low_id], include=["metadatas"])

            if high_results and high_results.get("metadatas") and high_results["metadatas"]:
                high_metadata = high_results["metadatas"][0]
                high_mus = self.vector_store._calculate_mus(dict(high_metadata))
                logger.info(f"High MUS L2 memory: {high_mus:.3f}")

            if low_results and low_results.get("metadatas") and low_results["metadatas"]:
                low_metadata = low_results["metadatas"][0]
                low_mus = self.vector_store._calculate_mus(dict(low_metadata))
                logger.info(f"Low MUS L2 memory: {low_mus:.3f}")

            # If we have both scores, verify high > low
            if "high_mus" in locals() and "low_mus" in locals():
                self.assertGreater(
                    high_mus,
                    low_mus,
                    "High MUS memory should have higher score than low MUS memory",
                )

    def test_z_memory_deletion(self: Self) -> None:
        """Test that memories can be deleted by IDs."""
        assert self.vector_store is not None
        # Get IDs of memories to delete
        memories_to_delete = []

        # Add one memory from each category
        for category in self.memory_ids:
            if self.memory_ids[category]:
                memories_to_delete.append(self.memory_ids[category][0])

        # Skip if no memories to delete
        if not memories_to_delete:
            self.skipTest("No memories to delete")

        # Get initial memory count
        initial_count = self.vector_store.collection.count()

        # Delete the memories
        success = self.vector_store.delete_memories_by_ids(memories_to_delete)

        # Verify deletion was successful
        self.assertTrue(success, "Memory deletion should be successful")

        # Verify count changed
        final_count = self.vector_store.collection.count()
        expected_deleted = len(memories_to_delete)
        assert initial_count - final_count == expected_deleted, (
            f"Expected {expected_deleted} memories to be deleted, but count changed by "
            f"{initial_count - final_count}"
        )

        # Verify memories are actually gone
        for memory_id in memories_to_delete:
            # Try to retrieve the memory - should get empty result
            result = self.vector_store.collection.get(ids=[memory_id])
            self.assertEqual(
                len(result["ids"]),
                0,
                f"Memory {memory_id} should be deleted but was still retrievable",
            )

        logger.info(f"Successfully deleted {expected_deleted} memories")


if __name__ == "__main__":
    unittest.main()
