#!/usr/bin/env python
"""
Test script to demonstrate how to calculate the Memory Utility Score (MUS) for advanced pruning.
This implements the formula from the advanced_memory_pruning_design_proposal.md.
"""

import os
import uuid
import math
import logging
import unittest
import shutil
from datetime import datetime, timedelta
import pytest
import time
import tempfile

from src.agents.memory.vector_store import ChromaVectorStoreManager
from tests.utils.mock_llm import MockLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("test_memory_utility_score")

@pytest.mark.integration
@pytest.mark.memory
@pytest.mark.mus
@pytest.mark.slow
class TestMemoryUtilityScore(unittest.TestCase):
    """
    Test case demonstrating the calculation of Memory Utility Score (MUS)
    for advanced memory pruning as outlined in the design proposal.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by:
        1. Creating a test-specific ChromaDB directory
        2. Adding test memories to the vector store
        """
        logger.info("Setting up test environment for memory utility score tests")

        # Enable mock mode for LLM to avoid Ollama 404 errors
        cls.mock_context = MockLLM({
            "default": "Mocked response for memory utility score tests",
        })
        cls.mock_context.__enter__()
        
        # Set vector store directory - use test-specific location
        cls.vector_store_dir = f"./test_memory_utility_score_{uuid.uuid4().hex[:6]}"
        
        # Remove any previous test database
        if os.path.exists(cls.vector_store_dir):
            logger.info(f"Removing previous test ChromaDB at {cls.vector_store_dir}")
            shutil.rmtree(cls.vector_store_dir)
        
        # Create vector store manager
        cls.vector_store = ChromaVectorStoreManager(persist_directory=cls.vector_store_dir)
        
        # Define test agent ID
        cls.agent_id = "test_agent_1"
        
        # Add several test memories with different characteristics
        cls.memory_ids = []
        
        # Memory 1: High relevance memory (will be retrieved often)
        memory_id = cls.vector_store.add_memory(
            agent_id=cls.agent_id,
            step=1,
            event_type="thought",
            content="Important insight about the core problem we're solving with the transformer architecture."
        )
        cls.memory_ids.append(memory_id)
        
        # Memory 2: Medium relevance memory
        memory_id = cls.vector_store.add_memory(
            agent_id=cls.agent_id,
            step=2,
            event_type="broadcast_sent",
            content="I've been thinking about how we could optimize the encoder-decoder architecture."
        )
        cls.memory_ids.append(memory_id)
        
        # Memory 3: Low relevance memory (will rarely be retrieved)
        memory_id = cls.vector_store.add_memory(
            agent_id=cls.agent_id,
            step=3,
            event_type="broadcast_perceived",
            content="The weather is quite nice today, isn't it?"
        )
        cls.memory_ids.append(memory_id)
        
        logger.info(f"Added {len(cls.memory_ids)} test memories to vector store")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after the tests"""
        # Disable mock mode
        cls.mock_context.__exit__(None, None, None)
        
        # Explicitly close the ChromaDB client if possible
        try:
            if hasattr(cls, 'vector_store') and hasattr(cls.vector_store, 'client'):
                # Try to explicitly close any connections
                if hasattr(cls.vector_store.client, 'close'):
                    cls.vector_store.client.close()
                    cls.vector_store = None
            
            # Try to remove the directory
            if os.path.exists(cls.vector_store_dir):
                try:
                    shutil.rmtree(cls.vector_store_dir)
                    logger.info(f"Removed test ChromaDB directory: {cls.vector_store_dir}")
                except PermissionError as e:
                    # On Windows, sometimes ChromaDB keeps locks on files
                    logger.warning(f"Could not remove test directory due to permission error: {e}")
                    logger.warning("This is normal on Windows and won't affect test results")
        except Exception as e:
            logger.warning(f"Error during test cleanup: {e}")
    
    def test_memory_utility_score_calculation(self):
        """
        Test calculating the Memory Utility Score for memories.
        This demonstrates the implementation of the MUS formula from the design proposal:
        MUS = (0.4 * RFS) + (0.4 * RS) + (0.2 * RecS)
        """
        # Perform multiple retrievals to simulate different usage patterns
        
        # Memory 1: High relevance - retrieve multiple times with highly relevant queries
        high_relevance_queries = [
            "transformer architecture core problem",
            "important insight architecture",
            "solving problem with transformers"
        ]
        
        for query in high_relevance_queries:
            self.vector_store.retrieve_relevant_memories(
                agent_id=self.agent_id,
                query=query,
                k=1
            )
        
        # Memory 2: Medium relevance - retrieve fewer times with somewhat relevant queries
        medium_relevance_queries = [
            "optimizing encoder-decoder",
            "architecture optimization"
        ]
        
        for query in medium_relevance_queries:
            self.vector_store.retrieve_relevant_memories(
                agent_id=self.agent_id,
                query=query,
                k=1
            )
        
        # Memory 3: Low relevance - retrieve once with low relevance query
        self.vector_store.retrieve_relevant_memories(
            agent_id=self.agent_id,
            query="weather forecast",
            k=1
        )
        
        # Calculate Memory Utility Score for all memories
        all_memory_mus = []
        
        # Get all memories with their metadata
        results = self.vector_store.collection.get(
            ids=self.memory_ids,
            include=["metadatas", "documents"]
        )
        
        # Process each memory
        for i, memory_id in enumerate(self.memory_ids):
            metadata = results["metadatas"][i]
            document = results["documents"][i]
            
            # Extract usage statistics
            retrieval_count = metadata.get("retrieval_count", 0)
            last_retrieved = metadata.get("last_retrieved_timestamp", "")
            relevance_count = metadata.get("retrieval_relevance_count", 0)
            accumulated_relevance = metadata.get("accumulated_relevance_score", 0.0)
            
            # Skip if never retrieved
            if retrieval_count == 0:
                logger.warning(f"Memory {memory_id} was never retrieved, skipping MUS calculation")
                all_memory_mus.append({
                    "id": memory_id,
                    "document": document,
                    "retrieval_count": 0,
                    "mus": 0.0
                })
                continue
            
            # Calculate RFS (Retrieval Frequency Score)
            rfs = math.log(1 + retrieval_count)
            
            # Calculate RS (Relevance Score - average of relevance scores)
            rs = 0.0
            if relevance_count > 0:
                rs = accumulated_relevance / relevance_count
            
            # Calculate RecS (Recency Score)
            recs = 0.0
            if last_retrieved:
                # Convert ISO string to datetime
                try:
                    last_retrieved_dt = datetime.fromisoformat(last_retrieved)
                    now = datetime.utcnow()
                    days_since = (now - last_retrieved_dt).total_seconds() / (24 * 3600)
                    recs = 1.0 / (1.0 + days_since)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid last_retrieved_timestamp format: {last_retrieved}")
                    recs = 0.0
            
            # Calculate MUS (Memory Utility Score)
            mus = (0.4 * rfs) + (0.4 * rs) + (0.2 * recs)
            
            # Store results
            all_memory_mus.append({
                "id": memory_id,
                "document": document,
                "retrieval_count": retrieval_count,
                "rfs": rfs,
                "rs": rs,
                "recs": recs,
                "mus": mus
            })
            
            # Log detailed information
            logger.info(f"Memory: {document[:50]}...")
            logger.info(f"  Retrieval Count: {retrieval_count}")
            if relevance_count > 0:
                logger.info(f"  Avg Relevance: {rs:.3f} ({accumulated_relevance:.3f}/{relevance_count})")
            logger.info(f"  RFS: {rfs:.3f}, RS: {rs:.3f}, RecS: {recs:.3f}")
            logger.info(f"  Memory Utility Score: {mus:.3f}")
        
        # Sort memories by MUS (highest to lowest)
        all_memory_mus.sort(key=lambda x: x["mus"], reverse=True)
        
        # Print ranking
        logger.info("\nMemory Ranking by Utility Score:")
        for i, mem in enumerate(all_memory_mus):
            logger.info(f"{i+1}. MUS={mem['mus']:.3f}, Retrievals={mem['retrieval_count']}: {mem['document'][:50]}...")
        
        # Demonstrate pruning decision
        if len(all_memory_mus) >= 3:
            # Verify that Memory 1 has a higher utility score than Memory 3
            high_relevance_mem = next((m for m in all_memory_mus if "transformer architecture" in m["document"]), None)
            low_relevance_mem = next((m for m in all_memory_mus if "weather" in m["document"]), None)
            
            if high_relevance_mem and low_relevance_mem:
                self.assertGreater(
                    high_relevance_mem["mus"], 
                    low_relevance_mem["mus"],
                    "High relevance memory should have higher utility score than low relevance memory"
                )
                logger.info("✓ High relevance memory has higher utility score than low relevance memory")
            
            # Top N retention demonstration
            top_n = 2  # Example: we want to keep top 2 memories
            memories_to_keep = all_memory_mus[:top_n]
            memories_to_prune = all_memory_mus[top_n:]
            
            logger.info(f"\nPruning demonstration - keeping top {top_n} memories:")
            for mem in memories_to_keep:
                logger.info(f"KEEP: MUS={mem['mus']:.3f}: {mem['document'][:50]}...")
            
            for mem in memories_to_prune:
                logger.info(f"PRUNE: MUS={mem['mus']:.3f}: {mem['document'][:50]}...")
            
            self.assertTrue(
                len(memories_to_keep) > 0 and len(memories_to_prune) > 0,
                "Should have both memories to keep and prune in this test"
            )


if __name__ == '__main__':
    unittest.main() 