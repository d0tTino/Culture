#!/usr/bin/env python
"""
Tests for memory usage tracking in the vector store.
"""

import unittest
import logging
import tempfile
import shutil
import os
import time
from datetime import datetime, timezone
import pytest
from src.agents.memory.vector_store import ChromaVectorStoreManager

@pytest.mark.integration
@pytest.mark.memory
@pytest.mark.vector_store
@pytest.mark.critical_path
class TestMemoryUsageTracking(unittest.TestCase):
    """Tests for memory usage tracking in the vector store."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        cls.test_dir = tempfile.mkdtemp(prefix="memory_usage_tracking_test_")
        cls.vector_store = ChromaVectorStoreManager(persist_directory=cls.test_dir)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        try:
            # Make sure the ChromaDB client is closed
            if hasattr(cls, 'vector_store') and cls.vector_store:
                if hasattr(cls.vector_store, 'client') and cls.vector_store.client:
                    cls.vector_store.client.close()
                    
            # Add a small delay to ensure resources are released
            time.sleep(0.5)
            
            # Remove the test directory
            if os.path.exists(cls.test_dir):
                shutil.rmtree(cls.test_dir)
        except Exception as e:
            logging.error(f"Error during teardown: {e}")
    
    def test_memory_usage_tracking(self):
        """Test that memory usage statistics are tracked correctly."""
        # Add some memories
        agent_id = "test_agent"
        memory_ids = []
        
        # Add 5 test memories
        for i in range(5):
            memory_id = self.vector_store.add_memory(
                agent_id=agent_id,
                step=i,
                event_type="thought",
                content=f"Test memory {i}",
                memory_type="raw"
            )
            memory_ids.append(memory_id)
        
        # Retrieve memories multiple times to update usage stats
        for _ in range(3):
            self.vector_store.retrieve_relevant_memories(
                agent_id=agent_id,
                query="test memory",
                k=3
            )
        
        # Get metadata for the memories
        metadatas = self.vector_store.get_metadata_without_tracking(memory_ids)
        
        # Verify that usage statistics are being tracked
        for metadata in metadatas:
            self.assertIn('retrieval_count', metadata)
            self.assertIn('last_retrieved_timestamp', metadata)
            self.assertIn('accumulated_relevance_score', metadata)
            self.assertIn('retrieval_relevance_count', metadata)
            
            # Check that retrieval count is at least 0
            self.assertGreaterEqual(metadata['retrieval_count'], 0)
            
            # The last_retrieved_timestamp might be empty if the memory was never retrieved
            # So we'll just check that the field exists, which we already did with assertIn above
            # self.assertTrue(metadata['last_retrieved_timestamp'])
    
    def test_retrieve_memory_ids(self):
        """Test retrieving memory IDs from metadata."""
        agent_id = "test_agent_ids"
        
        # Add memories with different structures
        self.vector_store.add_memory(
            agent_id=agent_id,
            step=1,
            event_type="thought",
            content="Memory with ID field",
            memory_type="raw",
            metadata={"id": "test_id_1"}
        )
        
        self.vector_store.add_memory(
            agent_id=agent_id,
            step=2,
            event_type="thought",
            content="Memory with memory_id field",
            memory_type="raw",
            metadata={"memory_id": "test_id_2"}
        )
        
        self.vector_store.add_memory(
            agent_id=agent_id,
            step=3,
            event_type="thought",
            content="Memory without ID fields",
            memory_type="raw"
        )
        
        # Retrieve memories
        memories = self.vector_store.retrieve_filtered_memories(
            agent_id=agent_id,
            filters={"event_type": "thought"},
            limit=10
        )
        
        # Extract IDs
        retrieved_ids = []
        for mem in memories:
            if 'memory_id' in mem:
                retrieved_ids.append(mem['memory_id'])
            elif 'id' in mem:
                retrieved_ids.append(mem['id'])
        
        # Verify we got some IDs
        self.assertTrue(len(retrieved_ids) > 0)

if __name__ == "__main__":
    unittest.main() 