#!/usr/bin/env python
"""
Test script to verify the hierarchical memory persistence functionality in ChromaDB.
This script validates that both Level 1 (session) and Level 2 (chapter) memory 
summaries are correctly persisted to and retrievable from ChromaDB.
"""

import unittest
import logging
import sys
import time
import os
import json
import shutil
import re
from pathlib import Path
from src.app import create_base_simulation
from src.agents.core.roles import ROLE_INNOVATOR, ROLE_ANALYZER, ROLE_FACILITATOR
from src.infra.llm_client import generate_response

# Define the log file path
LOG_FILE = "test_hierarchical_memory_persistence.log"

# Remove any existing log file to start fresh
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

# Configure logging - ensure we have file logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,  # Force reconfiguration to avoid issues with existing loggers
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode='w')  # Use 'w' mode to ensure fresh log file
    ]
)

# Set more verbose logging for relevant modules
logging.getLogger('src.sim.simulation').setLevel(logging.DEBUG)
logging.getLogger('src.agents.graphs.basic_agent_graph').setLevel(logging.DEBUG)
logging.getLogger('src.agents.core.base_agent').setLevel(logging.DEBUG)
logging.getLogger('src.infra.memory.vector_store').setLevel(logging.DEBUG)

# Create a specific logger for this test script
logger = logging.getLogger("test_hierarchical_memory_persistence")
logger.setLevel(logging.INFO)  # Ensure logger level matches basicConfig

# Add a direct file handler to the specific logger as well to ensure messages are captured
file_handler = logging.FileHandler(LOG_FILE, mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Log startup message to verify logging is working
logger.info("Starting hierarchical memory persistence test script")
logger.info(f"Logging to file: {os.path.abspath(LOG_FILE)}")

class TestHierarchicalMemoryPersistence(unittest.TestCase):
    """
    Test case to verify hierarchical memory persistence in ChromaDB.
    Tests both Level 1 (session) summaries and Level 2 (chapter) summaries.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by:
        1. Creating a test-specific ChromaDB directory
        2. Running a simulation to generate memory data
        3. Ensuring all memory consolidation processes complete
        """
        logger.info("Setting up test environment for hierarchical memory persistence tests")
        
        # Set vector store directory - use test-specific location
        cls.vector_store_dir = "./test_chroma_db_persistence_specific"
        
        # Remove any previous test database
        if os.path.exists(cls.vector_store_dir):
            logger.info(f"Removing previous test ChromaDB at {cls.vector_store_dir}")
            shutil.rmtree(cls.vector_store_dir)
        
        # Create a test scenario that will encourage varied agent interactions
        test_scenario = """
        HIERARCHICAL MEMORY PERSISTENCE TEST SCENARIO:
        
        This simulation tests the ChromaDB persistence of hierarchical memory.
        
        The goal is to generate varied interactions that will be consolidated into:
        1. Level 1 (session) summaries
        2. Level 2 (chapter) summaries
        
        Each agent should:
        - Share information frequently
        - Propose ideas about different topics
        - Ask questions and form varied relationship dynamics
        - Act in ways that generate rich memory content
        
        This simulation will run for enough steps to ensure multiple Level 1 and Level 2 consolidations.
        """
        
        # Create a simulation with 3 agents, each with a different role
        # Run for 25 steps to ensure multiple Level 1 and Level 2 consolidations
        cls.sim = create_base_simulation(
            num_agents=3,
            use_vector_store=True,
            scenario=test_scenario,
            steps=25,
            vector_store_dir=cls.vector_store_dir
        )
        
        # Configure the agents with varied roles and initial conditions
        agents = cls.sim.agents
        agents[0].state.role = ROLE_INNOVATOR
        agents[1].state.role = ROLE_ANALYZER
        agents[2].state.role = ROLE_FACILITATOR
        
        # Initialize agent states to encourage varied interactions
        for i, agent in enumerate(agents):
            agent.state.ip = 15.0
            agent.state.du = 20.0
            # Set different relationship values
            for j, other_agent in enumerate(agents):
                if i != j:
                    agent.state.relationships[other_agent.agent_id] = (i - j) * 0.2
        
        # Ensure LLM client is available for consolidation
        from src.infra.llm_client import get_default_llm_client
        cls.sim.llm_client = get_default_llm_client()
        
        # Store agent IDs for later testing
        cls.agent_ids = [agent.agent_id for agent in agents]
        
        # Log initial state
        logger.info("Initial agent states configured with varied roles and relationships")
        
        # Run the simulation for the specified steps
        logger.info(f"Running simulation for {cls.sim.steps_to_run} steps...")
        cls.sim.run(cls.sim.steps_to_run)
        logger.info("Simulation completed")
        
        # Ensure the ChromaDB client persists changes
        if hasattr(cls.sim, 'vector_store_manager') and cls.sim.vector_store_manager:
            if hasattr(cls.sim.vector_store_manager, 'client') and cls.sim.vector_store_manager.client:
                logger.info("Ensuring ChromaDB changes are persisted to disk...")
                try:
                    # Some versions of ChromaDB might have a persist method
                    if hasattr(cls.sim.vector_store_manager.client, 'persist'):
                        cls.sim.vector_store_manager.client.persist()
                        logger.info("ChromaDB persist() method called successfully")
                except Exception as e:
                    logger.warning(f"Error calling ChromaDB persist() method: {e}")
        
        # Store reference to vector store manager for test methods
        cls.vector_store_manager = cls.sim.vector_store_manager
        
        # Wait a moment to ensure all operations complete
        time.sleep(2)
        
        logger.info("Test environment setup complete")
    
    def test_level1_summaries_persistence(self):
        """
        Test that Level 1 (session) summaries are properly persisted to ChromaDB.
        
        Verifies:
        1. Each agent has Level 1 summaries in ChromaDB
        2. Summaries have appropriate metadata
        3. Summaries have non-empty, meaningful content
        """
        logger.info("Testing Level 1 (session) summary persistence")
        
        # For each agent, query their Level 1 summaries from ChromaDB
        for agent_id in self.agent_ids:
            # Query for Level 1 consolidated summaries
            query = f"consolidated summaries for agent {agent_id}"
            
            level1_summaries = self.vector_store_manager.retrieve_filtered_memories(
                agent_id=agent_id,
                query_text=query,
                filters={"memory_type": "consolidated_summary"},
                k=10
            )
            
            # Verify we found at least some Level 1 summaries
            self.assertGreater(
                len(level1_summaries), 
                0, 
                f"Agent {agent_id} should have at least one Level 1 summary in ChromaDB"
            )
            
            logger.info(f"Agent {agent_id} has {len(level1_summaries)} Level 1 summaries in ChromaDB")
            
            # Verify each summary has the correct metadata and content
            for summary in level1_summaries:
                # Check metadata
                self.assertEqual(
                    summary.get("agent_id"), 
                    agent_id, 
                    "Summary should have correct agent_id"
                )
                
                self.assertEqual(
                    summary.get("memory_type"), 
                    "consolidated_summary", 
                    "Summary should have memory_type='consolidated_summary'"
                )
                
                # Verify content
                content = summary.get("content", "")
                self.assertNotEqual(content, "", "Summary content should not be empty")
                self.assertGreater(
                    len(content), 
                    50, 
                    "Summary content should be substantial (>50 chars)"
                )
                
                # Verify step number exists and is reasonable
                self.assertIn("step", summary, "Summary should have a 'step' field")
                step = summary.get("step")
                self.assertGreaterEqual(step, 0, "Step number should be >= 0")
                self.assertLessEqual(
                    step, 
                    self.sim.steps_to_run, 
                    f"Step number should be <= {self.sim.steps_to_run}"
                )
                
                logger.debug(f"Verified Level 1 summary at step {step} for agent {agent_id}")
    
    def test_level2_summaries_persistence(self):
        """
        Test that Level 2 (chapter) summaries are properly persisted to ChromaDB.
        
        Verifies:
        1. Each agent has Level 2 summaries in ChromaDB
        2. Summaries have appropriate metadata
        3. Summaries have non-empty, meaningful content
        """
        logger.info("Testing Level 2 (chapter) summary persistence")
        
        # For each agent, query their Level 2 summaries from ChromaDB
        for agent_id in self.agent_ids:
            # Query for Level 2 chapter summaries
            query = f"chapter summaries for agent {agent_id}"
            
            level2_summaries = self.vector_store_manager.retrieve_filtered_memories(
                agent_id=agent_id,
                query_text=query,
                filters={"memory_type": "chapter_summary"},
                k=5
            )
            
            # Verify we found at least some Level 2 summaries
            # Since Level 2 summaries are generated every ~10 steps,
            # and we ran for 25 steps, we should have at least 2
            self.assertGreater(
                len(level2_summaries), 
                0, 
                f"Agent {agent_id} should have at least one Level 2 summary in ChromaDB"
            )
            
            logger.info(f"Agent {agent_id} has {len(level2_summaries)} Level 2 summaries in ChromaDB")
            
            # Verify each summary has the correct metadata and content
            for summary in level2_summaries:
                # Check metadata
                self.assertEqual(
                    summary.get("agent_id"), 
                    agent_id, 
                    "Summary should have correct agent_id"
                )
                
                self.assertEqual(
                    summary.get("memory_type"), 
                    "chapter_summary", 
                    "Summary should have memory_type='chapter_summary'"
                )
                
                # Verify content
                content = summary.get("content", "")
                self.assertNotEqual(content, "", "Summary content should not be empty")
                self.assertGreater(
                    len(content), 
                    100, 
                    "Chapter summary content should be substantial (>100 chars)"
                )
                
                # Verify step number exists and is reasonable
                self.assertIn("step", summary, "Summary should have a 'step' field")
                step = summary.get("step")
                self.assertGreaterEqual(step, 0, "Step number should be >= 0")
                self.assertLessEqual(
                    step, 
                    self.sim.steps_to_run, 
                    f"Step number should be <= {self.sim.steps_to_run}"
                )
                
                logger.debug(f"Verified Level 2 summary at step {step} for agent {agent_id}")
    
    def test_chronological_ordering(self):
        """
        Test that memory summaries follow chronological ordering.
        
        Verifies:
        1. Level 2 summaries are generated after Level 1 summaries
        2. Level 2 summaries are generated at appropriate intervals (~10 steps)
        """
        logger.info("Testing chronological ordering of memory summaries")
        
        # Test for each agent
        for agent_id in self.agent_ids:
            # Get all summaries for this agent
            query = f"all memory summaries for agent {agent_id}"
            
            all_summaries = self.vector_store_manager.retrieve_relevant_memories(
                agent_id=agent_id,
                query_text=query,
                k=100  # Get a large number to ensure we get all
            )
            
            # Group summaries by type
            level1_summaries = [s for s in all_summaries if s.get("memory_type") == "consolidated_summary"]
            level2_summaries = [s for s in all_summaries if s.get("memory_type") == "chapter_summary"]
            
            # Skip this agent if we don't have both levels of summaries
            if not level1_summaries or not level2_summaries:
                logger.warning(f"Agent {agent_id} doesn't have both level types - skipping chronology test")
                continue
            
            # Group summaries by step
            level1_steps = [s.get("step") for s in level1_summaries if "step" in s]
            level2_steps = [s.get("step") for s in level2_summaries if "step" in s]
            
            # Sort steps for comparison
            level1_steps.sort()
            level2_steps.sort()
            
            # Verify Level 2 summaries appear after some Level 1 summaries
            # (There should be at least one Level 1 summary before the first Level 2)
            self.assertLess(
                level1_steps[0], 
                level2_steps[0], 
                "First Level 1 summary should come before first Level 2 summary"
            )
            
            # Verify Level 2 summaries appear at reasonable intervals
            if len(level2_steps) >= 2:
                for i in range(1, len(level2_steps)):
                    interval = level2_steps[i] - level2_steps[i-1]
                    # Level 2 summaries should be ~10 steps apart (allow 7-13 steps between)
                    self.assertGreaterEqual(
                        interval, 
                        7, 
                        f"Level 2 summaries should be generated at least 7 steps apart (found {interval})"
                    )
                    self.assertLessEqual(
                        interval, 
                        13, 
                        f"Level 2 summaries should be generated at most 13 steps apart (found {interval})"
                    )
            
            logger.info(f"Verified chronological ordering for agent {agent_id}")
    
    def test_content_relevance(self):
        """
        Test that Level 2 summaries actually contain content relevant to Level 1 summaries.
        Uses a heuristic approach by checking for keyword overlap.
        """
        logger.info("Testing content relevance between Level 1 and Level 2 summaries")
        
        # Test for each agent
        for agent_id in self.agent_ids:
            # Get all Level 1 and Level 2 summaries for this agent
            query_l1 = f"consolidated summaries for agent {agent_id}"
            level1_summaries = self.vector_store_manager.retrieve_filtered_memories(
                agent_id=agent_id,
                query_text=query_l1,
                filters={"memory_type": "consolidated_summary"},
                k=30
            )
            
            query_l2 = f"chapter summaries for agent {agent_id}"
            level2_summaries = self.vector_store_manager.retrieve_filtered_memories(
                agent_id=agent_id,
                query_text=query_l2,
                filters={"memory_type": "chapter_summary"},
                k=10
            )
            
            # Skip this agent if we don't have both types of summaries
            if not level1_summaries or not level2_summaries:
                logger.warning(f"Agent {agent_id} doesn't have both summary types - skipping content test")
                continue
            
            # Extract significant keywords from Level 1 summaries
            level1_content = " ".join([s.get("content", "") for s in level1_summaries])
            
            # Simple keyword extraction - get words of 5+ chars that appear at least twice
            words = re.findall(r'\b\w{5,}\b', level1_content.lower())
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            keywords = [word for word, count in word_counts.items() if count >= 2 and word not in 
                       ['about', 'there', 'their', 'would', 'should', 'could', 'which', 'these', 'those']]
            
            # Check if Level 2 summaries contain these keywords
            for level2_summary in level2_summaries:
                level2_content = level2_summary.get("content", "").lower()
                
                # Check for keyword overlap
                found_keywords = [keyword for keyword in keywords if keyword in level2_content]
                
                # We should find at least some keyword overlap
                self.assertGreater(
                    len(found_keywords), 
                    0, 
                    f"Level 2 summary should contain at least one keyword from Level 1 summaries"
                )
                
                overlap_percentage = len(found_keywords) / len(keywords) if keywords else 0
                logger.info(f"Agent {agent_id} Level 2 summary has {len(found_keywords)}/{len(keywords)} keyword overlap ({overlap_percentage:.1%})")
                
                # Log the found keywords for debugging
                if found_keywords:
                    logger.debug(f"Keywords found in Level 2 summary: {', '.join(found_keywords)}")
    
    def test_metadata_storage(self):
        """
        Test that memory summaries have correct and complete metadata.
        
        Verifies:
        1. Required metadata fields are present
        2. Metadata values are correct
        """
        logger.info("Testing metadata storage for memory summaries")
        
        # Define required metadata fields by memory type
        level1_required_fields = ["agent_id", "step", "memory_type", "event_type", "content"]
        level2_required_fields = ["agent_id", "step", "memory_type", "event_type", "content"]
        
        # Test for each agent
        for agent_id in self.agent_ids:
            # Test Level 1 summaries
            query_l1 = f"consolidated summaries for agent {agent_id}"
            level1_summaries = self.vector_store_manager.retrieve_filtered_memories(
                agent_id=agent_id,
                query_text=query_l1,
                filters={"memory_type": "consolidated_summary"},
                k=5
            )
            
            for summary in level1_summaries:
                # Check all required fields are present
                for field in level1_required_fields:
                    self.assertIn(
                        field, 
                        summary, 
                        f"Level 1 summary should have '{field}' field"
                    )
                
                # Check metadata values
                self.assertEqual(
                    summary.get("agent_id"), 
                    agent_id, 
                    "agent_id metadata should match"
                )
                
                self.assertEqual(
                    summary.get("memory_type"), 
                    "consolidated_summary", 
                    "memory_type should be 'consolidated_summary'"
                )
                
                # Check event_type is appropriate
                self.assertIn(
                    summary.get("event_type"), 
                    ["consolidated_summary", "memory_consolidation"], 
                    "event_type should indicate memory consolidation"
                )
            
            # Test Level 2 summaries
            query_l2 = f"chapter summaries for agent {agent_id}"
            level2_summaries = self.vector_store_manager.retrieve_filtered_memories(
                agent_id=agent_id,
                query_text=query_l2,
                filters={"memory_type": "chapter_summary"},
                k=5
            )
            
            for summary in level2_summaries:
                # Check all required fields are present
                for field in level2_required_fields:
                    self.assertIn(
                        field, 
                        summary, 
                        f"Level 2 summary should have '{field}' field"
                    )
                
                # Check metadata values
                self.assertEqual(
                    summary.get("agent_id"), 
                    agent_id, 
                    "agent_id metadata should match"
                )
                
                self.assertEqual(
                    summary.get("memory_type"), 
                    "chapter_summary", 
                    "memory_type should be 'chapter_summary'"
                )
                
                # Check event_type is appropriate
                self.assertEqual(
                    summary.get("event_type"), 
                    "chapter_summary", 
                    "event_type should be 'chapter_summary'"
                )
                
                # Level 2 should optionally have the is_level_2 metadata flag
                if "metadata" in summary and isinstance(summary["metadata"], dict):
                    if "is_level_2" in summary["metadata"]:
                        self.assertTrue(
                            summary["metadata"]["is_level_2"], 
                            "is_level_2 flag should be True for Level 2 summaries"
                        )
            
            logger.info(f"Verified metadata for agent {agent_id}")
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up after tests - comment out the cleanup code if you want to inspect the database.
        """
        logger.info("Tests complete - cleaning up")
        
        # Uncomment to delete the test ChromaDB after tests
        # if os.path.exists(cls.vector_store_dir):
        #     logger.info(f"Removing test ChromaDB at {cls.vector_store_dir}")
        #     shutil.rmtree(cls.vector_store_dir)

if __name__ == "__main__":
    unittest.main() 