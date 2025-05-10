#!/usr/bin/env python
"""
Test script to verify the memory pruning functionality in hierarchical memory.
This test verifies that Level 1 (session) summaries are properly pruned after
they have been consolidated into Level 2 (chapter) summaries, respecting the
configured delay between L2 creation and L1 pruning.
"""

import unittest
import logging
import sys
import time
import os
import json
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.app import create_base_simulation
from src.agents.core.roles import ROLE_INNOVATOR, ROLE_ANALYZER, ROLE_FACILITATOR
from src.infra.llm_client import generate_response
from src.infra.memory.vector_store import ChromaVectorStoreManager
from src.infra import config

# Define the log file path
LOG_FILE = "test_memory_pruning.log"

# Remove any existing log file to start fresh
if os.path.exists(LOG_FILE):
    try:
        os.remove(LOG_FILE)
    except PermissionError:
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not remove existing log file {LOG_FILE} - it may be in use by another process. Will append to it.")

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
logger = logging.getLogger("test_memory_pruning")
logger.setLevel(logging.INFO)

# Add a direct file handler to the specific logger
file_handler = logging.FileHandler(LOG_FILE, mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Log startup message
logger.info("Starting memory pruning test script")

# Override configuration settings for this test
# This is a bit hacky but avoids modifying the config file directly
config.MEMORY_PRUNING_ENABLED = True
config.MEMORY_PRUNING_L1_DELAY_STEPS = 5  # Small delay for testing purposes
config.MEMORY_PRUNING_L2_ENABLED = True
config.MEMORY_PRUNING_L2_MAX_AGE_DAYS = 30  # Max age for L2 summaries
config.MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS = 10  # Check interval for testing purposes
logger.info(f"Memory pruning settings for test: ENABLED={config.MEMORY_PRUNING_ENABLED}, DELAY={config.MEMORY_PRUNING_L1_DELAY_STEPS}")
logger.info(f"L2 pruning settings for test: ENABLED={config.MEMORY_PRUNING_L2_ENABLED}, MAX_AGE={config.MEMORY_PRUNING_L2_MAX_AGE_DAYS} days, CHECK_INTERVAL={config.MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS} steps")

class TestMemoryPruning(unittest.TestCase):
    """
    Test case to verify memory pruning functionality.
    Tests that Level 1 summaries are properly pruned after consolidation into Level 2 summaries.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by:
        1. Creating a test-specific ChromaDB directory
        2. Running a simulation to generate memory data and trigger pruning
        """
        logger.info("Setting up test environment for memory pruning tests")
        
        # Set vector store directory - use test-specific location
        cls.vector_store_dir = f"./test_chroma_pruning_{uuid.uuid4().hex[:6]}"
        
        # Remove any previous test database
        if os.path.exists(cls.vector_store_dir):
            logger.info(f"Removing previous test ChromaDB at {cls.vector_store_dir}")
            shutil.rmtree(cls.vector_store_dir)
        
        # Create a test scenario
        test_scenario = """
        MEMORY PRUNING TEST SCENARIO:
        
        This simulation tests the memory pruning functionality.
        Agents should interact normally to generate varied memory entries.
        The simulation will run long enough to trigger multiple levels of memory consolidation
        and pruning of older Level 1 summaries.
        
        Pruning should occur automatically once Level 1 summaries have been consolidated
        into Level 2 summaries and the configured delay has passed.
        """
        
        # The number of steps to run - needs to be long enough for pruning to occur
        # With L2 consolidation every 10 steps and pruning delay of 5:
        # - First L2 summary at step 10
        # - Pruning of L1s (steps 1-10) at step 15
        # - Second L2 summary at step 20
        # - Pruning of L1s (steps 11-20) at step 25
        # So we need at least 25-30 steps
        sim_steps = 30
        
        # Create a simulation with 3 agents
        logger.info(f"Creating simulation with {sim_steps} steps")
        cls.sim = create_base_simulation(
            num_agents=3,
            use_vector_store=True,
            scenario=test_scenario,
            steps=sim_steps,
            vector_store_dir=cls.vector_store_dir
        )
        
        # Configure the agents with varied roles
        agents = cls.sim.agents
        agents[0].state.role = ROLE_INNOVATOR
        agents[1].state.role = ROLE_ANALYZER
        agents[2].state.role = ROLE_FACILITATOR
        
        # Initialize agent states to encourage interaction
        for i, agent in enumerate(agents):
            agent.state.ip = 15.0
            agent.state.du = 20.0
        
        # Make sure the LLM client is available for consolidation
        from src.infra.llm_client import get_default_llm_client
        cls.sim.llm_client = get_default_llm_client()
        
        # Store agent IDs for later testing
        cls.agent_ids = [agent.agent_id for agent in agents]
        
        # Log initial state
        logger.info("Initial agent states configured")
        
        # Run the simulation for the specified steps
        logger.info(f"Running simulation for {cls.sim.steps_to_run} steps...")
        cls.sim.run(cls.sim.steps_to_run)
        logger.info("Simulation completed")
        
        # Create a direct connection to the vector store for testing
        cls.vector_store_manager = ChromaVectorStoreManager(persist_directory=cls.vector_store_dir)
        logger.info("Vector store manager connected for testing")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after the tests"""
        if os.path.exists(cls.vector_store_dir):
            shutil.rmtree(cls.vector_store_dir)
            logger.info(f"Removed test ChromaDB directory: {cls.vector_store_dir}")
    
    def test_level1_pruning(self):
        """
        Test that Level 1 summaries are properly pruned after being consolidated
        into Level 2 summaries and after the configured delay has passed.
        """
        logger.info("Testing Level 1 summary pruning")
        
        L2_interval = 10  # Level 2 summaries happen every 10 steps
        pruning_delay = config.MEMORY_PRUNING_L1_DELAY_STEPS
        
        # Test for each agent
        for agent_id in self.agent_ids:
            # For each agent, check several things:
            # 1. Level 1 summaries in early steps (1-10) should be pruned
            # 2. Level 1 summaries from recent steps should still exist
            # 3. All Level 2 summaries should exist
            
            # First, get all Level 1 summaries for this agent
            query = f"all consolidated summaries for agent {agent_id}"
            l1_summaries = self.vector_store_manager.retrieve_filtered_memories(
                agent_id=agent_id,
                query_text=query,
                filters={"memory_type": "consolidated_summary"},
                k=100  # Large number to get all
            )
            
            # Next, get all Level 2 summaries for this agent
            query = f"all chapter summaries for agent {agent_id}"
            l2_summaries = self.vector_store_manager.retrieve_filtered_memories(
                agent_id=agent_id,
                query_text=query,
                filters={"memory_type": "chapter_summary"},
                k=100  # Large number to get all
            )
            
            # Group summaries by step
            l1_steps = [s.get("step") for s in l1_summaries if "step" in s]
            l2_steps = [s.get("step") for s in l2_summaries if "step" in s]
            
            # Log what we found
            logger.info(f"Agent {agent_id}: Found {len(l1_summaries)} Level 1 summaries at steps {sorted(l1_steps)}")
            logger.info(f"Agent {agent_id}: Found {len(l2_summaries)} Level 2 summaries at steps {sorted(l2_steps)}")
            
            # VERIFY: Level 2 summaries should exist at steps 10, 20, 30, etc.
            expected_l2_steps = [step for step in range(L2_interval, self.sim.steps_to_run + 1, L2_interval)]
            for step in expected_l2_steps:
                self.assertIn(
                    step, 
                    l2_steps,
                    f"Expected Level 2 summary at step {step} but none found for agent {agent_id}"
                )
            
            # VERIFY: First group of Level 1 summaries (steps 1-10) should be pruned after delay
            # If simulation ran for >= 15 steps (with default 5 step delay)
            pruned_end_step = L2_interval  # 10
            if self.sim.steps_to_run >= (pruned_end_step + pruning_delay):
                # Check that no L1 summaries exist in the first range (1-10)
                early_steps = [step for step in l1_steps if 1 <= step <= pruned_end_step]
                
                self.assertEqual(
                    len(early_steps), 
                    0, 
                    f"Level 1 summaries from steps 1-{pruned_end_step} should have been pruned but found: {early_steps}"
                )
                
                logger.info(f"Agent {agent_id}: Verified pruning of Level 1 summaries for steps 1-{pruned_end_step}")
            
            # VERIFY: Second group of Level 1 summaries (steps 11-20) should be pruned if enough steps have passed
            second_pruned_end_step = L2_interval * 2  # 20
            if self.sim.steps_to_run >= (second_pruned_end_step + pruning_delay):
                # Check that no L1 summaries exist in the second range (11-20)
                second_steps = [step for step in l1_steps if pruned_end_step < step <= second_pruned_end_step]
                
                self.assertEqual(
                    len(second_steps), 
                    0, 
                    f"Level 1 summaries from steps {pruned_end_step+1}-{second_pruned_end_step} should have been pruned but found: {second_steps}"
                )
                
                logger.info(f"Agent {agent_id}: Verified pruning of Level 1 summaries for steps {pruned_end_step+1}-{second_pruned_end_step}")
            
            # VERIFY: Recent Level 1 summaries (within last 10+delay steps) should still exist
            recent_start_step = self.sim.steps_to_run - L2_interval - pruning_delay
            if recent_start_step > 0:
                # There should be at least some L1 summaries in the most recent range
                recent_steps = [step for step in l1_steps if step > recent_start_step]
                
                # Only assert if we expect to find summaries (if simulation ran long enough)
                if self.sim.steps_to_run > (L2_interval + 1):  # At least one step after first L2 summary
                    self.assertGreater(
                        len(recent_steps), 
                        0, 
                        f"Expected to find recent Level 1 summaries after step {recent_start_step} but found none"
                    )
                    
                    logger.info(f"Agent {agent_id}: Verified existence of recent Level 1 summaries after step {recent_start_step}")
    
    def test_level2_pruning(self):
        """
        Test that Level 2 summaries are properly pruned based on their age.
        This test manually stores L2 summaries with different timestamps and
        verifies they are correctly pruned based on their age.
        """
        logger.info("Testing Level 2 summary age-based pruning")
        
        test_agent_id = self.agent_ids[0]  # Use the first agent for this test
        
        # Get the current time to use as a reference
        current_time = datetime.utcnow()
        
        # Add some test L2 summaries with different ages
        old_summary_content = "This is an old L2 summary that should be pruned."
        recent_summary_content = "This is a recent L2 summary that should not be pruned."
        
        # Create an old summary (from 40 days ago)
        old_timestamp = (current_time - timedelta(days=40)).isoformat()
        old_summary_id = self.vector_store_manager.add_memory(
            agent_id=test_agent_id,
            step=100,  # Use a step number that won't conflict with the simulation
            event_type="chapter_summary",
            content=old_summary_content,
            memory_type="chapter_summary",
            metadata={
                "memory_type": "chapter_summary",
                "simulation_step_end_timestamp": old_timestamp,
                "summary_level": 2,
            }
        )
        
        # Create a recent summary (from 15 days ago)
        recent_timestamp = (current_time - timedelta(days=15)).isoformat()
        recent_summary_id = self.vector_store_manager.add_memory(
            agent_id=test_agent_id,
            step=101,  # Different step number
            event_type="chapter_summary",
            content=recent_summary_content,
            memory_type="chapter_summary",
            metadata={
                "memory_type": "chapter_summary",
                "simulation_step_end_timestamp": recent_timestamp,
                "summary_level": 2,
            }
        )
        
        # Verify both summaries are stored correctly
        self.assertTrue(old_summary_id, "Failed to store old L2 summary")
        self.assertTrue(recent_summary_id, "Failed to store recent L2 summary")
        
        # Call the L2 pruning method with max age of 30 days
        with patch('datetime.datetime') as mock_datetime:
            # Mock current time to be the same as what we used above
            mock_datetime.utcnow.return_value = current_time
            # Mock fromisoformat to return the actual datetime objects
            mock_datetime.fromisoformat = datetime.fromisoformat
            
            # Get the L2 summaries older than 30 days
            old_summaries = self.vector_store_manager.get_l2_summaries_older_than(30)
            
            # Verify only the old summary was found
            self.assertEqual(len(old_summaries), 1, "Should find exactly one old L2 summary")
            self.assertEqual(old_summaries[0], old_summary_id, "Should find the old summary ID")
            
            # Delete the old summaries
            result = self.vector_store_manager.delete_memories_by_ids(old_summaries)
            self.assertTrue(result, "Failed to delete old L2 summaries")
            
            # Verify the old summary was pruned and the recent summary still exists
            all_l2_summaries = self.vector_store_manager.get_l2_summaries_older_than(0)  # Get all L2 summaries
            self.assertEqual(len(all_l2_summaries), 1, "Should be one L2 summary remaining")
            self.assertEqual(all_l2_summaries[0], recent_summary_id, "Recent summary should still exist")
    
    def test_pruning_logging(self):
        """
        Test that memory pruning operations are properly logged.
        """
        # Read the log file to check for pruning log entries
        with open(LOG_FILE, 'r') as f:
            log_content = f.read()
        
        # Check for pruning-related log entries
        self.assertIn(
            "Pruning Level 1 summaries", 
            log_content,
            "Expected to find pruning operation log entries"
        )
        
        self.assertIn(
            "Successfully pruned", 
            log_content,
            "Expected to find successful pruning log entries"
        )
        
        logger.info("Verified logging of pruning operations")

if __name__ == "__main__":
    unittest.main() 