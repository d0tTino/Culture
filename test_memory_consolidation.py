#!/usr/bin/env python
"""
Test script to verify the hierarchical memory consolidation functionality.
Runs a simulation for multiple steps to populate agent memories and then
checks if consolidated summaries are generated and stored properly.
"""

import logging
import sys
import time
import os
import json
import shutil
from pathlib import Path
from src.app import create_base_simulation
from src.agents.core.roles import ROLE_INNOVATOR, ROLE_ANALYZER, ROLE_FACILITATOR
from src.infra.llm_client import generate_response

# Define the log file path
LOG_FILE = "memory_consolidation_test.log"

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

# Set more verbose logging for specific modules
logging.getLogger('src.sim.simulation').setLevel(logging.DEBUG)
logging.getLogger('src.agents.graphs.basic_agent_graph').setLevel(logging.DEBUG)
logging.getLogger('src.agents.core.base_agent').setLevel(logging.DEBUG)
logging.getLogger('src.infra.memory.vector_store').setLevel(logging.DEBUG)

# Create a specific logger for this test script
logger = logging.getLogger("test_memory_consolidation")
logger.setLevel(logging.INFO)  # Ensure logger level matches basicConfig

# Add a direct file handler to the specific logger as well to ensure messages are captured
file_handler = logging.FileHandler(LOG_FILE, mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Log startup message to verify logging is working
logger.info("Starting memory consolidation test script")
logger.info(f"Logging to file: {os.path.abspath(LOG_FILE)}")

def test_memory_consolidation():
    """
    Tests that the hierarchical memory consolidation works properly.
    Creates agents, runs a simulation to populate memories, and then verifies:
    1. Consolidated summaries are generated
    2. Summaries are stored in agent memory
    3. Summaries are persisted to the vector store
    """
    logger.info("Starting memory consolidation verification test")
    
    # Create a test scenario that will encourage varied agent interactions
    test_scenario = """
    MEMORY CONSOLIDATION TEST SCENARIO:
    
    This is a simulation to test memory consolidation functionality.
    
    The goal is to generate varied interactions between agents that will be
    consolidated into meaningful summaries.
    
    Each agent should:
    1. Share information frequently through messages
    2. Propose ideas about different topics
    3. Ask questions about others' ideas
    4. Form different relationship dynamics
    5. Participate in projects together
    
    These varied interactions will populate the short-term memory and test
    the system's ability to generate coherent session summaries.
    """
    
    # Set vector store directory for later inspection - use test-specific location
    vector_store_dir = "./test_chroma_db"
    
    # Remove any previous test database
    if os.path.exists(vector_store_dir):
        logger.info(f"Removing previous test ChromaDB at {vector_store_dir}")
        shutil.rmtree(vector_store_dir)
    
    # Create a simulation with 3 agents, each with a different role
    sim = create_base_simulation(
        num_agents=3,
        use_vector_store=True,  # Enable vector store for memory persistence
        scenario=test_scenario,
        steps=10,  # Run for 10 steps to ensure enough memories for consolidation
        vector_store_dir=vector_store_dir  # Explicitly set the vector store directory
    )
    
    # Prepare the simulation with varied roles and initial conditions
    agents = sim.agents
    agents[0].state.role = ROLE_INNOVATOR
    agents[1].state.role = ROLE_ANALYZER
    agents[2].state.role = ROLE_FACILITATOR
    
    # Set different initial states to encourage varied interactions
    for i, agent in enumerate(agents):
        agent.state.ip = 15.0  # Enough resources for various actions
        agent.state.du = 20.0
        # Initialize relationships to create varied agent dynamics
        for j, other_agent in enumerate(agents):
            if i != j:  # Don't set relationship with self
                agent.state.relationships[other_agent.agent_id] = (i - j) * 0.2  # Creates varied relationship scores
    
    # Make sure LLM client is available for consolidation
    from src.infra.llm_client import get_default_llm_client
    sim.llm_client = get_default_llm_client()
    
    # Log the initial state of each agent
    logger.info("Initial agent states:")
    for agent in agents:
        logger.info(f"Agent {agent.agent_id}: Role={agent.state.role}, IP={agent.state.ip}, DU={agent.state.du}")
        logger.info(f"  Relationships: {agent.state.relationships}")
        logger.info(f"  Short-term memory size: {len(agent.state.short_term_memory)}")
    
    # Run the simulation for the specified steps
    logger.info("Running simulation...")
    sim.run(sim.steps_to_run)
    
    # Log the final state of each agent
    logger.info("Final agent states:")
    for agent in agents:
        logger.info(f"Agent {agent.agent_id}: Role={agent.state.role}, IP={agent.state.ip}, DU={agent.state.du}")
        logger.info(f"  Short-term memory size: {len(agent.state.short_term_memory)}")
        
        # Count consolidated summaries in short-term memory
        consolidated_count = sum(1 for memory in agent.state.short_term_memory if memory.get('type') == 'consolidated_summary')
        logger.info(f"  Consolidated summaries in short-term memory: {consolidated_count}")
        
        # Print a sample of short-term memory entries to verify content
        logger.info(f"  Sample of short-term memory entries:")
        for memory in list(agent.state.short_term_memory)[-5:]:  # Show last 5 entries
            memory_step = memory.get('step', 'unknown')
            memory_type = memory.get('type', 'unknown')
            memory_content = memory.get('content', 'No content')[:100] + "..." if len(memory.get('content', '')) > 100 else memory.get('content', 'No content')
            logger.info(f"    Step {memory_step}, {memory_type}: {memory_content}")
    
    # Ensure the ChromaDB client persists changes by forcing the vector store to flush to disk
    if hasattr(sim, 'vector_store_manager') and sim.vector_store_manager:
        if hasattr(sim.vector_store_manager, 'client') and sim.vector_store_manager.client:
            logger.info("Ensuring ChromaDB changes are persisted to disk...")
            try:
                # Some versions of ChromaDB might have a persist method
                if hasattr(sim.vector_store_manager.client, 'persist'):
                    sim.vector_store_manager.client.persist()
                    logger.info("ChromaDB persist() method called successfully")
            except Exception as e:
                logger.warning(f"Error calling ChromaDB persist() method: {e}")
    
    # Inspect the vector store content (if available)
    logger.info("Checking vector store for consolidated memories...")
    vector_store_path = Path(vector_store_dir)
    if vector_store_path.exists():
        collections_dir = vector_store_path / "collections"
        if collections_dir.exists():
            # List all collections (one per agent)
            collections = [d for d in collections_dir.iterdir() if d.is_dir()]
            logger.info(f"Found {len(collections)} collections in vector store")
            
            for collection_dir in collections:
                # Try to load metadata from the collection
                metadata_file = collection_dir / "metadata.json"
                count_consolidated = 0
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            
                        # Count consolidated memories
                        if 'documents' in metadata:
                            for doc_info in metadata.get('documents', []):
                                if 'type' in doc_info and doc_info['type'] == 'consolidated_summary':
                                    count_consolidated += 1
                                    
                        logger.info(f"Collection {collection_dir.name}: Found {count_consolidated} consolidated summaries")
                    except Exception as e:
                        logger.error(f"Error reading metadata from {metadata_file}: {e}")
                else:
                    logger.warning(f"No metadata file found for collection {collection_dir.name}")
        else:
            logger.warning(f"No collections directory found in {vector_store_dir}")
    else:
        logger.warning(f"Vector store directory {vector_store_dir} not found")
    
    # Overall assessment
    if consolidated_count > 0:
        logger.info("SUCCESS: Memory consolidation is working - consolidated summaries were generated")
    else:
        logger.warning("FAILURE: No consolidated summaries were found in agent memory")
    
    logger.info("Memory consolidation test complete")
    
    # Uncomment this to clean up the test database after the test
    # if os.path.exists(vector_store_dir):
    #     logger.info(f"Cleaning up test ChromaDB at {vector_store_dir}")
    #     shutil.rmtree(vector_store_dir)

if __name__ == "__main__":
    test_memory_consolidation() 