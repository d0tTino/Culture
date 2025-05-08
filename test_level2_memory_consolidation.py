#!/usr/bin/env python
"""
Test script to verify the level 2 hierarchical memory consolidation functionality.
Runs a simulation for at least 20 steps to ensure level 2 consolidation occurs,
then checks if chapter summaries are generated and stored properly.
"""

import logging
import sys
import time
import os
import json
from pathlib import Path
from src.app import create_base_simulation
from src.agents.core.roles import ROLE_INNOVATOR, ROLE_ANALYZER, ROLE_FACILITATOR
from src.infra.llm_client import generate_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("level2_memory_consolidation_test.log")  # Save logs to file for analysis
    ]
)

# Set more verbose logging for specific modules
logging.getLogger('src.sim.simulation').setLevel(logging.DEBUG)
logging.getLogger('src.agents.graphs.basic_agent_graph').setLevel(logging.DEBUG)
logging.getLogger('src.agents.core.base_agent').setLevel(logging.DEBUG)
# Add additional loggers for memory operations
logging.getLogger('src.infra.memory.vector_store').setLevel(logging.DEBUG)

logger = logging.getLogger("test_level2_memory_consolidation")

def test_level2_memory_consolidation():
    """
    Tests that the level 2 hierarchical memory consolidation works properly.
    Creates agents, runs a simulation for at least 20 steps to ensure level 2
    consolidation occurs, and then verifies:
    1. Level 1 consolidated summaries are generated
    2. Level 2 chapter summaries are generated after at least 10 steps
    3. Chapter summaries are stored in agent memory
    4. Chapter summaries are persisted to the vector store
    """
    logger.info("Starting level 2 memory consolidation verification test")
    
    # Create a test scenario that will encourage varied agent interactions
    test_scenario = """
    LEVEL 2 MEMORY CONSOLIDATION TEST SCENARIO:
    
    This is a simulation to test level 2 memory consolidation functionality.
    
    The goal is to generate varied interactions between agents that will be
    consolidated into meaningful level 1 summaries, which will then be
    consolidated into level 2 chapter summaries.
    
    Each agent should:
    1. Share information frequently through messages
    2. Propose ideas about different topics
    3. Ask questions about others' ideas
    4. Form different relationship dynamics
    5. Participate in projects together
    
    These varied interactions will populate the memory and test
    the system's ability to generate coherent chapter summaries
    after multiple simulation steps.
    """
    
    # Set vector store directory for later inspection
    vector_store_dir = "./chroma_db"
    
    # Create a simulation with 3 agents, each with a different role
    # Run for at least 20 steps to ensure level 2 consolidation occurs (every 10 steps)
    sim = create_base_simulation(
        num_agents=3,
        use_vector_store=True,  # Enable vector store for memory persistence
        scenario=test_scenario,
        steps=25  # Run for 25 steps to ensure level 2 consolidation occurs at least twice
    )
    
    # Prepare the simulation with varied roles and initial conditions
    agents = sim.agents
    agents[0].state.role = ROLE_INNOVATOR
    agents[1].state.role = ROLE_ANALYZER
    agents[2].state.role = ROLE_FACILITATOR
    
    # Set different initial states to encourage varied interactions
    for i, agent in enumerate(agents):
        agent.state.ip = 25.0  # Enough resources for various actions
        agent.state.du = 30.0
        
        # Ensure the last_level_2_consolidation_step is initialized to 0
        agent.state.last_level_2_consolidation_step = 0
        
        # Initialize relationships to create varied agent dynamics
        for j, other_agent in enumerate(agents):
            if i != j:  # Don't set relationship with self
                agent.state.relationships[other_agent.agent_id] = (i - j) * 0.2  # Creates varied relationship scores
    
    # Make sure LLM client is available for consolidation
    from src.infra.llm_client import get_default_llm_client
    llm_client = get_default_llm_client()
    sim.llm_client = llm_client
    
    # Make sure each agent has access to the LLM client for memory consolidation
    for agent in agents:
        agent.state.llm_client = llm_client
    
    # Log the initial state of each agent
    logger.info("Initial agent states:")
    for agent in agents:
        logger.info(f"Agent {agent.agent_id}: Role={agent.state.role}, IP={agent.state.ip}, DU={agent.state.du}")
        logger.info(f"  Relationships: {agent.state.relationships}")
        logger.info(f"  Short-term memory size: {len(agent.state.short_term_memory)}")
        logger.info(f"  Last level 2 consolidation step: {agent.state.last_level_2_consolidation_step}")
    
    # Run the simulation for the specified steps
    logger.info(f"Running simulation for {sim.steps_to_run} steps...")
    sim.run(sim.steps_to_run)
    
    # Log the final state of each agent
    logger.info("Final agent states:")
    for agent in agents:
        logger.info(f"Agent {agent.agent_id}: Role={agent.state.role}, IP={agent.state.ip}, DU={agent.state.du}")
        logger.info(f"  Short-term memory size: {len(agent.state.short_term_memory)}")
        logger.info(f"  Last level 2 consolidation step: {agent.state.last_level_2_consolidation_step}")
        
        # Count level 1 consolidated summaries in short-term memory
        level1_count = sum(1 for memory in agent.state.short_term_memory if memory.get('type') == 'consolidated_summary')
        logger.info(f"  Level 1 consolidated summaries in short-term memory: {level1_count}")
        
        # Count level 2 chapter summaries in short-term memory
        level2_count = sum(1 for memory in agent.state.short_term_memory if memory.get('type') == 'chapter_summary')
        logger.info(f"  Level 2 chapter summaries in short-term memory: {level2_count}")
        
        # Print a sample of the most recent memory entries to verify content
        logger.info(f"  Sample of recent memory entries:")
        for memory in list(agent.state.short_term_memory)[-8:]:  # Show last 8 entries
            memory_step = memory.get('step', 'unknown')
            memory_type = memory.get('type', 'unknown')
            memory_content = memory.get('content', 'No content')[:100] + "..." if len(memory.get('content', '')) > 100 else memory.get('content', 'No content')
            logger.info(f"    Step {memory_step}, {memory_type}: {memory_content}")
    
    # Inspect the vector store content (if available)
    logger.info("Checking vector store for level 2 chapter summaries...")
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
                count_level1 = 0
                count_level2 = 0
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            
                        # Count level 1 and level 2 memories
                        if 'documents' in metadata:
                            for doc_info in metadata.get('documents', []):
                                memory_type = doc_info.get('memory_type', '')
                                if memory_type == 'consolidated_summary':
                                    count_level1 += 1
                                elif memory_type == 'chapter_summary':
                                    count_level2 += 1
                                    
                        logger.info(f"Collection {collection_dir.name}: Found {count_level1} level 1 summaries and {count_level2} level 2 chapter summaries")
                        
                        # List the level 2 summaries if any exist
                        if count_level2 > 0:
                            logger.info(f"Level 2 chapter summaries in collection {collection_dir.name}:")
                            level2_summaries = [doc for doc in metadata.get('documents', []) if doc.get('memory_type') == 'chapter_summary']
                            for i, summary in enumerate(level2_summaries):
                                step = summary.get('step', 'unknown')
                                content = summary.get('content', 'No content')[:150] + "..." if len(summary.get('content', '')) > 150 else summary.get('content', 'No content')
                                logger.info(f"  Chapter Summary {i+1} (Step {step}): {content}")
                    
                    except Exception as e:
                        logger.error(f"Error reading metadata from {metadata_file}: {e}")
                else:
                    logger.warning(f"No metadata file found for collection {collection_dir.name}")
        else:
            logger.warning(f"No collections directory found in {vector_store_dir}")
    else:
        logger.warning(f"Vector store directory {vector_store_dir} not found")
    
    # Overall assessment
    if level2_count > 0:
        logger.info("SUCCESS: Level 2 memory consolidation is working - chapter summaries were generated")
    else:
        logger.warning("FAILURE: No level 2 chapter summaries were found in agent memory")
    
    logger.info("Level 2 memory consolidation test complete")

if __name__ == "__main__":
    test_level2_memory_consolidation() 