#!/usr/bin/env python
"""
Script to check the actual pruning status in the ChromaDB database.
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from src.infra.memory.vector_store import ChromaVectorStoreManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Manually set pruning configuration constants
MEMORY_PRUNING_ENABLED = True
MEMORY_PRUNING_L1_DELAY_STEPS = 5
MEMORY_PRUNING_L2_ENABLED = True
MEMORY_PRUNING_L2_MAX_AGE_DAYS = 30
MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS = 10

def check_memory_pruning():
    """Check the pruning status of memories in the database."""
    # Connect to the test ChromaDB
    vector_store = ChromaVectorStoreManager(persist_directory="./test_chroma_pruning_48fe8e")
    logger.info("Connected to ChromaDB for pruning verification")
    
    # Print pruning configuration
    logger.info(f"L1 Pruning configuration: ENABLED={MEMORY_PRUNING_ENABLED}, DELAY={MEMORY_PRUNING_L1_DELAY_STEPS}")
    logger.info(f"L2 Pruning configuration: ENABLED={MEMORY_PRUNING_L2_ENABLED}, MAX_AGE={MEMORY_PRUNING_L2_MAX_AGE_DAYS} days, CHECK_INTERVAL={MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS} steps")
    
    # Get all agents 
    all_agents = ["agent_1", "agent_2", "agent_3"]
    
    for agent_id in all_agents:
        # Check level 1 summaries
        logger.info(f"\nChecking Level 1 summaries for {agent_id}")
        try:
            # In ChromaDB, the 'where' clause needs to use the right format
            l1_summaries = vector_store.collection.get(
                where={"$and": [
                    {"agent_id": {"$eq": agent_id}},
                    {"memory_type": {"$eq": "consolidated_summary"}}
                ]},
                include=["metadatas"]
            )
            
            steps = []
            if "metadatas" in l1_summaries and l1_summaries["metadatas"]:
                for metadata in l1_summaries["metadatas"]:
                    if "step" in metadata:
                        steps.append(int(metadata["step"]))
                
                steps.sort()
                logger.info(f"Found {len(steps)} Level 1 summaries at steps: {steps}")
                
                # Expected pruning - level 1 summaries before step 15 should be pruned
                if MEMORY_PRUNING_ENABLED:
                    if steps and min(steps) <= 10:
                        logger.warning(f"PRUNING ISSUE: Found Level 1 summaries at step {min(steps)}, which should have been pruned")
                    else:
                        logger.info("Level 1 pruning appears to be working correctly.")
            else:
                logger.info("No Level 1 summaries found")
        except Exception as e:
            logger.error(f"Error querying Level 1 summaries: {e}")
        
        # Check level 2 summaries
        logger.info(f"\nChecking Level 2 summaries for {agent_id}")
        try:
            l2_summaries = vector_store.collection.get(
                where={"$and": [
                    {"agent_id": {"$eq": agent_id}},
                    {"memory_type": {"$eq": "chapter_summary"}}
                ]},
                include=["metadatas"]
            )
            
            l2_steps = []
            l2_timestamps = []
            if "metadatas" in l2_summaries and l2_summaries["metadatas"]:
                for metadata in l2_summaries["metadatas"]:
                    if "step" in metadata:
                        l2_steps.append(int(metadata["step"]))
                    if "simulation_step_end_timestamp" in metadata:
                        l2_timestamps.append(metadata["simulation_step_end_timestamp"])
                
                l2_steps.sort()
                logger.info(f"Found {len(l2_steps)} Level 2 summaries at steps: {l2_steps}")
                
                # Check if any L2 summaries are older than MAX_AGE_DAYS
                if MEMORY_PRUNING_L2_ENABLED and l2_timestamps:
                    now = datetime.utcnow()
                    cutoff_date = now - timedelta(days=MEMORY_PRUNING_L2_MAX_AGE_DAYS)
                    old_summaries = 0
                    
                    for timestamp_str in l2_timestamps:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str)
                            if timestamp < cutoff_date:
                                old_summaries += 1
                        except ValueError:
                            logger.warning(f"Could not parse timestamp: {timestamp_str}")
                    
                    if old_summaries > 0:
                        logger.warning(f"PRUNING CHECK: Found {old_summaries} Level 2 summaries older than {MEMORY_PRUNING_L2_MAX_AGE_DAYS} days that could be pruned")
                    else:
                        logger.info(f"No Level 2 summaries older than {MEMORY_PRUNING_L2_MAX_AGE_DAYS} days found")
            else:
                logger.info("No Level 2 summaries found")
        except Exception as e:
            logger.error(f"Error querying Level 2 summaries: {e}")
    
    # Check the database directly
    logger.info("\nDirect check of the vector store:")
    total_count = vector_store.collection.count()
    logger.info(f"Total memories in the database: {total_count}")
    
    # Verify if target_pruning_end_step calculation is correct
    logger.info("\nVerifying target_pruning_end_step logic:")
    for current_step in [15, 25, 30]:
        pruning_delay = MEMORY_PRUNING_L1_DELAY_STEPS
        target_pruning_end_step = current_step - pruning_delay
        L2_consolidation_interval = 10
        
        should_prune = target_pruning_end_step > 0 and target_pruning_end_step % L2_consolidation_interval == 0
        
        logger.info(f"Step={current_step}, target_pruning_end_step={target_pruning_end_step}, should_prune={should_prune}")
        
        if should_prune:
            prune_start_step = target_pruning_end_step - L2_consolidation_interval + 1
            prune_end_step = target_pruning_end_step
            logger.info(f"  Would prune steps {prune_start_step}-{prune_end_step}")
            
            # Check if these steps were pruned
            for agent_id in all_agents:
                try:
                    # Using correct ChromaDB query format
                    pruned_check = vector_store.collection.get(
                        where={"$and": [
                            {"agent_id": {"$eq": agent_id}},
                            {"memory_type": {"$eq": "consolidated_summary"}},
                            {"step": {"$gte": prune_start_step}},
                            {"step": {"$lte": prune_end_step}}
                        ]},
                        include=["metadatas"]
                    )
                    
                    if "ids" in pruned_check and pruned_check["ids"]:
                        logger.warning(f"  PRUNING ISSUE: Agent {agent_id} has {len(pruned_check['ids'])} Level 1 summaries in range {prune_start_step}-{prune_end_step} which should have been pruned")
                    else:
                        logger.info(f"  OK: Agent {agent_id} has no Level 1 summaries in range {prune_start_step}-{prune_end_step}")
                except Exception as e:
                    logger.error(f"  Error checking pruning for agent {agent_id}: {e}")

if __name__ == "__main__":
    check_memory_pruning() 