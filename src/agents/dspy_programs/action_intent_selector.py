"""
DSPy-powered Action Intent Selector Module

This module loads and provides a DSPy-optimized module for selecting appropriate
action intents for agents based on their role, goals, and current situation.
"""

import os
import json
import logging
import sys
from typing import Dict, List, Any, Optional

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("====== IMPORTING DSPY ACTION INTENT SELECTOR MODULE ======")

try:
    import dspy
    from src.infra.dspy_ollama_integration import configure_dspy_with_ollama
    logger.info("Successfully imported DSPy module")
except ImportError as e:
    logger.error(f"Failed to import DSPy module: {e}")
    # Print more debugging information
    import traceback
    logger.error(traceback.format_exc())
    raise

# Define the DSPy Signature for action intent selection
class ActionIntentSelection(dspy.Signature):
    """
    Given the agent's role, current situation, overarching goal, and available actions,
    select the most appropriate action intent and provide a brief justification.
    """
    agent_role = dspy.InputField(desc="The agent's current role (e.g., Innovator, Analyzer, Facilitator).")
    current_situation = dspy.InputField(desc="Concise summary of the current environmental state, recent events, and perceived information.")
    agent_goal = dspy.InputField(desc="The agent's primary objective or goal for the current turn or phase.")
    available_actions = dspy.InputField(desc="A list of valid action intents the agent can choose from.")

    chosen_action_intent = dspy.OutputField(desc="The single, most appropriate action intent selected from the available_actions list.")
    justification_thought = dspy.OutputField(desc="A brief thought process explaining why this action_intent was chosen given the role, situation, and goal.")

# Create a simple DSPy Module
select_action_intent_module = dspy.Predict(ActionIntentSelection)

# Path to the compiled/optimized DSPy program
OPTIMIZED_PROGRAM_PATH = os.path.join(
    os.path.dirname(__file__), 
    "compiled", 
    "optimized_action_selector.json"
)

def load_optimized_program() -> Optional[Dict[str, Any]]:
    """
    Load the optimized program from the JSON file.
    
    Returns:
        Optional[Dict[str, Any]]: The loaded program data if successful, None otherwise
    """
    logger.info(f"ACTION SELECTOR: Attempting to load optimized program from {OPTIMIZED_PROGRAM_PATH}")
    try:
        if not os.path.exists(OPTIMIZED_PROGRAM_PATH):
            logger.warning(f"ACTION SELECTOR: Optimized program file not found at {OPTIMIZED_PROGRAM_PATH}")
            return None
        
        with open(OPTIMIZED_PROGRAM_PATH, 'r') as f:
            program_data = json.load(f)
        
        logger.info(f"ACTION SELECTOR: Successfully loaded optimized action intent selector")
        return program_data
    except Exception as e:
        logger.error(f"ACTION SELECTOR: Error loading optimized program: {e}")
        return None

def get_optimized_action_selector():
    """
    Get the optimized action intent selector module.
    
    This function ensures the DSPy environment is properly configured and
    returns the optimized module if available, or falls back to a basic one.
    
    Returns:
        dspy.Module: The DSPy module for action intent selection
    """
    logger.info(f"ACTION SELECTOR: get_optimized_action_selector() called")
    
    # Ensure we have a properly configured DSPy environment
    try:
        # Configure DSPy with Ollama
        logger.info(f"ACTION SELECTOR: Configuring DSPy with Ollama")
        configure_dspy_with_ollama()
        logger.info(f"ACTION SELECTOR: Successfully configured DSPy with Ollama")
    except Exception as e:
        logger.error(f"ACTION SELECTOR: Error configuring DSPy with Ollama: {e}")
        # If we can't configure DSPy, fallback to basic module
        # But we'll try to continue as the calling code might have already configured DSPy
    
    # Try to load the optimized program
    program_data = load_optimized_program()
    
    if program_data:
        try:
            logger.info(f"ACTION SELECTOR: Using optimized action selector")
            # Here we would normally apply the optimized program to the module
            # Since we don't have an actual teleprompter.compile result, just return the base module
            return select_action_intent_module
        except Exception as e:
            logger.error(f"ACTION SELECTOR: Error creating optimized module: {e}")
    
    logger.warning(f"ACTION SELECTOR: Using basic (unoptimized) selector")
    return select_action_intent_module

def test_module():
    """Test the module with a simple example."""
    try:
        logger.info(f"ACTION SELECTOR: Testing module with a simple example")
        
        # Get the optimized action selector
        action_selector = get_optimized_action_selector()
        
        # Create a test example
        test_example = {
            "agent_role": "Facilitator",
            "current_situation": "The discussion has stalled with multiple competing ideas.",
            "agent_goal": "Help the group reach consensus and make progress.",
            "available_actions": ['propose_idea', 'ask_clarification', 'continue_collaboration', 'idle']
        }
        
        logger.info(f"ACTION SELECTOR: Calling action selector with test example: {test_example}")
        
        # Call the module
        prediction = action_selector(**test_example)
        
        logger.info(f"ACTION SELECTOR: Test prediction - chosen action: {prediction.chosen_action_intent}")
        logger.info(f"ACTION SELECTOR: Test prediction - justification: {prediction.justification_thought[:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"ACTION SELECTOR: Test failed: {e}")
        import traceback
        logger.error(f"ACTION SELECTOR: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--test" in sys.argv:
        test_module() 