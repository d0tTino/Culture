"""
DSPy-powered Action Intent Selector Module

This module loads and provides a DSPy-optimized module for selecting appropriate
action intents for agents based on their role, goals, and current situation.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)
logger.info("====== IMPORTING DSPY ACTION INTENT SELECTOR MODULE ======")

try:
    from src.infra.dspy_ollama_integration import configure_dspy_with_ollama, dspy

    logger.info("Successfully imported DSPy module")
except ImportError as e:
    logger.error(f"Failed to import DSPy module: {e}")
    # Print more debugging information
    import traceback

    logger.error(traceback.format_exc())
    raise


# dspy lacks type hints, so stubs provide minimal types
class ActionIntentSelection(dspy.Signature):
    """
    Given the agent's role, current situation, overarching goal, and available actions,
    select the most appropriate action intent and provide a brief justification.
    """

    agent_role = dspy.InputField(
        desc="The agent's current role (e.g., Innovator, Analyzer, Facilitator)."
    )
    current_situation = dspy.InputField(
        desc=(
            "Concise summary of the current environmental state, recent events, "
            "and perceived information."
        )
    )
    agent_goal = dspy.InputField(
        desc="The agent's primary objective or goal for the current turn or phase."
    )
    available_actions = dspy.InputField(
        desc="A list of valid action intents the agent can choose from."
    )

    chosen_action_intent = dspy.OutputField(
        desc="The single, most appropriate action intent selected from the available_actions list."
    )
    justification_thought = dspy.OutputField(
        desc=(
            "A brief thought process explaining why this action_intent was chosen given the "
            "role, situation, and goal."
        )
    )


select_action_intent_module = dspy.Predict(ActionIntentSelection)

OPTIMIZED_PROGRAM_PATH = (
    Path(__file__).resolve().parent / "compiled" / "optimized_action_selector.json"
)


def load_optimized_program() -> dict[str, Any] | None:
    """
    Load the optimized program from the JSON file.

    Returns:
        Optional[Dict[str, Any]]: The loaded program data if successful, None otherwise
    """
    logger.info(
        f"ACTION SELECTOR: Attempting to load optimized program from {OPTIMIZED_PROGRAM_PATH}"
    )
    try:
        if not OPTIMIZED_PROGRAM_PATH.exists():
            logger.warning(
                f"ACTION SELECTOR: Optimized program file not found at {OPTIMIZED_PROGRAM_PATH}"
            )
            return None

        with OPTIMIZED_PROGRAM_PATH.open() as f:
            program_data = json.load(f)
        if isinstance(program_data, dict):
            logger.info("ACTION SELECTOR: Successfully loaded optimized action intent selector")
            return program_data
        logger.error("ACTION SELECTOR: Loaded program data is not a dict")
        return None
    except Exception as e:
        logger.error(f"ACTION SELECTOR: Error loading optimized program: {e}")
        return None


class FailsafeActionIntentSelector:
    """
    Failsafe version of the ActionIntentSelector. Always returns a safe default action intent.
    """

    def __call__(self: "FailsafeActionIntentSelector", *args: object, **kwargs: object) -> object:
        return type(
            "FailsafeResult",
            (),
            {
                "chosen_action_intent": "idle",
                "justification_thought": (
                    "Failsafe: Unable to process action intent due to DSPy error."
                ),
            },
        )()


def get_optimized_action_selector() -> (
    Any
):  # Justification: DSPy module returns dynamic callable, type unknown
    """
    Get the optimized action intent selector module with robust fallback logic.
    Returns a callable that tries optimized, then base, then failsafe at call time.
    """
    logger.info("ACTION SELECTOR: get_optimized_action_selector() called")
    # Try to load optimized and base selectors at import time
    optimized_selector = None
    base_selector = None
    try:
        logger.info("ACTION SELECTOR: Configuring DSPy with Ollama")
        configure_dspy_with_ollama()
        logger.info("ACTION SELECTOR: Successfully configured DSPy with Ollama")
    except Exception as e:
        logger.error(f"ACTION SELECTOR: Error configuring DSPy with Ollama: {e}")
    try:
        program_data = load_optimized_program()
        if program_data:
            logger.info("ACTION SELECTOR: Using optimized action selector")
            optimized_selector = select_action_intent_module
    except Exception as e:
        logger.error(f"ACTION SELECTOR: Error loading optimized program: {e}")
    try:
        logger.warning("ACTION SELECTOR: Using basic (unoptimized) selector as fallback")
        base_selector = select_action_intent_module
    except Exception as e:
        logger.critical(f"ACTION SELECTOR: Failed to load base selector: {e}")
    failsafe_selector = FailsafeActionIntentSelector()

    def selector_wrapper(*args: object, **kwargs: object) -> object:
        # Try optimized
        if optimized_selector:
            try:
                return optimized_selector(*args, **kwargs)
            except Exception as e:
                logger.error(f"ACTION SELECTOR: Optimized selector failed at call time: {e}")
        # Try base
        if base_selector:
            try:
                return base_selector(*args, **kwargs)
            except Exception as e:
                logger.error(f"ACTION SELECTOR: Base selector failed at call time: {e}")
        # Failsafe
        logger.critical("ACTION SELECTOR: All selector calls failed. Using failsafe selector.")
        return failsafe_selector(*args, **kwargs)

    return selector_wrapper


def test_module() -> bool:
    """Test the module with a simple example."""
    try:
        logger.info("ACTION SELECTOR: Testing module with a simple example")

        # Get the optimized action selector
        action_selector = get_optimized_action_selector()

        # Create a test example
        test_example = {
            "agent_role": "Facilitator",
            "current_situation": "The discussion has stalled with multiple competing ideas.",
            "agent_goal": "Help the group reach consensus and make progress.",
            "available_actions": [
                "propose_idea",
                "ask_clarification",
                "continue_collaboration",
                "idle",
            ],
        }

        logger.info(f"ACTION SELECTOR: Calling action selector with test example: {test_example}")

        # Call the module
        prediction = action_selector(**test_example)

        logger.info(
            f"ACTION SELECTOR: Test prediction - chosen action: {prediction.chosen_action_intent}"
        )
        logger.info(
            f"ACTION SELECTOR: Test prediction - justification: {prediction.justification_thought[:100]}..."
        )

        return True
    except Exception as e:
        logger.error(f"ACTION SELECTOR: Test failed: {e}")
        import traceback

        logger.error(f"ACTION SELECTOR: {traceback.format_exc()}")
        return False


def get_failsafe_output(*args: object, **kwargs: object) -> object:
    return type(
        "FailsafeResult",
        (),
        {
            "chosen_action_intent": "idle",
            "justification_thought": "Failsafe: Unable to process action intent due to DSPy error.",
        },
    )()


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_module()
