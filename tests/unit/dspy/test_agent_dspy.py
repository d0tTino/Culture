import logging
import sys
from pathlib import Path

import pytest

pytest.skip("DSPy wrapper manual test", allow_module_level=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# First test direct import
logger.info("Test 1: Direct import of DSPy role thought generator")
try:
    from src.agents.dspy_programs.role_thought_generator import generate_role_prefixed_thought

    logger.info("SUCCESS: Direct import worked")
except ImportError as e:
    logger.error(f"FAILED: Direct import failed: {e}")
    import traceback

    logger.error(f"Traceback: {traceback.format_exc()}")

# Then test importing basic_agent_graph and check DSPY_AVAILABLE flag
logger.info("Test 2: Import basic_agent_graph and check DSPY_AVAILABLE flag")
try:
    from src.agents.graphs.basic_agent_graph import DSPY_AVAILABLE

    logger.info(f"SUCCESS: basic_agent_graph import worked, DSPY_AVAILABLE = {DSPY_AVAILABLE}")
except ImportError as e:
    logger.error(f"FAILED: basic_agent_graph import failed: {e}")
    import traceback

    logger.error(f"Traceback: {traceback.format_exc()}")

# Test calling the DSPy module directly
logger.info("Test 3: Try to call the DSPy module directly")
try:
    if "generate_role_prefixed_thought" in locals():
        result = generate_role_prefixed_thought(
            agent_role="Tester",
            current_situation="Testing the DSPy role thought generator directly.",
        )
        if hasattr(result, "thought_process"):
            logger.info(f"SUCCESS: DSPy call worked, result: {result.thought_process[:100]}...")
        else:
            logger.info(f"SUCCESS: DSPy call worked, result: {str(result)[:100]}...")
    else:
        logger.warning(
            "SKIPPED: Cannot run test 3 because generate_role_prefixed_thought is not available"
        )
except Exception as e:
    logger.error(f"FAILED: DSPy call failed: {e}")
    import traceback

    logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    logger.info("Tests completed")
