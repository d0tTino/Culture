# ruff: noqa: E501, ANN101, ANN401
import logging
import os

import dspy

# from src.infra.dspy_ollama_integration import OllamaLM, configure_dspy_ollama_async # REMOVED

# from src.utils.logging_utils import get_logger # REMOVED
logger = logging.getLogger(__name__)  # ADDED Standard Python Logger


class RoleThoughtGenerator(dspy.Signature):  # type: ignore[misc, no-any-unimported]  # Mypy cannot follow dspy.Signature import
    """
    Generate an agent's internal thought process that strictly begins with 'As a [ROLE],' or
    'As an [ROLE],' and reflects the agent's role and current situation.
    """

    role_name = dspy.InputField(desc="The assigned role of the agent.")
    context = dspy.InputField(desc="The current situational context or recent interactions.")
    thought = dspy.OutputField(desc="The agent's generated thought, prefixed by their role.")


try:
    from src.agents.dspy_programs.role_thought_examples import examples as role_thought_examples
except ImportError:
    logger.error("Could not import role_thought_examples. Please ensure the examples file exists.")
    role_thought_examples = []


class _RolePrefixedThoughtResult:
    def __init__(self: "_RolePrefixedThoughtResult", dspy_result: object) -> None:
        self.thought_process = getattr(dspy_result, "thought", None)
        self._dspy_result = dspy_result


class FailsafeRoleThoughtGenerator:
    """
    Failsafe version of the RoleThoughtGenerator. Always returns a safe default thought.
    """

    def __call__(self: "FailsafeRoleThoughtGenerator", role_name: str, context: str) -> object:
        return type(
            "FailsafeResult",
            (),
            {"thought": f"Failsafe: Unable to generate thought for role {role_name}."},
        )()


def get_role_thought_generator() -> object:
    """
    Get the role thought generator module with robust fallback logic.
    Returns the optimized module if available, else the base, else a failsafe.
    """
    try:
        # For now, assume DSPy is configured globally by conftest.py or similar
        if not dspy.settings.lm:
            logger.error(
                "DSPy LM not configured prior to calling async_generate_role_prefixed_thought. This will likely fail."
            )
            # Attempt a last-ditch configuration, though this is not ideal
            # await configure_dspy_ollama_async() # Still keep this part commented for now to see if global config holds

        # Try to load optimized/compiled version
        compiled_path = os.path.join(
            os.path.dirname(__file__), "compiled", "optimized_role_thought_generator.json"
        )
        generator = dspy.Predict(RoleThoughtGenerator)
        if os.path.exists(compiled_path):
            logger.debug(f"Attempting to load optimized RoleThoughtGenerator from {compiled_path}")
            try:
                generator.load(compiled_path)
                logger.info(
                    f"ROLE THOUGHT GENERATOR: Loaded optimized generator from {compiled_path}"
                )
                return generator
            except Exception as e:
                logger.error(f"ROLE THOUGHT GENERATOR: Failed to load optimized generator: {e}")
        else:
            logger.warning(
                f"Optimized RoleThoughtGenerator JSON not found at {compiled_path}. Will attempt base generator."
            )
        logger.info("Using base (unoptimized) RoleThoughtGenerator.")
        return generator
    except Exception as e:
        logger.critical(
            f"ROLE THOUGHT GENERATOR: All loading attempts failed. Using failsafe generator. Error: {e}"
        )
        return FailsafeRoleThoughtGenerator()


def generate_role_prefixed_thought(agent_role: str, current_situation: str) -> str:
    """
    Generate a role-prefixed thought using the robust loader.
    Returns a string thought process (for compatibility with agent graph).
    """
    generator = get_role_thought_generator()
    logger.debug(
        f"RoleThoughtGenerator inputs: agent_role='{agent_role}' (type: {type(agent_role)}), current_situation='{current_situation[:200]}...' (type: {type(current_situation)})"
    )
    if callable(generator):
        dspy_result = generator(role_name=agent_role, context=current_situation)
        return str(
            getattr(
                _RolePrefixedThoughtResult(dspy_result),
                "thought_process",
                "Failsafe: Unable to generate thought.",
            )
        )
    return "Failsafe: Unable to generate thought (generator not callable)."


def get_failsafe_output(*args: object, **kwargs: object) -> str:
    role_name = args[0] if args else kwargs.get("role_name", "unknown")
    return f"Failsafe: Unable to generate thought for role {role_name}."


def test_role_prefixed_thought() -> None:
    # Implementation of test_role_prefixed_thought method
    pass


def test_signature_fields() -> None:
    # Implementation of test_signature_fields method
    pass


def test_signature_output() -> None:
    # Implementation of test_signature_output method
    pass


def test_signature_input_fields() -> None:
    # Implementation of test_signature_input_fields method
    pass
