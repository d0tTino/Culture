import logging
import os
import sys
from pathlib import Path
from typing import cast

import pytest

if os.environ.get("ENABLE_DSPY_TESTS") != "1":
    pytest.skip("DSPy tests disabled", allow_module_level=True)

import pytest
from typing_extensions import Self

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


@pytest.mark.unit
@pytest.mark.dspy
@pytest.mark.critical_path
def test_direct_call() -> None:
    """Test direct call to DSPy thought generator with imported LM."""
    try:
        # Import dspy first
        import dspy

        # Create a properly formed DSPy LM client
        class OllamaLM(dspy.LM):  # type: ignore[no-any-unimported] # Mypy cannot follow dspy.LM import; see https://mypy.readthedocs.io/en/stable/common_issues.html
            model_name: str
            temperature: float

            def __init__(self: Self) -> None:
                self.model_name = "mistral:latest"
                self.temperature = 0.1
                # Pass model parameter to the superclass
                super().__init__(model=self.model_name)
                import ollama

                self.ollama = ollama
                logger.info(f"Initialized OllamaLM with model {self.model_name}")

            def forward(
                self: Self,
                prompt: str | None = None,
                messages: list[dict[str, str]] | None = None,
                **kwargs: object,
            ) -> object:
                if prompt is None and messages:
                    prompt = messages[0].get("content", "")
                text = self.basic_request(prompt or "", **kwargs)
                from types import SimpleNamespace

                choice = SimpleNamespace(
                    message=SimpleNamespace(content=text), finish_reason="stop"
                )
                return SimpleNamespace(model=self.model_name, choices=[choice], usage={})

            def basic_request(self: Self, prompt: str, **kwargs: object) -> str:
                """Required method for dspy.LM that handles basic requests."""
                logger.info(
                    f"Calling Ollama model {self.model_name} with prompt length: {len(prompt)}"
                )
                try:
                    response = self.ollama.chat(
                        model="mistral:latest",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        stream=False,
                    )
                    return cast(str, response["message"]["content"])
                except Exception as e:
                    logger.error(f"Error calling Ollama API: {e}")
                    return f"Error: {e}"

        # Configure DSPy with our properly implemented LM
        lm = OllamaLM()
        dspy.settings.configure(lm=lm)
        logger.info("DSPy configured with OllamaLM")

        # Define the DSPy Signature for role-prefixed thoughts
        class RolePrefixedThought(dspy.Signature):  # type: ignore[no-any-unimported] # Mypy cannot follow dspy.Signature import; see https://mypy.readthedocs.io/en/stable/common_issues.html
            """
            Generate an agent's internal thought process that strictly begins with
            'As a [ROLE],' or 'As an [ROLE],' and reflects the agent's role and current situation.
            """

            agent_role = dspy.InputField(desc="The role of the agent.")
            current_situation = dspy.InputField(desc="The current situation or context.")
            thought_process = dspy.OutputField(desc="The generated thought process.")

            def __init__(self: Self, agent_role: str, current_situation: str) -> None:
                self.agent_role = agent_role
                self.current_situation = current_situation

            def __call__(self: Self) -> str:
                return f"As a {self.agent_role}, {self.current_situation}"

        # Create the DSPy module
        generate_thought = dspy.Predict(RolePrefixedThought)

        # Test parameters
        agent_role = "Innovator"
        current_situation = (
            "Simulation step 1. Need to introduce myself to others and "
            "potentially propose a creative idea."
        )

        # Call the function directly
        logger.info(f"Calling DSPy generate_thought with role={agent_role}")
        result = generate_thought(agent_role=agent_role, current_situation=current_situation)

        # Display the result
        logger.info("DSPy successfully generated a thought")

        # Clean up the generated thought to handle any unexpected formatting
        thought_process = result.thought_process

        # Check if we need to clean the string
        if not thought_process.startswith("As a") and not thought_process.startswith("As an"):
            # Find the first occurrence of "As a" or "As an"
            as_a_index = thought_process.find("As a")
            as_an_index = thought_process.find("As an")

            # Find the earliest occurrence that's not -1 (not found)
            start_indices = [i for i in [as_a_index, as_an_index] if i != -1]

            if start_indices:
                # Get the earliest occurrence
                start_index = min(start_indices)
                # Trim the string
                thought_process = thought_process[start_index:]
                logger.info("Cleaned thought process by removing leading characters")

        logger.info(f"Generated thought: {thought_process}")

        # Verify role prefix
        expected_prefix_a = f"As a {agent_role},"
        expected_prefix_an = f"As an {agent_role},"

        prefix_is_correct = thought_process.startswith(
            expected_prefix_a
        ) or thought_process.startswith(expected_prefix_an)

        if prefix_is_correct:
            logger.info("SUCCESS: Generated thought correctly starts with the role prefix")
        else:
            logger.warning("WARNING: Generated thought does not start with expected prefix")
            logger.warning(f"Actual beginning: {thought_process[:50]}...")

        # Use assertion instead of return value
        assert prefix_is_correct, (
            f"Generated thought does not start with expected role prefix: "
            f"{thought_process[:50]}..."
        )
    except Exception as e:
        logger.error(f"Error in DSPy direct test: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        # Fail the test with the appropriate error message
        assert False, f"DSPy direct test failed with error: {e}"


if __name__ == "__main__":
    test_direct_call()
