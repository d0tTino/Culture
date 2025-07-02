import logging
import os
import sys
from typing import cast

import pytest

if os.environ.get("ENABLE_DSPY_TESTS") != "1":
    pytest.skip("DSPy tests disabled", allow_module_level=True)

import dspy
import pytest
from typing_extensions import Self

import ollama

if not hasattr(dspy, "Predict"):
    pytest.skip("dspy Predict not available", allow_module_level=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class OllamaLM(dspy.LM):  # type: ignore[no-any-unimported]
    """A simple wrapper for Ollama to use with DSPy."""

    model_name = "ollama/mistral:latest"
    temperature = 0.1

    def __init__(self: Self) -> None:
        # Pass model parameter to the superclass
        super().__init__(model=self.model_name)
        logger.info(f"Initialized OllamaLM with model: {self.model_name}")

    def basic_request(self: Self, prompt: str, **kwargs: object) -> str:
        """Required method from dspy.LM that handles the basic request to the LM."""
        try:
            logger.info(f"Calling Ollama with prompt length: {len(prompt)} chars")
            response = ollama.chat(
                model="mistral:latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False,
            )
            logger.info("Received response from Ollama")
            return cast(str, response["message"]["content"])
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Error: {e}"

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

        choice = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")
        return SimpleNamespace(model=self.model_name, choices=[choice], usage={})


# Initialize the LM with our Ollama wrapper
lm = OllamaLM()
dspy.settings.configure(lm=lm)
logger.info("DSPy configured with OllamaLM")


# Define the signature for role-prefixed thoughts
class RolePrefixedThought(dspy.Signature):  # type: ignore[no-any-unimported]
    """Generate an agent's internal thought process that strictly begins with 'As a [ROLE],' or
    'As an [ROLE],'
    and reflects the agent's role and current situation.
    """

    agent_role = dspy.InputField(
        desc="The current role of the agent (e.g., Innovator, Analyzer, Facilitator)."
    )
    current_situation = dspy.InputField(
        desc=(
            "A detailed description of the agent's current situation, including previous "
            "thoughts, perceptions, goals, and relevant environmental information."
        )
    )
    thought_process = dspy.OutputField(
        desc=(
            "The agent's internal thought process. CRITICAL: This MUST start with 'As a "
            "{agent_role},' or 'As an {agent_role},'."
        )
    )


# Create a DSPy predictor for generating thoughts
generate_role_prefixed_thought = dspy.Predict(RolePrefixedThought)


@pytest.mark.unit
@pytest.mark.dspy
def test_role_prefix_adherence() -> None:
    """Test that the DSPy program correctly generates thoughts with role prefixes."""

    # Test roles (including ones that need "an" instead of "a")
    test_roles = ["Innovator", "Analyzer", "Facilitator", "Expert", "Architect"]

    # Test situation context
    test_situation = (
        "The team is discussing a new project about improving communication efficiency."
    )

    success_count = 0
    total_tests = len(test_roles)

    logger.info(f"Testing DSPy role adherence with {total_tests} different roles...")

    for role in test_roles:
        logger.info(f"\n=== Testing with role: {role} ===")

        # Generate thought using DSPy
        try:
            # Explicitly pass the language model
            prediction = generate_role_prefixed_thought(
                agent_role=role, current_situation=test_situation, lm=lm
            )

            # Get the raw thought from the model
            raw_thought = prediction.thought_process

            # Clean up the generated thought to handle any unexpected formatting
            thought = raw_thought

            # Check if we need to clean the string
            if not thought.startswith("As a") and not thought.startswith("As an"):
                # Find the first occurrence of "As a" or "As an"
                as_a_index = thought.find("As a")
                as_an_index = thought.find("As an")

                # Find the earliest occurrence that's not -1 (not found)
                start_indices = [i for i in [as_a_index, as_an_index] if i != -1]

                if start_indices:
                    # Get the earliest occurrence
                    start_index = min(start_indices)
                    # Trim the string
                    thought = thought[start_index:]
                    logger.info(f"Cleaned thought by removing {start_index} leading characters")

            logger.info(f"Generated thought: {thought[:100]}...")

            # Check if thought starts with proper role prefix
            expected_prefix_a = f"As a {role},"
            expected_prefix_an = f"As an {role},"

            if thought.startswith(expected_prefix_a) or thought.startswith(expected_prefix_an):
                logger.info(f"✅ ROLE_ADHERENCE: Successfully begins with role phrase '{role}'")
                success_count += 1
            else:
                logger.error(
                    f"❌ ROLE_ADHERENCE: Does NOT begin with expected role phrase '{role}'"
                )
                logger.error(f"   Thought begins with: '{thought[:50]}...'")

        except Exception as e:
            logger.error(f"Error generating thought for role {role}: {e}")

    # Report overall success rate
    success_rate = (success_count / total_tests) * 100
    logger.info("\n=== TEST SUMMARY ===")
    logger.info(f"Success rate: {success_rate:.1f}% ({success_count}/{total_tests} successful)")

    if success_count == total_tests:
        logger.info(
            "✅ All tests passed! DSPy role adherence implementation is working correctly."
        )
    else:
        logger.warning(
            f"⚠️ {total_tests - success_count} tests failed. DSPy role adherence needs improvement."
        )

    # Using assertion instead of return value
    assert (
        success_count == total_tests
    ), f"Only {success_count}/{total_tests} role adherence tests passed"


if __name__ == "__main__":
    test_role_prefix_adherence()
    # Exit with status code 0 as the test will raise an AssertionError if it fails
    sys.exit(0)
