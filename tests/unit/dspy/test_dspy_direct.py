import os
import sys
import logging
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

@pytest.mark.unit
@pytest.mark.dspy_program
@pytest.mark.critical_path
def test_direct_call():
    """Test direct call to DSPy thought generator with imported LM."""
    try:
        # Import dspy first
        import dspy
        
        # Create a properly formed DSPy LM client
        class OllamaLM(dspy.LM):
            model_name = "ollama/mistral:latest"
            temperature = 0.1
            
            def __init__(self):
                # Pass model parameter to the superclass
                super().__init__(model=self.model_name)
                import ollama
                self.ollama = ollama
                logger.info(f"Initialized OllamaLM with model {self.model_name}")
                
            def basic_request(self, prompt, **kwargs):
                """Required method for dspy.LM that handles basic requests."""
                logger.info(f"Calling Ollama model {self.model_name} with prompt length: {len(prompt)}")
                try:
                    response = self.ollama.chat(
                        model="mistral:latest",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        stream=False
                    )
                    return response["message"]["content"]
                except Exception as e:
                    logger.error(f"Error calling Ollama API: {e}")
                    return f"Error: {e}"
        
        # Configure DSPy with our properly implemented LM
        lm = OllamaLM()
        dspy.settings.configure(lm=lm)
        logger.info("DSPy configured with OllamaLM")
        
        # Define the DSPy Signature for role-prefixed thoughts
        class RolePrefixedThought(dspy.Signature):
            """Generate an agent's internal thought process that strictly begins with 'As a [ROLE],' or 'As an [ROLE],' and reflects the agent's role and current situation."""
            agent_role = dspy.InputField(desc="The current role of the agent (e.g., Innovator, Analyzer, Facilitator).")
            current_situation = dspy.InputField(desc="A concise summary of the agent's current context, including previous thoughts, perceptions, goals, and relevant environmental information.")
            thought_process = dspy.OutputField(desc="The agent's internal thought process. CRITICAL: This MUST start with 'As a {agent_role},' or 'As an {agent_role},'.")
        
        # Create the DSPy module
        generate_thought = dspy.Predict(RolePrefixedThought)
        
        # Test parameters
        agent_role = "Innovator"
        current_situation = "Simulation step 1. Need to introduce myself to others and potentially propose a creative idea."
        
        # Call the function directly
        logger.info(f"Calling DSPy generate_thought with role={agent_role}")
        result = generate_thought(
            agent_role=agent_role,
            current_situation=current_situation
        )
        
        # Display the result
        logger.info(f"DSPy successfully generated a thought")
        
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
                logger.info(f"Cleaned thought process by removing leading characters")
        
        logger.info(f"Generated thought: {thought_process}")
        
        # Verify role prefix
        expected_prefix_a = f"As a {agent_role},"
        expected_prefix_an = f"As an {agent_role},"
        
        prefix_is_correct = thought_process.startswith(expected_prefix_a) or thought_process.startswith(expected_prefix_an)
        
        if prefix_is_correct:
            logger.info(f"SUCCESS: Generated thought correctly starts with the role prefix")
        else:
            logger.warning(f"WARNING: Generated thought does not start with expected prefix")
            logger.warning(f"Actual beginning: {thought_process[:50]}...")
        
        # Use assertion instead of return value
        assert prefix_is_correct, f"Generated thought does not start with expected role prefix: {thought_process[:50]}..."
    except Exception as e:
        logger.error(f"Error in DSPy direct test: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Fail the test with the appropriate error message
        assert False, f"DSPy direct test failed with error: {e}"

if __name__ == "__main__":
    test_direct_call() 