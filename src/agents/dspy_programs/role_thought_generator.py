import os
import logging
import sys

# Configure basic logging to debug the import process
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("====== IMPORTING DSPy ROLE THOUGHT GENERATOR MODULE ======")

try:
    import dspy
    logger.info("Successfully imported DSPy module")
except ImportError as e:
    logger.error(f"Failed to import DSPy module: {e}")
    raise

# SimpleOllamaClientForDSPy implementation for local language model
class SimpleOllamaClientForDSPy:
    """A simple Ollama client for DSPy to use with local Ollama models."""
    
    def __init__(self, model_name="mistral:latest", temperature=0.1):
        self.model_name = model_name
        self.temperature = temperature
        try:
            import ollama
            self.ollama = ollama
            logger.info(f"Successfully initialized Ollama client with model: {model_name}")
        except ImportError:
            logger.error("Failed to import ollama. Please install with 'pip install ollama'")
            sys.exit(1)

    def __call__(self, prompt, **kwargs):
        """Call the Ollama API with the given prompt."""
        try:
            logger.info(f"Calling Ollama API with model {self.model_name}")
            response = self.ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Error: {e}"

# Configure DSPy with the SimpleOllamaClientForDSPy
try:
    lm = SimpleOllamaClientForDSPy(model_name="mistral:latest", temperature=0.1)
    dspy.settings.configure(lm=lm)
    logger.info("DSPy configured with SimpleOllamaClientForDSPy")
except Exception as e:
    logger.error(f"Failed to configure DSPy with Ollama client: {e}")

# Define the DSPy Signature for role-prefixed thoughts
class RolePrefixedThought(dspy.Signature):
    """Generate an agent's internal thought process that strictly begins with 'As a [ROLE],' or 'As an [ROLE],' and reflects the agent's role and current situation."""
    agent_role = dspy.InputField(desc="The current role of the agent (e.g., Innovator, Analyzer, Facilitator).")
    current_situation = dspy.InputField(desc="A concise summary of the agent's current context, including previous thoughts, perceptions, goals, and relevant environmental information.")
    thought_process = dspy.OutputField(desc="The agent's internal thought process. CRITICAL: This MUST start with 'As a {agent_role},' or 'As an {agent_role},'.")

# Create a simple DSPy Module
generate_role_prefixed_thought_module = dspy.Predict(RolePrefixedThought)

# Training examples for DSPy optimization
trainset = [
    dspy.Example(
        agent_role="Innovator",
        current_situation="Team needs new ideas for user engagement.",
        thought_process="As an Innovator, I'm thinking we could try a viral marketing stunt combined with gamification elements to boost user engagement. What if we created a challenge that encourages users to share their experiences with our product?"
    ),
    dspy.Example(
        agent_role="Analyzer",
        current_situation="Project performance metrics are declining.",
        thought_process="As an Analyzer, I notice several concerning trends in the performance data. The click-through rates have dropped by 15% since last month, while load times have increased. This suggests we may have introduced performance regressions in the latest update."
    ),
    dspy.Example(
        agent_role="Facilitator",
        current_situation="Team meeting to resolve conflict between departments.",
        thought_process="As a Facilitator, I need to ensure everyone feels heard while guiding the conversation toward productive solutions. I'll start by acknowledging both departments' concerns, then suggest a structured approach to identify common goals."
    ),
    dspy.Example(
        agent_role="Expert",
        current_situation="Team asking about technical feasibility of new feature.",
        thought_process="As an Expert, I can see several technical challenges with implementing this feature. The current database schema doesn't support the hierarchical structure needed, and we'd need to refactor the authentication system to accommodate the new permission levels."
    ),
    dspy.Example(
        agent_role="Architect",
        current_situation="Planning a system redesign to improve scalability.",
        thought_process="As an Architect, I believe we should consider moving to a microservices architecture with event-driven communication. This would address our current scalability bottlenecks while allowing teams to deploy independently, though it will require significant refactoring of the monolith."
    )
]

# Define a metric function to evaluate role adherence
def role_adherence_metric(example, prediction):
    """Evaluate if the generated thought correctly starts with 'As a [ROLE],' or 'As an [ROLE],'.
    Returns 1.0 if correct, 0.0 otherwise."""
    role = example.agent_role
    thought = prediction.thought_process
    
    # Check if thought starts with either "As a [ROLE]," or "As an [ROLE],"
    expected_prefix_a = f"As a {role},"
    expected_prefix_an = f"As an {role},"
    
    if thought.startswith(expected_prefix_a) or thought.startswith(expected_prefix_an):
        return 1.0
    else:
        return 0.0

# Optimize the module using the trainset and metric
try:
    # Direct usage without optimization for simplicity
    generate_role_prefixed_thought = generate_role_prefixed_thought_module
    
    # For a more robust implementation, we could use teleprompter or other DSPy optimization
    # approaches, but for now we'll use the basic predict module
    logger.info("Using the basic DSPy predict module for role-prefixed thought generation")
except Exception as e:
    logger.warning(f"Error in setup: {e}")
    generate_role_prefixed_thought = generate_role_prefixed_thought_module

# Create an __init__.py file to make the module importable
init_file_path = os.path.join(os.path.dirname(__file__), "__init__.py")
if not os.path.exists(init_file_path):
    with open(init_file_path, "w") as f:
        f.write("")
    logger.info(f"Created __init__.py at {init_file_path}")

def test_module():
    """Test the module with a simple example."""
    try:
        prediction = generate_role_prefixed_thought(
            agent_role="Tester",
            current_situation="Testing the DSPy role thought generator."
        )
        logger.info(f"Test prediction: {prediction.thought_process[:100]}...")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--test" in sys.argv:
        test_module() 