import logging
import dspy

logger = logging.getLogger(__name__)

class RoleThoughtGenerator(dspy.Signature):
    """Generate an agent's internal thought process that strictly begins with 'As a [ROLE],' or 'As an [ROLE],' and reflects the agent's role and current situation."""
    role_name = dspy.InputField(desc="The assigned role of the agent.")
    context = dspy.InputField(desc="The current situational context or recent interactions.")
    thought = dspy.OutputField(desc="The agent's generated thought, prefixed by their role.")

try:
    from src.agents.dspy_programs.role_thought_examples import examples as role_thought_examples
except ImportError:
    logger.error("Could not import role_thought_examples. Please ensure the examples file exists.")
    role_thought_examples = []

class _RolePrefixedThoughtResult:
    def __init__(self, dspy_result):
        self.thought_process = getattr(dspy_result, 'thought', None)
        self._dspy_result = dspy_result

# Create the DSPy module
_generate_role_prefixed_thought_module = dspy.Predict(RoleThoughtGenerator)

def generate_role_prefixed_thought(agent_role: str, current_situation: str):
    """
    Generate a role-prefixed thought using the DSPy module.
    Returns an object with a .thought_process attribute (for compatibility with agent graph).
    """
    dspy_result = _generate_role_prefixed_thought_module(
        role_name=agent_role,
        context=current_situation
    )
    return _RolePrefixedThoughtResult(dspy_result) 