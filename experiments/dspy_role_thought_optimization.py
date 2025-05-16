import os
import logging
import sys

import dspy

# Proper DSPy LM subclass for Ollama
class OllamaLM(dspy.LM):
    def __init__(self, model_name="mistral:latest", temperature=0.1):
        super().__init__(model=model_name)
        self.model_name = model_name
        self.temperature = temperature
        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            print("Failed to import ollama. Please install with 'pip install ollama'")
            sys.exit(1)
    def basic_request(self, prompt, **kwargs):
        response = self.ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=False
        )
        return response["message"]["content"]

lm = OllamaLM(model_name="ollama/mistral:latest", temperature=0.1)
dspy.settings.configure(lm=lm)

from src.agents.dspy_programs.role_thought_generator import RoleThoughtGenerator
from src.agents.dspy_programs.role_thought_examples import examples as role_thought_examples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dspy_role_thought_optimization")

# Prepare DSPy Examples
trainset = [
    dspy.Example(
        role_name=ex["role_name"],
        context=ex["context"],
        thought=ex["thought"]
    ).with_inputs("role_name", "context") for ex in role_thought_examples
]

# Metric: Role prefix adherence
def role_prefix_adherence_metric(example, prediction, trace=None):
    role = example.role_name
    thought = prediction.thought
    expected_prefix_a = f"As a {role},"
    expected_prefix_an = f"As an {role},"
    return 1.0 if (thought.startswith(expected_prefix_a) or thought.startswith(expected_prefix_an)) else 0.0

# Metric: LLM-as-judge for relevance (using DSPy Predict)
class RelevanceJudgeSignature(dspy.Signature):
    """Judge if the thought is relevant and appropriate for the role/context."""
    role_name = dspy.InputField()
    context = dspy.InputField()
    thought = dspy.InputField()
    judgment = dspy.OutputField(desc="yes or no")

judge = dspy.Predict(RelevanceJudgeSignature)
def thought_relevance_metric(example, prediction, trace=None):
    judge_result = judge(
        role_name=example.role_name,
        context=example.context,
        thought=prediction.thought
    )
    return 1.0 if ("yes" in judge_result.judgment.lower()) else 0.0

# Combined metric (average)
def combined_metric(example, prediction, trace=None):
    return 0.5 * role_prefix_adherence_metric(example, prediction) + 0.5 * thought_relevance_metric(example, prediction)

# Run BootstrapFewShot optimization
from dspy.teleprompt import BootstrapFewShot
optimizer = BootstrapFewShot(metric=combined_metric, max_labeled_demos=4, max_bootstrapped_demos=4)
logger.info("Starting DSPy BootstrapFewShot optimization for RoleThoughtGenerator...")
optimized_program = optimizer.compile(dspy.Predict(RoleThoughtGenerator), trainset=trainset)

# Save the optimized program
save_path = os.path.join(os.path.dirname(__file__), "../src/agents/dspy_programs/compiled/optimized_role_thought_generator.json")
optimized_program.save(save_path)
logger.info(f"Optimized RoleThoughtGenerator saved to {save_path}")

# Evaluate on trainset
scores = [combined_metric(ex, optimized_program(role_name=ex.role_name, context=ex.context)) for ex in trainset]
logger.info(f"Final average combined metric on trainset: {sum(scores)/len(scores):.3f}")

# Simple validation
for ex in trainset[:3]:
    pred = optimized_program(role_name=ex.role_name, context=ex.context)
    logger.info(f"Input: role={ex.role_name}, context={ex.context}\nOutput: {pred.thought}\n---") 