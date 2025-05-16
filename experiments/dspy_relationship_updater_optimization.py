import os
import logging
import sys
import dspy

# Proper DSPy LM subclass for Ollama
class OllamaLM(dspy.LM):
    def __init__(self, model_name="ollama/mistral:latest", temperature=0.1):
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

from src.agents.dspy_programs.relationship_updater import RelationshipUpdater
from src.agents.dspy_programs.relationship_examples import examples as relationship_examples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dspy_relationship_updater_optimization")

# Prepare DSPy Examples
trainset = [
    dspy.Example(
        current_relationship_score=ex["current_relationship_score"],
        interaction_summary=ex["interaction_summary"],
        agent1_persona=ex["agent1_persona"],
        agent2_persona=ex["agent2_persona"],
        interaction_sentiment=ex["interaction_sentiment"],
        new_relationship_score=ex["new_relationship_score"],
        relationship_change_rationale=ex["relationship_change_rationale"]
    ).with_inputs(
        "current_relationship_score", "interaction_summary", "agent1_persona", "agent2_persona", "interaction_sentiment"
    ) for ex in relationship_examples
]

# Metric: Score plausibility (LLM-as-judge)
class ScorePlausibilityJudge(dspy.Signature):
    current_relationship_score = dspy.InputField()
    interaction_sentiment = dspy.InputField()
    new_relationship_score = dspy.InputField()
    judgment = dspy.OutputField(desc="plausible or implausible")

score_judge = dspy.Predict(ScorePlausibilityJudge)
def score_plausibility_metric(example, prediction, trace=None):
    judge_result = score_judge(
        current_relationship_score=example.current_relationship_score,
        interaction_sentiment=example.interaction_sentiment,
        new_relationship_score=prediction.new_relationship_score
    )
    return 1.0 if ("plausible" in judge_result.judgment.lower()) else 0.0

# Metric: Rationale coherence (LLM-as-judge)
class RationaleCoherenceJudge(dspy.Signature):
    interaction_summary = dspy.InputField()
    rationale = dspy.InputField()
    judgment = dspy.OutputField(desc="coherent or incoherent")

rationale_judge = dspy.Predict(RationaleCoherenceJudge)
def rationale_coherence_metric(example, prediction, trace=None):
    judge_result = rationale_judge(
        interaction_summary=example.interaction_summary,
        rationale=prediction.relationship_change_rationale
    )
    return 1.0 if ("coherent" in judge_result.judgment.lower()) else 0.0

# Combined metric (average)
def combined_metric(example, prediction, trace=None):
    return 0.5 * score_plausibility_metric(example, prediction) + 0.5 * rationale_coherence_metric(example, prediction)

# Run BootstrapFewShot optimization
from dspy.teleprompt import BootstrapFewShot
optimizer = BootstrapFewShot(metric=combined_metric, max_labeled_demos=4, max_bootstrapped_demos=4)
logger.info("Starting DSPy BootstrapFewShot optimization for RelationshipUpdater...")
optimized_program = optimizer.compile(dspy.Predict(RelationshipUpdater), trainset=trainset)

# Ensure instructions are present before saving
if not hasattr(optimized_program, 'instructions') or not optimized_program.instructions:
    if hasattr(RelationshipUpdater, 'instructions'):
        optimized_program.instructions = RelationshipUpdater.instructions
    elif hasattr(RelationshipUpdater, 'signature') and hasattr(RelationshipUpdater.signature, 'instructions'):
        optimized_program.instructions = RelationshipUpdater.signature.instructions
    else:
        # Fallback: use the desc fields to construct instructions
        optimized_program.instructions = (
            "Inputs: current_relationship_score (Current relationship score between the two agents (e.g., -1.0 to 1.0)), "
            "interaction_summary (Summary of the most recent interaction between the agents), "
            "agent1_persona (Persona/role description of agent 1), "
            "agent2_persona (Persona/role description of agent 2), "
            "interaction_sentiment (Sentiment score of the interaction (-1.0 to 1.0))\n"
            "Outputs: new_relationship_score (Updated relationship score after the interaction), "
            "relationship_change_rationale (Explanation for the relationship score change)"
        )

# Save the optimized program
save_path = os.path.join(os.path.dirname(__file__), "../src/agents/dspy_programs/compiled/optimized_relationship_updater.json")
optimized_program.save(save_path)
logger.info(f"Optimized RelationshipUpdater saved to {save_path}")

# Evaluate on trainset
scores = [combined_metric(ex, optimized_program(
    current_relationship_score=ex.current_relationship_score,
    interaction_summary=ex.interaction_summary,
    agent1_persona=ex.agent1_persona,
    agent2_persona=ex.agent2_persona,
    interaction_sentiment=ex.interaction_sentiment
)) for ex in trainset]
logger.info(f"Final average combined metric on trainset: {sum(scores)/len(scores):.3f}")

# Simple validation
for ex in trainset[:3]:
    pred = optimized_program(
        current_relationship_score=ex.current_relationship_score,
        interaction_summary=ex.interaction_summary,
        agent1_persona=ex.agent1_persona,
        agent2_persona=ex.agent2_persona,
        interaction_sentiment=ex.interaction_sentiment
    )
    logger.info(f"Input: score={ex.current_relationship_score}, summary={ex.interaction_summary}\nOutput: new_score={pred.new_relationship_score}, rationale={pred.relationship_change_rationale}\n---") 