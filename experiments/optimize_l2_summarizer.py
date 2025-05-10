#!/usr/bin/env python
"""
Optimization script for L2 summary generator using DSPy BootstrapFewShot.

This script optimizes the DSPy-based L2 summary generation by:
1. Loading example data from l2_summary_examples.py
2. Defining an evaluation metric for summary quality
3. Using BootstrapFewShot to find optimal few-shot examples
4. Saving the optimized model for use in the agent
"""

import os
import sys
import logging
from typing import Dict, List, Optional
import json
from pathlib import Path
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiments/l2_summary_optimization.log")
    ]
)
logger = logging.getLogger("l2_summary_optimizer")

# Import DSPy and related modules
try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    from src.agents.dspy_programs.l2_summary_generator import L2SummaryGenerator, GenerateL2SummarySignature
    from src.agents.dspy_programs.l2_summary_examples import example1, example2, example3
    from src.infra.dspy_ollama_integration import configure_dspy_with_ollama, OllamaLM
    
    logger.info("Successfully imported DSPy modules")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Configure DSPy with Ollama
ollama_model = "mistral:latest"
temperature = 0.1

logger.info(f"Configuring DSPy with Ollama model: {ollama_model}, temperature: {temperature}")
lm = configure_dspy_with_ollama(model_name=ollama_model, temperature=temperature)

if not lm:
    logger.error("Failed to configure DSPy with Ollama")
    sys.exit(1)

logger.info("Successfully configured DSPy with Ollama")

# Define evaluation signature for LLM-as-Judge approach
class EvaluateL2SummaryQualitySignature(dspy.Signature):
    """Evaluate the quality of a generated L2 summary."""
    
    l1_summaries_context = dspy.InputField(desc="The original L1 summaries that were synthesized")
    agent_role = dspy.InputField(desc="The role of the agent")
    overall_mood_trend = dspy.InputField(desc="The overall mood trend of the agent", required=False)
    agent_goals = dspy.InputField(desc="The agent's goals", required=False)
    gold_summary = dspy.InputField(desc="The gold standard L2 summary (human-written or ideal summary)")
    generated_summary = dspy.InputField(desc="The machine-generated L2 summary to evaluate")
    
    quality_score = dspy.OutputField(desc="A score from 1-5 rating the quality of the generated summary (5 is best)")
    feedback = dspy.OutputField(desc="Detailed feedback explaining the rating, considering comprehensiveness, relevance, and coherence")

# Create the summary quality evaluator
l2_summary_evaluator = dspy.Predict(EvaluateL2SummaryQualitySignature)

def prepare_examples():
    """
    Prepare the example data, splitting into training and validation sets.
    """
    # Collect all examples
    examples = [example1, example2, example3]
    
    # Use the last example for validation, rest for training
    trainset = examples[:-1]
    devset = examples[-1:]
    
    logger.info(f"Prepared {len(trainset)} examples for training and {len(devset)} for validation")
    
    # Ensure examples are correctly formatted with inputs specified
    trainset = [ex.with_inputs('agent_role', 'l1_summaries_context', 'overall_mood_trend', 'agent_goals') for ex in trainset]
    devset = [ex.with_inputs('agent_role', 'l1_summaries_context', 'overall_mood_trend', 'agent_goals') for ex in devset]
    
    return trainset, devset

def l2_summary_quality_metric(gold_example, prediction, trace=None):
    """
    Evaluate the quality of a generated L2 summary using LLM-as-Judge approach.
    
    Args:
        gold_example: Example containing the ideal summary
        prediction: The predicted example with generated summary
        trace: Optional trace object
        
    Returns:
        float: Quality score normalized to 0-1 range
    """
    try:
        # Get the original context, gold summary, and generated summary
        l1_summaries_context = gold_example.l1_summaries_context
        gold_summary = gold_example.l2_summary
        generated_summary = prediction.l2_summary
        agent_role = gold_example.agent_role
        overall_mood_trend = gold_example.overall_mood_trend if hasattr(gold_example, 'overall_mood_trend') else None
        agent_goals = gold_example.agent_goals if hasattr(gold_example, 'agent_goals') else None
        
        # Use the LLM to evaluate the summary quality
        evaluation = l2_summary_evaluator(
            l1_summaries_context=l1_summaries_context,
            agent_role=agent_role,
            overall_mood_trend=overall_mood_trend,
            agent_goals=agent_goals,
            gold_summary=gold_summary,
            generated_summary=generated_summary
        )
        
        # Extract and convert the quality score
        try:
            # First, clean the quality_score to extract just the number
            score_text = evaluation.quality_score.strip()
            # Use regex to extract the first number found in the string
            score_match = re.search(r'(\d+(\.\d+)?)', score_text)
            if score_match:
                quality_score = float(score_match.group(1))
            else:
                logger.error(f"Could not find a numeric score in: {score_text}")
                return 0.0
                
            # Ensure score is between 1-5
            quality_score = max(1.0, min(5.0, quality_score))
            # Normalize to 0-1 range
            normalized_score = (quality_score - 1) / 4.0
            
            logger.info(f"L2 Summary quality metric: {quality_score}/5 = {normalized_score:.2f}")
            logger.info(f"Feedback: {evaluation.feedback}")
            
            return normalized_score
        except ValueError:
            logger.error(f"Failed to parse quality score: {evaluation.quality_score}")
            return 0.0
    except Exception as e:
        logger.error(f"Error in quality metric: {e}")
        return 0.0

def optimize_l2_summarizer():
    """
    Main function to optimize the L2 summarizer using BootstrapFewShot.
    """
    # Prepare examples
    trainset, devset = prepare_examples()
    
    # Create the base L2 generator
    base_l2_generator = L2SummaryGenerator()
    base_predictor = base_l2_generator.l2_predictor
    
    logger.info("Evaluating base model before optimization")
    # Evaluate the base model on devset
    base_scores = []
    for example in devset:
        prediction = base_predictor(
            agent_role=example.agent_role,
            l1_summaries_context=example.l1_summaries_context,
            overall_mood_trend=example.overall_mood_trend if hasattr(example, 'overall_mood_trend') else None,
            agent_goals=example.agent_goals if hasattr(example, 'agent_goals') else None
        )
        score = l2_summary_quality_metric(example, prediction)
        base_scores.append(score)
    
    avg_base_score = sum(base_scores) / len(base_scores) if base_scores else 0
    logger.info(f"Base model average score: {avg_base_score:.4f}")
    
    # Configure the teleprompter
    logger.info("Initializing BootstrapFewShot teleprompter")
    teleprompter = BootstrapFewShot(
        metric=l2_summary_quality_metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=2
    )
    
    # Optimize the predictor
    logger.info("Starting optimization with BootstrapFewShot")
    try:
        optimized_predictor = teleprompter.compile(
            student=base_predictor,
            trainset=trainset
        )
        
        logger.info("Successfully compiled optimized predictor")
        
        # Evaluate the optimized model
        optimized_scores = []
        for example in devset:
            prediction = optimized_predictor(
                agent_role=example.agent_role,
                l1_summaries_context=example.l1_summaries_context,
                overall_mood_trend=example.overall_mood_trend if hasattr(example, 'overall_mood_trend') else None,
                agent_goals=example.agent_goals if hasattr(example, 'agent_goals') else None
            )
            score = l2_summary_quality_metric(example, prediction)
            optimized_scores.append(score)
        
        avg_optimized_score = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0
        logger.info(f"Optimized model average score: {avg_optimized_score:.4f}")
        
        # Save the optimized model if it improved
        if avg_optimized_score > avg_base_score:
            # Create the compiled directory if it doesn't exist
            compiled_dir = os.path.join(project_root, "src", "agents", "dspy_programs", "compiled")
            os.makedirs(compiled_dir, exist_ok=True)
            
            # Save the optimized model
            compiled_path = os.path.join(compiled_dir, "optimized_l2_summarizer.json")
            optimized_predictor.save(compiled_path)
            logger.info(f"Saved optimized L2 summarizer to {compiled_path}")
            
            # Also save to experiments directory for reference
            exp_compiled_path = os.path.join(project_root, "experiments", "compiled_models", "optimized_l2_summarizer.json")
            optimized_predictor.save(exp_compiled_path)
            logger.info(f"Also saved optimized L2 summarizer to {exp_compiled_path}")
            
            # Generate a simple report
            report = {
                "base_model_score": avg_base_score,
                "optimized_model_score": avg_optimized_score,
                "improvement": avg_optimized_score - avg_base_score,
                "percent_improvement": ((avg_optimized_score - avg_base_score) / avg_base_score) * 100 if avg_base_score > 0 else "N/A",
                "optimization_method": "BootstrapFewShot",
                "model": ollama_model,
                "temperature": temperature
            }
            
            # Log the improvement percentage
            logger.info(f"L2 summarizer improved by {report['percent_improvement']}% (from {avg_base_score:.2f} to {avg_optimized_score:.2f})")
            
            report_path = os.path.join(project_root, "experiments", "l2_summarizer_optimization_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved optimization report to {report_path}")
            return True
        else:
            logger.warning("Optimized model did not improve performance, not saving")
            return False
    
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting L2 summarizer optimization")
    success = optimize_l2_summarizer()
    if success:
        logger.info("L2 summarizer optimization completed successfully")
    else:
        logger.error("L2 summarizer optimization failed or did not improve performance") 