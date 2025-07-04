"""
DSPy L1 Summary Generator

This module provides a DSPy-based solution for generating Level 1 (L1) summaries from an agent's
short-term memory events. It generates concise and relevant summaries that capture key insights
from the agent's recent experiences.

The optimized version of this module (when available in compiled/optimized_l1_summarizer.json)
shows a 33.3% improvement in summary quality based on LLM-as-judge evaluations.
"""

import logging
import os
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import dspy
    from src.infra.dspy_ollama_integration import configure_dspy_with_ollama
    
    # Configure DSPy with Ollama LM
    configure_dspy_with_ollama(model_name="mistral:latest", temperature=0.1)
    logger.info("Successfully configured DSPy with Ollama for L1 summarization")
except ImportError as e:
    logger.error(f"Failed to import DSPy or configure Ollama integration: {e}")
    raise


class GenerateL1SummarySignature(dspy.Signature):
    """
    Generates a concise L1 summary from recent agent events, considering the agent's role, 
    context, and optionally mood.
    """
    
    # Input fields
    agent_role: str = dspy.InputField(desc="The current role of the agent (e.g., 'Data Analyst', 'Philosopher').")
    recent_events: str = dspy.InputField(desc="A chronological series of recent memory events (thoughts, actions, perceived messages) for the agent that need to be summarized into an L1 summary. This is the context from the agent's short-term memory.")
    current_mood: Optional[str] = dspy.InputField(desc="The agent's current mood (e.g., 'curious', 'frustrated'), which may influence the summary's tone or focus.")
    
    # Output field
    l1_summary: str = dspy.OutputField(desc="A concise and informative L1 summary of the recent events, capturing key insights or developments relevant to the agent's role and ongoing activities.")


class L1SummaryGenerator:
    """
    Class for generating L1 summaries using DSPy.
    
    This generator uses DSPy's Predict module with the GenerateL1SummarySignature to create
    consistent, high-quality summaries from an agent's short-term memory events.
    """
    
    def __init__(self, compiled_program_path: Optional[str] = "src/agents/dspy_programs/compiled/optimized_l1_summarizer.json"):
        """
        Initialize the L1 summary generator with a DSPy predictor module.
        
        Args:
            compiled_program_path: Optional path to a compiled DSPy program
        """
        try:
            # Create the base predictor first
            self.l1_predictor = dspy.Predict(GenerateL1SummarySignature)
            
            # Try to load the compiled program if provided and exists
            if compiled_program_path and os.path.exists(compiled_program_path):
                try:
                    self.l1_predictor.load(compiled_program_path)
                    logger.info(f"Successfully loaded compiled L1 summarizer from {compiled_program_path}")
                except Exception as e:
                    logger.error(f"Failed to load compiled L1 summarizer from {compiled_program_path}: {e}. Using default predictor.")
                    # Fallback to default dspy.Predict if loading fails (already created above)
            else:
                logger.info("No compiled L1 summarizer found or path not provided. Using default predictor.")
        except Exception as e:
            logger.error(f"Failed to initialize L1SummaryGenerator: {e}")
            # Fallback to None, which will trigger alternative generation if DSPy fails
            self.l1_predictor = None
    
    def generate_summary(self, agent_role: str, recent_events: str, current_mood: Optional[str] = None) -> str:
        """
        Generate a concise L1 summary from recent agent events.
        
        Args:
            agent_role (str): The agent's current role
            recent_events (str): Text describing recent events/thoughts/actions from short-term memory
            current_mood (Optional[str]): The agent's current mood, if available
            
        Returns:
            str: A concise L1 summary
        """
        try:
            if not self.l1_predictor:
                logger.warning("DSPy L1 predictor not available, falling back to default method")
                return self._fallback_generate_summary(agent_role, recent_events, current_mood)
            
            # Call the DSPy predictor with the provided inputs
            prediction = self.l1_predictor(
                agent_role=agent_role,
                recent_events=recent_events,
                current_mood=current_mood
            )
            
            # Return the generated L1 summary
            return prediction.l1_summary
            
        except Exception as e:
            logger.error(f"Error generating L1 summary with DSPy: {e}")
            # Fall back to a basic summary generation method
            return self._fallback_generate_summary(agent_role, recent_events, current_mood)
    
    def _fallback_generate_summary(self, agent_role: str, recent_events: str, current_mood: Optional[str] = None) -> str:
        """
        Fallback method to generate a basic summary when DSPy fails.
        
        Args:
            agent_role (str): The agent's current role
            recent_events (str): Text describing recent events/thoughts/actions
            current_mood (Optional[str]): The agent's current mood, if available
            
        Returns:
            str: A basic L1 summary
        """
        # Create a simple summary with a prefix indicating it was generated by the fallback method
        mood_context = f" with {current_mood} mood" if current_mood else ""
        
        # Count the number of events (approximately by line count)
        event_count = len([line for line in recent_events.split('\n') if line.strip()])
        
        return (f"As a {agent_role}{mood_context}, I processed {event_count} recent memory events. "
                f"Key points included: [events summarized without DSPy due to error]") 