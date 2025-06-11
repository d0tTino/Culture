# ruff: noqa: E501, ANN101
"""
DSPy L2 Summary Generator

This module provides a DSPy-based solution for generating Level 2 (L2) summaries from a series
of Level 1 (L1) summaries. These L2 summaries represent higher-level insights over a longer
time period, creating a hierarchical memory structure for agents.

The optimized version of this module (when available in compiled/optimized_l2_summarizer.json)
shows a 16.7% improvement in summary quality based on LLM-as-judge evaluations.
"""

import logging
import os
from typing import Optional

from src.infra.dspy_ollama_integration import configure_dspy_with_ollama, dspy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    configure_dspy_with_ollama(model_name="mistral:latest", temperature=0.1)
    logger.info("Successfully configured DSPy with Ollama for L2 summarization")
except ImportError as e:
    logger.error(f"Failed to import DSPy or configure Ollama integration: {e}")
    dspy = None


class GenerateL2SummarySignature(dspy.Signature):  # type: ignore[misc, no-any-unimported]
    """
    Generates a high-level L2 insight summary from a series of L1 summaries,
    considering agent role, mood trends, and goals.
    """

    agent_role = dspy.InputField(
        desc="The current role of the agent (e.g., 'Data Analyst', 'Philosopher')."
    )

    l1_summaries_context = dspy.InputField(
        desc=(
            "A compilation of recent L1 summaries that form the basis for the L2 summary. "
            "This represents a session's worth of condensed information."
        )
    )

    overall_mood_trend = dspy.InputField(
        desc=(
            "A general trend of the agent's mood over the period covered by the L1 summaries "
            "(e.g., 'improving', 'declining', 'stable optimistic')."
        ),
        required=False,
    )

    agent_goals = dspy.InputField(
        desc=(
            "A brief statement of the agent's current primary goals or objectives, "
            "to help focus the L2 summary on relevant insights."
        ),
        required=False,
    )

    l2_summary = dspy.OutputField(
        desc=(
            "A comprehensive yet concise L2 summary synthesizing key insights, themes, "
            "or progress from the provided L1 summaries, relevant to the agent's role and goals."
        )
    )


class L2SummaryGenerator:
    """
    Class for generating L2 summaries using DSPy.

    This generator uses DSPy's Predict module with the GenerateL2SummarySignature to create
    comprehensive higher-level summaries from a series of L1 summaries, providing deeper
    insights and patterns over longer time periods.
    """

    def __init__(
        self: "L2SummaryGenerator",
        compiled_program_path: Optional[
            str
        ] = "src/agents/dspy_programs/compiled/optimized_l2_summarizer.json",
    ) -> None:
        """
        Initialize the L2 summary generator with a DSPy predictor module.

        Args:
            compiled_program_path: Optional path to a compiled DSPy program
        """
        try:
            if not dspy:
                logger.error("DSPy is not available, L2SummaryGenerator will not function")
                self.l2_predictor = None
                return

            # Create the base predictor first
            self.l2_predictor = dspy.Predict(GenerateL2SummarySignature)

            # Try to load the compiled program if provided and exists
            if compiled_program_path and os.path.exists(compiled_program_path):
                try:
                    self.l2_predictor.load(compiled_program_path)
                    logger.info(
                        f"Successfully loaded compiled L2 summarizer from {compiled_program_path}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to load compiled L2 summarizer from {compiled_program_path}: {e}. "
                        "Using default predictor."
                    )
            else:
                logger.info(
                    "No compiled L2 summarizer found or path not provided. Using default predictor."
                )

        except Exception as e:
            logger.error(f"Failed to initialize L2SummaryGenerator: {e}")
            # Fallback to None, which will trigger alternative generation if DSPy fails
            self.l2_predictor = None

    def generate_summary(
        self: "L2SummaryGenerator",
        agent_role: str,
        l1_summaries_context: str,
        overall_mood_trend: Optional[str] = None,
        agent_goals: Optional[str] = None,
    ) -> str:
        """
        Generate a comprehensive L2 summary from a series of L1 summaries.

        Args:
            agent_role (str): The agent's current role
            l1_summaries_context (str): Compiled L1 summaries to be synthesized into an L2 summary
            overall_mood_trend (Optional[str]): The agent's mood trend over the period, if available
            agent_goals (Optional[str]): The agent's current goals, if available

        Returns:
            str: A comprehensive L2 summary synthesizing insights from the L1 summaries
        """
        try:
            if not self.l2_predictor or not dspy:
                # Fallback if DSPy is not available - this would need a direct LLM implementation
                logger.warning(
                    "DSPy not available for L2 summary generation - fallback not implemented"
                )
                return ""

            logger.debug(f"Generating L2 summary for agent in role: {agent_role}")
            logger.debug(f"L1 summaries context: {l1_summaries_context[:200]}...")
            logger.debug(f"Overall mood trend: {overall_mood_trend}")
            logger.debug(f"Agent goals: {agent_goals}")

            # Call the DSPy predictor with the inputs
            prediction = self.l2_predictor(
                agent_role=agent_role,
                l1_summaries_context=l1_summaries_context,
                overall_mood_trend=overall_mood_trend,
                agent_goals=agent_goals,
            )

            # Get the L2 summary from the prediction
            l2_summary = getattr(prediction, "l2_summary", "")
            # Clean up the L2 summary (remove extra newlines, strip)
            cleaned_summary = str(l2_summary).strip()
            logger.debug(f"Generated L2 summary: {cleaned_summary[:200]}...")

            return cleaned_summary

        except Exception as e:
            logger.error(f"Error generating L2 summary: {e}")
            return ""

    @staticmethod
    def get_failsafe_output(*args: object, **kwargs: object) -> str:
        return "Failsafe: No summary available due to processing error."
