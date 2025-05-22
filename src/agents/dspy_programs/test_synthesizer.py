#!/usr/bin/env python
"""
Test script for the RAGContextSynthesizer class
"""

import logging
import sys
from pathlib import Path

# Add the project root to the Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the RAGContextSynthesizer
from src.agents.dspy_programs.rag_context_synthesizer import RAGContextSynthesizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_synthesizer")


def main():
    """Test the RAGContextSynthesizer with a real example"""
    # Create the synthesizer
    logger.info("Creating RAGContextSynthesizer...")
    synthesizer = RAGContextSynthesizer()

    # Sample context and question
    context = """
    The agent decision-making process involves multiple stages: 
    1. Perception of the environment
    2. Memory retrieval using RAG
    3. Thought generation
    4. Action intent selection
    5. Message generation or action execution
    
    Each step is optimized with DSPy-trained LLM programs to ensure coherent and appropriate behavior.
    
    The thought generation process creates an internal reflection based on the agent's role,
    goals, and the current situation.
    
    Action intents are selected from a predefined list of possible actions, with the selection
    guided by the agent's thought process, role-specific behavior patterns, and current goals.
    """

    question = "How do agents make decisions in the simulation?"

    # Synthesize an answer
    logger.info("Synthesizing answer...")
    answer = synthesizer.synthesize(context, question)

    # Display the result
    logger.info("Question: %s", question)
    logger.info("Context: %s", context[:100] + "..." if len(context) > 100 else context)
    logger.info("Answer: %s", answer)

    # Test with a different example
    context2 = """
    The Knowledge Board is a central repository where agents can post ideas and information, 
    which is then perceived by other agents.
    
    Posting to the Knowledge Board costs both Influence Points and Data Units, with more 
    detailed ideas costing more resources.
    
    Agents can reference and build upon ideas from the Knowledge Board in their discussions.
    
    The Knowledge Board serves as a form of collective memory and shared context for all agents 
    in the simulation.
    """

    question2 = "What is the Knowledge Board and how is it used?"

    # Synthesize an answer for the second example
    logger.info("\nSynthesizing answer for second example...")
    answer2 = synthesizer.synthesize(context2, question2)

    # Display the result
    logger.info("Question: %s", question2)
    logger.info("Context: %s", context2[:100] + "..." if len(context2) > 100 else context2)
    logger.info("Answer: %s", answer2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
