"""
DSPy RAG Context Synthesizer

This module provides a DSPy-based solution for synthesizing concise answers from retrieved contexts
based on a question. It can load an optimized DSPy program or fall back to a basic version.

The module handles the "]]" prefix artifact that sometimes appears in DSPy-Ollama outputs.
"""

import os
import logging
import json
import re
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("====== IMPORTING DSPY RAG CONTEXT SYNTHESIZER MODULE ======")

try:
    import dspy
    from src.infra.dspy_ollama_integration import configure_dspy_with_ollama
    logger.info("Successfully imported DSPy module")
except ImportError as e:
    logger.error(f"Failed to import DSPy module: {e}")
    import traceback
    logger.error(traceback.format_exc())
    raise

# Define the DSPy Signature for RAG synthesis
class RAGSynthesis(dspy.Signature):
    """
    Given a query and a list of retrieved context passages, synthesize a concise and relevant
    answer or insight that addresses the query based strictly on the provided contexts.
    """
    context = dspy.InputField(desc="Relevant passages or information to synthesize from.")
    question = dspy.InputField(desc="The question or topic to guide the synthesis.")
    
    synthesized_answer = dspy.OutputField(desc="A concise answer or insight synthesized strictly from the provided contexts that directly addresses the question.")

class RAGContextSynthesizer:
    """
    A class that uses DSPy to synthesize answers from context based on questions.
    Can use either an optimized/compiled DSPy program or a default one.
    """
    
    def __init__(self, compiled_program_path: Optional[str] = None):
        """
        Initialize the RAG Context Synthesizer.
        
        Args:
            compiled_program_path: Path to a compiled DSPy program JSON file.
                Defaults to a standard path in the compiled directory.
        """
        logger.info("Initializing RAGContextSynthesizer")
        
        # Set default path if none provided
        if compiled_program_path is None:
            compiled_program_path = os.path.join(
                os.path.dirname(__file__),
                "compiled",
                "optimized_rag_synthesizer.json"
            )
        
        # Configure DSPy with Ollama
        try:
            configure_dspy_with_ollama(model_name="mistral:latest", temperature=0.1)
            logger.info("Successfully configured DSPy with Ollama")
        except Exception as e:
            logger.error(f"Error configuring DSPy with Ollama: {e}")
            logger.warning("DSPy configuration failed, synthesis may not work properly")
        
        # Try to load the compiled program, or use default
        self.dspy_program = self._load_program(compiled_program_path)
    
    def _load_program(self, compiled_program_path: str) -> dspy.Module:
        """
        Load a compiled DSPy program or create a default one.
        
        Args:
            compiled_program_path: Path to the compiled program JSON
            
        Returns:
            A DSPy module for RAG synthesis
        """
        # Create the default DSPy module
        default_module = dspy.Predict(RAGSynthesis)
        
        # Check if compiled program exists
        if not os.path.exists(compiled_program_path):
            logger.warning(f"Compiled program not found at {compiled_program_path}")
            logger.info("Using default (unoptimized) RAG synthesis module")
            return default_module
        
        # Try to load the compiled program
        try:
            logger.info(f"Loading compiled program from {compiled_program_path}")
            
            # Instead of direct load, we'll take a more cautious approach
            # to handle potential structure mismatches between expected format
            # and what's in the file
            program = default_module
            try:
                program.load(compiled_program_path)
                logger.info("Successfully loaded compiled RAG synthesis program")
                return program
            except KeyError as ke:
                # Handle signature format incompatibility gracefully
                logger.error(f"Error loading compiled program: {ke}")
                logger.info("The compiled program format is incompatible. Using default module.")
                return default_module
            
        except Exception as e:
            logger.error(f"Error loading compiled program: {e}")
            logger.info("Falling back to default (unoptimized) RAG synthesis module")
            return default_module
    
    def _clean_output(self, text: str) -> str:
        """
        Clean the output from DSPy, removing artifacts like the "]]" prefix.
        
        Args:
            text: The raw output from the DSPy program
            
        Returns:
            Cleaned text with artifacts removed
        """
        if not text:
            return ""
        
        # Remove the "]]" prefix if present
        # This handles variations like "]]text", "]] text", or even multiple brackets
        cleaned_text = re.sub(r'^(\s*\]{2,})+\s*', '', text)
        
        # Remove ']' prefix
        if cleaned_text.startswith(']'):
            cleaned_text = cleaned_text[1:].lstrip()
        
        # Remove "[[ ## completed ## ]]" or similar completion markers
        cleaned_text = re.sub(r'\[\[\s*##\s*completed\s*##\s*\]\]', '', cleaned_text)
        
        # Trim any leading/trailing whitespace
        cleaned_text = cleaned_text.strip()
        
        # Log if cleaning was performed
        if cleaned_text != text:
            logger.debug(f"Output cleaning performed: '{text}' -> '{cleaned_text}'")
        
        return cleaned_text
    
    def synthesize(self, context: str, question: str) -> str:
        """
        Synthesize an answer from context based on a question.
        
        Args:
            context: The retrieved context passages
            question: The question or topic to guide synthesis
            
        Returns:
            A concise, synthesized answer
        """
        logger.info(f"Synthesizing answer for question: {question[:50]}...")
        
        try:
            # Call the DSPy program
            prediction = self.dspy_program(
                context=context,
                question=question
            )
            
            # Extract the raw synthesized answer
            raw_answer = prediction.synthesized_answer
            logger.debug(f"Raw answer from DSPy: {raw_answer[:100]}...")
            
            # Clean the output and return
            cleaned_answer = self._clean_output(raw_answer)
            logger.debug(f"Cleaned answer: {cleaned_answer[:100]}...")
            
            return cleaned_answer
            
        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""  # Return empty string on error

def test_module():
    """Test the module with a simple example."""
    try:
        logger.info("Testing RAGContextSynthesizer")
        
        # Create an instance
        synthesizer = RAGContextSynthesizer()
        
        # Test context and question
        context = """
        The hierarchical memory system in the simulation has two levels.
        Level 1 consists of session summaries that consolidate short-term memories.
        Level 2 consists of chapter summaries that further consolidate Level 1 summaries.
        Chapter summaries are typically created every 10 steps.
        The memory structures are stored in ChromaDB and retrieved via RAG.
        """
        
        question = "How does the memory system work in the simulation?"
        
        # Test synthesis
        answer = synthesizer.synthesize(context, question)
        
        logger.info(f"Test synthesis result: {answer}")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    if "--test" in __import__("sys").argv:
        test_module() 