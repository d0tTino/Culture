"""
DSPy RAG Context Synthesizer

This module provides a DSPy-based solution for synthesizing concise answers from retrieved
contexts based on a question. It can load an optimized DSPy program or fall back to a basic
version.

The module handles the "]]" prefix artifact that sometimes appears in DSPy-Ollama outputs.
"""

import logging
import re
from pathlib import Path
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)
logger.info("====== IMPORTING DSPY RAG CONTEXT SYNTHESIZER MODULE ======")

try:
    from src.infra.dspy_ollama_integration import configure_dspy_with_ollama, dspy

    logger.info("Successfully imported DSPy module")
except ImportError as e:
    logger.error(f"Failed to import DSPy module: {e}")
    import traceback

    logger.error(traceback.format_exc())
    raise


# dspy lacks type hints, so Signature resolves to ``Any``.
class RAGSynthesis(dspy.Signature):
    """
    Given a query and a list of retrieved context passages, synthesize a concise and relevant
    answer or insight that addresses the query based strictly on the provided contexts.
    """

    context = dspy.InputField(desc="Relevant passages or information to synthesize from.")
    question = dspy.InputField(desc="The question or topic to guide the synthesis.")

    synthesized_answer = dspy.OutputField(
        desc=(
            "A concise answer or insight synthesized strictly from the provided contexts "
            "that directly addresses the question."
        )
    )


class FailsafeRAGContextSynthesizer:
    """
    Failsafe version of the RAGContextSynthesizer. Always returns a safe default answer.
    """

    def synthesize(self: "FailsafeRAGContextSynthesizer", context: str, question: str) -> str:
        return "Failsafe: No answer available due to processing error."


class RAGContextSynthesizer:
    """
    A class that uses DSPy to synthesize answers from context based on questions.
    Can use either an optimized/compiled DSPy program or a default one, with robust fallback logic.
    """

    def __init__(self: "RAGContextSynthesizer", compiled_program_path: Optional[str] = None):
        """
        Initialize the RAG Context Synthesizer.

        Args:
            compiled_program_path: Path to a compiled DSPy program JSON file.
                Defaults to a standard path in the compiled directory.
        """
        logger.info("Initializing RAGContextSynthesizer")
        self.failsafe = FailsafeRAGContextSynthesizer()
        if compiled_program_path is None:
            compiled_program_path = str(
                Path(__file__).resolve().parent / "compiled" / "optimized_rag_synthesizer.json"
            )
        try:
            configure_dspy_with_ollama(model_name="mistral:latest", temperature=0.1)
            logger.info("Successfully configured DSPy with Ollama")
        except Exception as e:
            logger.error(f"Error configuring DSPy with Ollama: {e}")
            logger.warning("DSPy configuration failed, synthesis may not work properly")
        try:
            self.dspy_program = self._load_program(str(compiled_program_path))
        except Exception as e:
            logger.critical(f"Failed to load RAGContextSynthesizer program: {e}. Using failsafe.")
            self.dspy_program = None

    def _load_program(self: "RAGContextSynthesizer", compiled_program_path: str) -> object:
        """
        Load a compiled DSPy program or create a default one.

        Args:
            compiled_program_path: Path to the compiled program JSON

        Returns:
            A DSPy module for RAG synthesis
        """
        default_module = dspy.Predict(RAGSynthesis)

        # Call Path.exists as a class method so tests can easily patch it
        if not Path.exists(compiled_program_path):
            logger.warning(f"Compiled program not found at {compiled_program_path}")
            logger.info("Using default (unoptimized) RAG synthesis module")
            return default_module

        try:
            logger.info(f"Loading compiled program from {compiled_program_path}")
            program = default_module
            try:
                program.load(str(compiled_program_path))
                logger.info("Successfully loaded compiled RAG synthesis program")
                return program
            except KeyError as ke:
                logger.error(f"Error loading compiled program: {ke}")
                logger.info("The compiled program format is incompatible. Using default module.")
                return default_module
        except Exception as e:
            logger.error(f"Error loading compiled program: {e}")
            logger.info("Falling back to default (unoptimized) RAG synthesis module")
            return default_module

    def _clean_output(self: "RAGContextSynthesizer", text: str) -> str:
        """
        Clean the output from DSPy, removing artifacts like the "]]" prefix.

        Args:
            text: The raw output from the DSPy program

        Returns:
            Cleaned text with artifacts removed
        """
        if not text:
            return ""

        cleaned_text = re.sub(r"^(\s*\]{2,})+\s*", "", text)

        if cleaned_text.startswith("]"):
            cleaned_text = cleaned_text[1:].lstrip()

        cleaned_text = re.sub(r"\[\[\s*##\s*completed\s*##\s*\]\]", "", cleaned_text)
        cleaned_text = cleaned_text.strip()

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
            if not callable(self.dspy_program):
                logger.warning(
                    "RAGContextSynthesizer DSPy program is not callable, using failsafe fallback."
                )
                return self.failsafe.synthesize(context, question)
            prediction = self.dspy_program(
                context=context, question=question
            )  # Justification: DSPy dynamic output, type unknown
            raw_answer = prediction.synthesized_answer
            cleaned_answer = self._clean_output(raw_answer)
            return cleaned_answer
        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            import traceback

            logger.error(traceback.format_exc())
            try:
                return self.failsafe.synthesize(context, question)
            except Exception as e2:
                logger.critical(f"FailsafeRAGContextSynthesizer also failed: {e2}.")
                return "Failsafe: No answer available due to processing error."

    @staticmethod
    def get_failsafe_output(*args: object, **kwargs: object) -> str:
        return "Failsafe: No answer available due to processing error."


def test_module() -> bool:
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
