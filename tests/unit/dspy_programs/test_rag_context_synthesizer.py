"""
Unit tests for the RAGContextSynthesizer class.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest
from typing_extensions import Self

pytest.importorskip("dspy")
import dspy

if not hasattr(dspy, "Predict"):
    pytest.skip("dspy Predict not available", allow_module_level=True)

from src.agents.dspy_programs.rag_context_synthesizer import RAGContextSynthesizer


@pytest.mark.unit
@pytest.mark.dspy
class TestRAGContextSynthesizer(unittest.TestCase):
    """Test cases for the RAGContextSynthesizer class."""

    def setUp(self: Self) -> None:
        """Set up for each test case"""
        # Create a patch for the dspy configuration to prevent actual Ollama calls
        patcher = patch(
            "src.agents.dspy_programs.rag_context_synthesizer.configure_dspy_with_ollama"
        )
        self.mock_configure = patcher.start()
        self.addCleanup(patcher.stop)

        # Create the synthesizer
        self.synthesizer = RAGContextSynthesizer()

    @patch("src.agents.dspy_programs.rag_context_synthesizer.Path.exists")
    def test_initialization_without_compiled_program(self: Self, mock_exists: MagicMock) -> None:
        """Test that the class initializes correctly when no compiled program exists."""
        # Set up mock
        mock_exists.return_value = False

        # Initialize the synthesizer
        synthesizer = RAGContextSynthesizer()

        # Verify configure_dspy_with_ollama was called
        self.mock_configure.assert_called()

        # Check that we have a dspy_program attribute that's not None
        self.assertIsNotNone(synthesizer.dspy_program)

    @patch("src.agents.dspy_programs.rag_context_synthesizer.Path.exists")
    @patch("dspy.Predict.load")
    def test_initialization_with_compiled_program(
        self: Self, mock_load: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Test that the class initializes correctly when a compiled program exists."""
        # Set up mocks
        mock_exists.return_value = True

        # Initialize with a dummy path
        test_path = "dummy/path.json"
        synthesizer = RAGContextSynthesizer(compiled_program_path=test_path)

        # Verify mocks were called
        mock_exists.assert_called_with(test_path)

        # Verify synthesizer has a dspy_program
        self.assertIsNotNone(synthesizer.dspy_program)

    def test_clean_output_with_bracket_prefix(self: Self) -> None:
        """Test that _clean_output removes the ']]' prefix from text."""
        # Test cases with the bracket prefix
        test_cases = [
            "]] This is a test answer.",
            "]]This is a test answer.",
            "]]] This is a test answer with extra bracket.",
            "]]  This is a test answer with extra spaces.",
            " ]] This is a test answer with leading space.",
        ]

        for test_input in test_cases:
            cleaned = self.synthesizer._clean_output(test_input)
            self.assertTrue(cleaned.startswith("This"))
            self.assertFalse("]]" in cleaned)

    def test_clean_output_without_bracket_prefix(self: Self) -> None:
        """Test that _clean_output handles text without the prefix correctly."""
        # Test cases without the bracket prefix
        test_cases = [
            "This is a clean test answer.",
            "  This is a test answer with leading spaces.",
            "This is a test answer with a ]] in the middle.",
            "",  # Empty string
        ]

        for test_input in test_cases:
            cleaned = self.synthesizer._clean_output(test_input)
            self.assertEqual(cleaned, test_input.strip())

    def test_synthesize_success(self: Self) -> None:
        """Test the synthesize method with a successful prediction."""
        # Create a mock prediction with the known artifact
        mock_prediction = MagicMock()
        mock_prediction.synthesized_answer = "]] This is a synthesized answer."

        # Replace the real dspy_program with our mock
        self.synthesizer.dspy_program = MagicMock()
        self.synthesizer.dspy_program.return_value = mock_prediction

        # Call synthesize
        result = self.synthesizer.synthesize(context="Test context", question="Test question")

        # Check the result
        self.assertEqual(result, "This is a synthesized answer.")

        # Verify the dspy_program was called with the right parameters
        self.synthesizer.dspy_program.assert_called_with(
            context="Test context", question="Test question"
        )

    def test_synthesize_error_handling(self: Self) -> None:
        """Test that synthesize handles errors gracefully."""
        # Make the dspy_program raise an exception when called
        self.synthesizer.dspy_program = MagicMock()
        self.synthesizer.dspy_program.side_effect = Exception("Test error")

        # Call synthesize
        result = self.synthesizer.synthesize(context="Test context", question="Test question")

        # Check that we get the new failsafe message as the fallback
        self.assertEqual(result, "Failsafe: No answer available due to processing error.")


if __name__ == "__main__":
    unittest.main()
