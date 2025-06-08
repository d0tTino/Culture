#!/usr/bin/env python
"""
Unit tests for DSPy-based L1 and L2 summary generators.
These tests focus on verifying the loading of compiled programs and fallback behavior.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)


class TestL1SummaryGeneratorLoading(unittest.TestCase):
    """
    Test the L1SummaryGenerator's ability to load compiled DSPy programs
    and fall back gracefully when loading fails.
    """

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_name = self.temp_dir.name

        # Sample valid output for the mocked LLM
        self.sample_summary = "This is a test summary of recent events."

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def _create_valid_l1_compiled_file(self, filename: str) -> str:
        """
        Create a valid compiled L1 summarizer JSON file.
        This represents what dspy.Module.save() would create.
        """
        compiled_data = {
            "signature_type": "GenerateL1SummarySignature",
            "demos": [
                {
                    "agent_role": "Innovator",
                    "recent_events": "Event 1: Agent discussed new ideas\nEvent 2: Agent shared a prototype",
                    "current_mood": "enthusiastic",
                    "l1_summary": "Agent enthusiastically developed and shared new prototype ideas.",
                }
            ],
            "config": {"temperature": 0.1},
            "module_type": "dspy.Predict",
        }

        filepath = os.path.join(self.temp_dir_name, filename)
        with open(filepath, "w") as f:
            json.dump(compiled_data, f)

        return filepath

    def _create_corrupted_l1_compiled_file(self, filename: str) -> str:
        """Create an invalid/corrupted JSON file."""
        filepath = os.path.join(self.temp_dir_name, filename)
        with open(filepath, "w") as f:
            f.write("{This is not valid JSON!")

        return filepath

    @patch("src.agents.dspy_programs.l1_summary_generator.L1SummaryGenerator.generate_summary")
    @patch("src.agents.dspy_programs.l1_summary_generator.L1SummaryGenerator.__init__")
    def test_successful_load_optimized_l1_program(
        self, mock_init: MagicMock, mock_generate: MagicMock
    ) -> None:
        """Test that the L1SummaryGenerator successfully loads a valid compiled program."""
        # Configure mocks
        mock_init.return_value = None  # __init__ should return None
        mock_generate.return_value = self.sample_summary

        # Create a valid compiled program file
        valid_file_path = self._create_valid_l1_compiled_file("optimized_l1_summarizer.json")

        # Import here to use the patched version
        from src.agents.dspy_programs.l1_summary_generator import L1SummaryGenerator

        # Instantiate the L1SummaryGenerator with the valid file path
        l1_generator = L1SummaryGenerator(compiled_program_path=valid_file_path)

        # Assert that L1SummaryGenerator.__init__ was called with the correct path
        mock_init.assert_called_once_with(compiled_program_path=valid_file_path)

        # Test that generate_summary works as expected
        result = l1_generator.generate_summary(
            agent_role="Innovator", recent_events="Event 1\nEvent 2", current_mood="happy"
        )

        # Verify the mock was called with the right parameters
        mock_generate.assert_called_once_with(
            agent_role="Innovator", recent_events="Event 1\nEvent 2", current_mood="happy"
        )

        # Assert that the result is the mocked summary
        self.assertEqual(result, self.sample_summary)

    @patch("src.agents.dspy_programs.l1_summary_generator.os.path.exists")
    @patch("src.agents.dspy_programs.l1_summary_generator.logger")
    def test_fallback_if_l1_program_missing(
        self, mock_logger: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Test that the L1SummaryGenerator falls back to default predictor if file is missing."""
        # Configure the mock to make file check return False (file doesn't exist)
        mock_exists.return_value = False

        # Import after mocking
        from src.agents.dspy_programs.l1_summary_generator import L1SummaryGenerator

        # Patch the actual dspy.Predict to mock its creation and usage
        with patch("src.agents.dspy_programs.l1_summary_generator.dspy.Predict") as mock_predict:
            # Configure the predictor mock
            mock_predictor = MagicMock()
            mock_predict.return_value = mock_predictor

            # Create a mock prediction result
            mock_prediction = MagicMock()
            mock_prediction.l1_summary = self.sample_summary
            mock_predictor.return_value = mock_prediction

            # Specify a non-existent file path
            non_existent_path = os.path.join(self.temp_dir_name, "non_existent_file.json")

            # Instantiate the L1SummaryGenerator with the non-existent file path
            l1_generator = L1SummaryGenerator(compiled_program_path=non_existent_path)

            # Verify that load was not called since the file doesn't exist
            mock_predictor.load.assert_not_called()

            # Verify logging indicates fallback
            mock_logger.info.assert_any_call(
                "No compiled L1 summarizer found or path not provided. Using default predictor."
            )

            # Test generate_summary with the default predictor
            result = l1_generator.generate_summary(
                agent_role="Innovator", recent_events="Event 1\nEvent 2", current_mood="happy"
            )

            # Assert that the predictor was called with the right parameters
            mock_predictor.assert_called_once_with(
                agent_role="Innovator", recent_events="Event 1\nEvent 2", current_mood="happy"
            )

            # Assert that the result is the mocked summary
            self.assertEqual(result, mock_prediction.l1_summary)

    @patch("src.agents.dspy_programs.l1_summary_generator.os.path.exists")
    @patch("src.agents.dspy_programs.l1_summary_generator.logger")
    def test_fallback_if_l1_program_corrupted(
        self, mock_logger: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Test that the L1SummaryGenerator falls back to default predictor if file is corrupted."""
        # Configure the mock to make file check return True
        mock_exists.return_value = True

        # Create a corrupted JSON file
        corrupted_file_path = self._create_corrupted_l1_compiled_file(
            "corrupted_l1_summarizer.json"
        )

        # Import after mocking
        from src.agents.dspy_programs.l1_summary_generator import L1SummaryGenerator

        # Patch the actual dspy.Predict to mock its creation and usage
        with patch("src.agents.dspy_programs.l1_summary_generator.dspy.Predict") as mock_predict:
            # Configure the predictor mock
            mock_predictor = MagicMock()
            mock_predict.return_value = mock_predictor

            # Set up the exception when trying to load
            mock_predictor.load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

            # Create a mock prediction result
            mock_prediction = MagicMock()
            mock_prediction.l1_summary = self.sample_summary
            mock_predictor.return_value = mock_prediction

            # Instantiate the L1SummaryGenerator with the corrupted file path
            l1_generator = L1SummaryGenerator(compiled_program_path=corrupted_file_path)

            # Assert that the predictor load method was called but failed
            mock_predictor.load.assert_called_once_with(corrupted_file_path)

            # Verify logging indicates error and fallback
            mock_logger.error.assert_called_once_with(
                f"Failed to load compiled L1 summarizer from {corrupted_file_path}: Invalid JSON: line 1 column 1 (char 0). Using default predictor."
            )

            # Test generate_summary with the default predictor
            result = l1_generator.generate_summary(
                agent_role="Innovator", recent_events="Event 1\nEvent 2", current_mood="happy"
            )

            # Assert that the result is the mocked summary
            self.assertEqual(result, mock_prediction.l1_summary)

    @patch("src.agents.dspy_programs.l1_summary_generator.logger")
    def test_fallback_if_l1_program_path_is_none(self, mock_logger: MagicMock) -> None:
        """Test that the L1SummaryGenerator falls back to default predictor if path is None."""
        # Import after mocking
        from src.agents.dspy_programs.l1_summary_generator import L1SummaryGenerator

        # Patch the actual dspy.Predict to mock its creation and usage
        with patch("src.agents.dspy_programs.l1_summary_generator.dspy.Predict") as mock_predict:
            # Configure the predictor mock
            mock_predictor = MagicMock()
            mock_predict.return_value = mock_predictor

            # Create a mock prediction result
            mock_prediction = MagicMock()
            mock_prediction.l1_summary = self.sample_summary
            mock_predictor.return_value = mock_prediction

            # Instantiate the L1SummaryGenerator with None path
            l1_generator = L1SummaryGenerator(compiled_program_path=None)

            # Assert that load was not called
            mock_predictor.load.assert_not_called()

            # Verify logging indicates use of default predictor
            mock_logger.info.assert_any_call(
                "No compiled L1 summarizer found or path not provided. Using default predictor."
            )

            # Test generate_summary with the default predictor
            result = l1_generator.generate_summary(
                agent_role="Innovator", recent_events="Event 1\nEvent 2", current_mood="happy"
            )

            # Assert that the result is the mocked summary
            self.assertEqual(result, mock_prediction.l1_summary)


class TestL2SummaryGeneratorLoading(unittest.TestCase):
    """
    Test the L2SummaryGenerator's ability to load compiled DSPy programs
    and fall back gracefully when loading fails.
    """

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_name = self.temp_dir.name

        # Sample valid output for the mocked LLM
        self.sample_summary = "This is a test L2 summary synthesizing multiple L1 summaries."

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def _create_valid_l2_compiled_file(self, filename: str) -> str:
        """
        Create a valid compiled L2 summarizer JSON file.
        This represents what dspy.Module.save() would create.
        """
        compiled_data = {
            "signature_type": "GenerateL2SummarySignature",
            "demos": [
                {
                    "agent_role": "Analyzer",
                    "l1_summaries_context": "L1 Summary 1: Analyzed data patterns\nL1 Summary 2: Identified key correlations",
                    "overall_mood_trend": "increasingly confident",
                    "agent_goals": "Discover meaningful patterns in complex data",
                    "l2_summary": "Over time, the agent successfully analyzed complex data patterns, identifying key correlations with increasing confidence, making progress toward the goal of discovering meaningful patterns.",
                }
            ],
            "config": {"temperature": 0.1},
            "module_type": "dspy.Predict",
        }

        filepath = os.path.join(self.temp_dir_name, filename)
        with open(filepath, "w") as f:
            json.dump(compiled_data, f)

        return filepath

    def _create_corrupted_l2_compiled_file(self, filename: str) -> str:
        """Create an invalid/corrupted JSON file."""
        filepath = os.path.join(self.temp_dir_name, filename)
        with open(filepath, "w") as f:
            f.write("{This is not valid JSON for L2 summarizer!")

        return filepath

    @patch("src.agents.dspy_programs.l2_summary_generator.L2SummaryGenerator.generate_summary")
    @patch("src.agents.dspy_programs.l2_summary_generator.L2SummaryGenerator.__init__")
    def test_successful_load_optimized_l2_program(
        self, mock_init: MagicMock, mock_generate: MagicMock
    ) -> None:
        """Test that the L2SummaryGenerator successfully loads a valid compiled program."""
        # Configure mocks
        mock_init.return_value = None  # __init__ should return None
        mock_generate.return_value = self.sample_summary

        # Create a valid compiled program file
        valid_file_path = self._create_valid_l2_compiled_file("optimized_l2_summarizer.json")

        # Import here to use the patched version
        from src.agents.dspy_programs.l2_summary_generator import L2SummaryGenerator

        # Instantiate the L2SummaryGenerator with the valid file path
        l2_generator = L2SummaryGenerator(compiled_program_path=valid_file_path)

        # Assert that L2SummaryGenerator.__init__ was called with the correct path
        mock_init.assert_called_once_with(compiled_program_path=valid_file_path)

        # Test that generate_summary works as expected
        result = l2_generator.generate_summary(
            agent_role="Analyzer",
            l1_summaries_context="L1 Summary 1\nL1 Summary 2",
            overall_mood_trend="positive",
            agent_goals="Analyze complex patterns",
        )

        # Verify the mock was called with the right parameters
        mock_generate.assert_called_once_with(
            agent_role="Analyzer",
            l1_summaries_context="L1 Summary 1\nL1 Summary 2",
            overall_mood_trend="positive",
            agent_goals="Analyze complex patterns",
        )

        # Assert that the result is the mocked summary
        self.assertEqual(result, self.sample_summary)

    @patch("src.agents.dspy_programs.l2_summary_generator.os.path.exists")
    @patch("src.agents.dspy_programs.l2_summary_generator.logger")
    def test_fallback_if_l2_program_missing(
        self, mock_logger: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Test that the L2SummaryGenerator falls back to default predictor if file is missing."""
        # Configure the mock to make file check return False (file doesn't exist)
        mock_exists.return_value = False

        # Import after mocking
        from src.agents.dspy_programs.l2_summary_generator import L2SummaryGenerator

        # Patch the actual dspy.Predict to mock its creation and usage
        with patch("src.agents.dspy_programs.l2_summary_generator.dspy.Predict") as mock_predict:
            # Configure the predictor mock
            mock_predictor = MagicMock()
            mock_predict.return_value = mock_predictor

            # Create a mock prediction result
            mock_prediction = MagicMock()
            mock_prediction.l2_summary = self.sample_summary
            mock_predictor.return_value = mock_prediction

            # Specify a non-existent file path
            non_existent_path = os.path.join(self.temp_dir_name, "non_existent_l2_file.json")

            # Instantiate the L2SummaryGenerator with the non-existent file path
            l2_generator = L2SummaryGenerator(compiled_program_path=non_existent_path)

            # Verify that load was not called since the file doesn't exist
            mock_predictor.load.assert_not_called()

            # Verify logging indicates fallback
            mock_logger.info.assert_any_call(
                "No compiled L2 summarizer found or path not provided. Using default predictor."
            )

            # Test generate_summary with the default predictor
            result = l2_generator.generate_summary(
                agent_role="Analyzer",
                l1_summaries_context="L1 Summary 1\nL1 Summary 2",
                overall_mood_trend="positive",
                agent_goals="Analyze complex patterns",
            )

            # Assert that the predictor was called with the right parameters
            mock_predictor.assert_called_once_with(
                agent_role="Analyzer",
                l1_summaries_context="L1 Summary 1\nL1 Summary 2",
                overall_mood_trend="positive",
                agent_goals="Analyze complex patterns",
            )

            # Assert that the result is the mocked summary
            self.assertEqual(result, mock_prediction.l2_summary)

    @patch("src.agents.dspy_programs.l2_summary_generator.os.path.exists")
    @patch("src.agents.dspy_programs.l2_summary_generator.logger")
    def test_fallback_if_l2_program_corrupted(
        self, mock_logger: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Test that the L2SummaryGenerator falls back to default predictor if file is corrupted."""
        # Configure the mock to make file check return True
        mock_exists.return_value = True

        # Create a corrupted JSON file
        corrupted_file_path = self._create_corrupted_l2_compiled_file(
            "corrupted_l2_summarizer.json"
        )

        # Import after mocking
        from src.agents.dspy_programs.l2_summary_generator import L2SummaryGenerator

        # Patch the actual dspy.Predict to mock its creation and usage
        with patch("src.agents.dspy_programs.l2_summary_generator.dspy.Predict") as mock_predict:
            # Configure the predictor mock
            mock_predictor = MagicMock()
            mock_predict.return_value = mock_predictor

            # Set up the exception when trying to load
            mock_predictor.load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

            # Create a mock prediction result
            mock_prediction = MagicMock()
            mock_prediction.l2_summary = self.sample_summary
            mock_predictor.return_value = mock_prediction

            # Instantiate the L2SummaryGenerator with the corrupted file path
            l2_generator = L2SummaryGenerator(compiled_program_path=corrupted_file_path)

            # Assert that the predictor load method was called but failed
            mock_predictor.load.assert_called_once_with(corrupted_file_path)

            # Verify logging indicates error and fallback
            mock_logger.error.assert_called_once_with(
                f"Failed to load compiled L2 summarizer from {corrupted_file_path}: Invalid JSON: line 1 column 1 (char 0). Using default predictor."
            )

            # Test generate_summary with the default predictor
            result = l2_generator.generate_summary(
                agent_role="Analyzer",
                l1_summaries_context="L1 Summary 1\nL1 Summary 2",
                overall_mood_trend="positive",
                agent_goals="Analyze complex patterns",
            )

            # Assert that the result is the mocked summary
            self.assertEqual(result, mock_prediction.l2_summary)

    @patch("src.agents.dspy_programs.l2_summary_generator.logger")
    def test_fallback_if_l2_program_path_is_none(self, mock_logger: MagicMock) -> None:
        """Test that the L2SummaryGenerator falls back to default predictor if path is None."""
        # Import after mocking
        from src.agents.dspy_programs.l2_summary_generator import L2SummaryGenerator

        # Patch the actual dspy.Predict to mock its creation and usage
        with patch("src.agents.dspy_programs.l2_summary_generator.dspy.Predict") as mock_predict:
            # Configure the predictor mock
            mock_predictor = MagicMock()
            mock_predict.return_value = mock_predictor

            # Create a mock prediction result
            mock_prediction = MagicMock()
            mock_prediction.l2_summary = self.sample_summary
            mock_predictor.return_value = mock_prediction

            # Instantiate the L2SummaryGenerator with None path
            l2_generator = L2SummaryGenerator(compiled_program_path=None)

            # Assert that load was not called
            mock_predictor.load.assert_not_called()

            # Verify logging indicates use of default predictor
            mock_logger.info.assert_any_call(
                "No compiled L2 summarizer found or path not provided. Using default predictor."
            )

            # Test generate_summary with the default predictor
            result = l2_generator.generate_summary(
                agent_role="Analyzer",
                l1_summaries_context="L1 Summary 1\nL1 Summary 2",
                overall_mood_trend="positive",
                agent_goals="Analyze complex patterns",
            )

            # Assert that the result is the mocked summary
            self.assertEqual(result, mock_prediction.l2_summary)


if __name__ == "__main__":
    unittest.main()
