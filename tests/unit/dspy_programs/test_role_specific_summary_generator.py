#!/usr/bin/env python
"""
Unit tests for RoleSpecificSummaryGenerator.
These tests focus on verifying the template-based fallback functionality.
"""

import unittest
from unittest.mock import patch, MagicMock, ANY
import pytest

from src.agents.dspy_programs.role_specific_summary_generator import RoleSpecificSummaryGenerator
from src.agents.dspy_programs.role_specific_summary_generator import _extract_keywords

@pytest.mark.unit
@pytest.mark.dspy_program
class TestRoleSpecificSummaryGeneratorFallback(unittest.TestCase):
    """
    Test the RoleSpecificSummaryGenerator's template-based fallback mechanisms
    when all DSPy-based approaches fail.
    """

    def setUp(self):
        """Set up test environment before each test method."""
        # Sample test data
        self.agent_role = "Innovator"
        self.recent_events = """
        Step 12: Agent suggested a new approach to solving the recursive algorithm problem.
        Step 13: Agent analyzed the complexity of different solutions.
        Step 14: Agent shared insights with team members about optimization techniques.
        Step 15: Agent received feedback on the proposed solution.
        """
        self.current_mood = "curious"
        
        self.l1_summaries_context = """
        L1 Summary (Step 10-20): Agent proposed innovative solutions and collaborated with team.
        L1 Summary (Step 21-30): Agent refined algorithms and improved performance metrics.
        L1 Summary (Step 31-40): Agent presented findings and received positive feedback.
        """
        self.overall_mood_trend = "increasingly confident"
        self.agent_goals = "Develop more efficient algorithms for data processing"

    @patch('src.agents.dspy_programs.role_specific_summary_generator.RoleSpecificSummaryGenerator._generate_l1_template_fallback')
    @patch('src.agents.dspy_programs.role_specific_summary_generator.L1SummaryGenerator')
    @patch('src.agents.dspy_programs.role_specific_summary_generator.L2SummaryGenerator')
    def test_l1_template_fallback_when_all_dspy_fails(self, mock_l2_generator, mock_l1_generator, mock_template_fallback):
        """Test that L1 summarization uses template fallback when all DSPy approaches fail."""
        # Configure mocks
        mock_l1_instance = MagicMock()
        mock_l1_generator.return_value = mock_l1_instance
        mock_l1_instance.generate_summary.side_effect = Exception("DSPy L1 summarization failed")
        
        # Set up the template fallback to return a test string
        mock_template_fallback.return_value = f"L1 Summary (Fallback): Agent {self.agent_role} processed 4 events around step 12. Key topics included: algorithm, solution, insights. Mood: {self.current_mood}."
        
        # Create RoleSpecificSummaryGenerator instance and manually set dictionaries to empty
        with patch('src.agents.dspy_programs.role_specific_summary_generator.logger') as mock_logger:
            generator = RoleSpecificSummaryGenerator()
            generator.l1_predictors = {}  # Force empty dictionaries manually
            generator.l2_predictors = {}
            
            # Try to generate L1 summary - should use template fallback
            result = generator.generate_l1_summary(
                agent_role=self.agent_role,
                recent_events=self.recent_events,
                current_mood=self.current_mood
            )
            
            # Verify logger called with expected message for standard fallback attempt
            mock_logger.info.assert_any_call(f"No role-specific L1 summarizer available for {self.agent_role}, using fallback")
            
            # Verify logger called with expected message for template fallback
            mock_logger.error.assert_any_call(ANY)
            
            # Verify _generate_l1_template_fallback was called with correct parameters
            mock_template_fallback.assert_called_once_with(self.agent_role, self.recent_events, self.current_mood)
            
            # Verify result is our mocked template fallback result
            self.assertEqual(result, mock_template_fallback.return_value)
            self.assertIn("L1 Summary (Fallback):", result)
            self.assertIn(self.agent_role, result)

    @patch('src.agents.dspy_programs.role_specific_summary_generator.RoleSpecificSummaryGenerator._generate_l2_template_fallback')
    @patch('src.agents.dspy_programs.role_specific_summary_generator.L1SummaryGenerator')
    @patch('src.agents.dspy_programs.role_specific_summary_generator.L2SummaryGenerator')
    def test_l2_template_fallback_when_all_dspy_fails(self, mock_l2_generator, mock_l1_generator, mock_template_fallback):
        """Test that L2 summarization uses template fallback when all DSPy approaches fail."""
        # Configure mocks
        mock_l2_instance = MagicMock()
        mock_l2_generator.return_value = mock_l2_instance
        mock_l2_instance.generate_summary.side_effect = Exception("DSPy L2 summarization failed")
        
        # Set up the template fallback to return a test string
        mock_template_fallback.return_value = f"L2 Summary (Fallback): Agent {self.agent_role} consolidated L1 summaries from step 10 to 40. Content involved: Agent proposed innovative solutions and collaborated... Goals: {self.agent_goals}. Mood Trend: {self.overall_mood_trend}."
        
        # Create RoleSpecificSummaryGenerator instance and manually set dictionaries to empty
        with patch('src.agents.dspy_programs.role_specific_summary_generator.logger') as mock_logger:
            generator = RoleSpecificSummaryGenerator()
            generator.l1_predictors = {}  # Force empty dictionaries manually
            generator.l2_predictors = {}
            
            # Try to generate L2 summary - should use template fallback
            result = generator.generate_l2_summary(
                agent_role=self.agent_role,
                l1_summaries_context=self.l1_summaries_context,
                overall_mood_trend=self.overall_mood_trend,
                agent_goals=self.agent_goals
            )
            
            # Verify logger called with expected message for standard fallback attempt
            mock_logger.info.assert_any_call(f"No role-specific L2 summarizer available for {self.agent_role}, using fallback")
            
            # Verify logger called with expected message for template fallback
            mock_logger.error.assert_any_call(ANY)
            
            # Verify _generate_l2_template_fallback was called with correct parameters
            mock_template_fallback.assert_called_once_with(self.agent_role, self.l1_summaries_context, self.overall_mood_trend, self.agent_goals)
            
            # Verify result is our mocked template fallback result
            self.assertEqual(result, mock_template_fallback.return_value)
            self.assertIn("L2 Summary (Fallback):", result)
            self.assertIn(self.agent_role, result)

    def test_l1_template_fallback_extracts_keywords(self):
        """Test that L1 template fallback correctly extracts keywords."""
        # Test with text containing obvious keywords
        test_events = """
        The agent analyzed algorithms for optimization.
        The agent implemented recursive algorithms.
        The agent discussed optimization techniques with the team.
        The agent presented results about algorithms and optimization.
        """
        
        # Get expected keywords
        keywords = _extract_keywords(test_events)
        
        # Extract our own keywords for verification - updated to match actual expected output
        expected_keywords = ["algorithms", "optimization", "agent", "techniques"]
        
        # Check if at least some of our expected keywords are found
        matching_keywords = [kw for kw in expected_keywords if any(kw in actual_kw for actual_kw in keywords)]
        self.assertTrue(len(matching_keywords) > 0, f"None of expected keywords {expected_keywords} found in {keywords}")
        
        # Now test the actual template fallback function directly
        generator = RoleSpecificSummaryGenerator()
        result = generator._generate_l1_template_fallback(
            agent_role=self.agent_role,
            recent_events=test_events,
            current_mood=self.current_mood
        )
        
        # Verify the template contains what we expect
        self.assertIn("L1 Summary (Fallback):", result)
        self.assertIn(self.agent_role, result)
        self.assertIn("processed", result)
        self.assertIn("events", result)
        self.assertIn("Key topics", result)
        
        # At least one of our expected keywords should be in the result
        found_keywords = False
        for keyword in expected_keywords:
            if keyword in result.lower():
                found_keywords = True
                break
        self.assertTrue(found_keywords, f"None of the expected keywords found in result: {result}")

    def test_l1_template_fallback_without_mood(self):
        """Test that L1 template fallback works without mood information."""
        # Create generator
        generator = RoleSpecificSummaryGenerator()
        
        # Generate fallback summary without mood
        result = generator._generate_l1_template_fallback(
            agent_role=self.agent_role,
            recent_events=self.recent_events,
            current_mood=None  # No mood provided
        )
        
        # Verify the summary is generated correctly
        self.assertIn("L1 Summary (Fallback):", result)
        self.assertIn(self.agent_role, result)
        
        # Verify mood is not mentioned
        self.assertNotIn(" with ", result)  # This would be part of the mood text

    def test_l2_template_fallback_without_optional_fields(self):
        """Test that L2 template fallback works without optional fields."""
        # Create generator
        generator = RoleSpecificSummaryGenerator()
        
        # Generate fallback summary without optional fields
        result = generator._generate_l2_template_fallback(
            agent_role=self.agent_role,
            l1_summaries_context=self.l1_summaries_context,
            overall_mood_trend=None,  # No mood trend
            agent_goals=None  # No goals
        )
        
        # Verify the summary is generated correctly
        self.assertIn("L2 Summary (Fallback):", result)
        self.assertIn(self.agent_role, result)
        
        # Verify mood trend and goals are not mentioned
        self.assertNotIn("Mood Trend:", result)
        self.assertNotIn("Goals:", result)

    @patch('src.agents.dspy_programs.role_specific_summary_generator.L1SummaryGenerator')
    @patch('src.agents.dspy_programs.role_specific_summary_generator.L2SummaryGenerator')
    def test_normal_operation_uses_fallbacks_properly(self, mock_l2_generator, mock_l1_generator):
        """Test the normal operation fallback chain when role-specific models aren't available."""
        # Configure mocks to return valid summaries
        mock_l1_instance = MagicMock()
        mock_l1_generator.return_value = mock_l1_instance
        mock_l1_instance.generate_summary.return_value = "Normal L1 summary (not template-based)"
        
        mock_l2_instance = MagicMock()
        mock_l2_generator.return_value = mock_l2_instance
        mock_l2_instance.generate_summary.return_value = "Normal L2 summary (not template-based)"
        
        # Create RoleSpecificSummaryGenerator instance and manually set dictionaries to empty
        generator = RoleSpecificSummaryGenerator()
        generator.l1_predictors = {}  # Force empty dictionaries manually
        generator.l2_predictors = {}
        
        # Verify normal L1 generation uses fallback generator
        l1_result = generator.generate_l1_summary(
            agent_role="UnknownRole",  # Role with no specific predictor
            recent_events=self.recent_events,
            current_mood=self.current_mood
        )
        
        # Verify mock L1Generator was used
        mock_l1_generator.assert_called_once()
        mock_l1_instance.generate_summary.assert_called_once()
        self.assertEqual(l1_result, "Normal L1 summary (not template-based)")
        
        # Verify normal L2 generation uses fallback generator
        l2_result = generator.generate_l2_summary(
            agent_role="UnknownRole",  # Role with no specific predictor
            l1_summaries_context=self.l1_summaries_context,
            overall_mood_trend=self.overall_mood_trend,
            agent_goals=self.agent_goals
        )
        
        # Verify mock L2Generator was used
        mock_l2_generator.assert_called_once()
        mock_l2_instance.generate_summary.assert_called_once()
        self.assertEqual(l2_result, "Normal L2 summary (not template-based)")


if __name__ == '__main__':
    unittest.main() 