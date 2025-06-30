#!/usr/bin/env python
"""
Unit tests for RoleSpecificSummaryGenerator.
These tests focus on verifying the template-based fallback functionality.
"""

from unittest.mock import ANY, MagicMock, patch

import pytest

from src.agents.dspy_programs.role_specific_summary_generator import (
    RoleSpecificSummaryGenerator,
    _extract_keywords,
)
from src.agents.memory.memory_models import importância

dspy = pytest.importorskip("dspy")


@pytest.fixture
def summary_data() -> dict[str, str]:
    """Provides sample data for summary generation tests."""
    return {
        "agent_role": "Innovator",
        "recent_events": """
        Step 12: Agent suggested a new approach to solving the recursive algorithm problem.
        Step 13: Agent analyzed the complexity of different solutions.
        Step 14: Agent shared insights with team members about optimization techniques.
        Step 15: Agent received feedback on the proposed solution.
        """,
        "current_mood": "curious",
        "l1_summaries_context": """
        L1 Summary (Step 10-20): Agent proposed innovative solutions and collaborated with team.
        L1 Summary (Step 21-30): Agent refined algorithms and improved performance metrics.
        L1 Summary (Step 31-40): Agent presented findings and received positive feedback.
        """,
        "overall_mood_trend": "increasingly confident",
        "agent_goals": "Develop more efficient algorithms for data processing",
    }


@pytest.mark.unit
@patch(
    "src.agents.dspy_programs.role_specific_summary_generator.RoleSpecificSummaryGenerator._generate_l1_template_fallback"
)
@patch("src.agents.dspy_programs.role_specific_summary_generator.L1SummaryGenerator")
@patch("src.agents.dspy_programs.role_specific_summary_generator.L2SummaryGenerator")
def test_l1_template_fallback_when_all_dspy_fails(
    mock_l2_generator: MagicMock,
    mock_l1_generator: MagicMock,
    mock_template_fallback: MagicMock,
    summary_data: dict[str, str],
) -> None:
    """Test that L1 summarization uses template fallback when all DSPy approaches fail."""
    mock_l1_instance = MagicMock()
    mock_l1_generator.return_value = mock_l1_instance
    mock_l1_instance.generate_summary.side_effect = Exception("DSPy L1 summarization failed")
    mock_template_fallback.return_value = f"L1 Summary (Fallback): Agent {summary_data['agent_role']} processed 4 events. Key topics: algorithm, solution. Mood: {summary_data['current_mood']}."

    with patch("src.agents.dspy_programs.role_specific_summary_generator.logger") as mock_logger:
        generator = RoleSpecificSummaryGenerator()
        generator.l1_predictors = {}
        generator.l2_predictors = {}
        with patch.object(
            generator.fallback_l1_generator,
            "generate_summary",
            side_effect=Exception("Fallback L1 failed"),
        ):
            with patch.object(
                generator.fallback_l1_generator,
                "_fallback_generate_summary",
                side_effect=Exception("Fallback L1 fallback failed"),
            ):
                result = generator.generate_l1_summary(
                    agent_role=summary_data["agent_role"],
                    recent_events=summary_data["recent_events"],
                    current_mood=summary_data["current_mood"],
                )
                mock_logger.info.assert_any_call(
                    f"No role-specific L1 summarizer available for {summary_data['agent_role']}, using fallback"
                )
                mock_logger.error.assert_any_call(ANY)
                assert result == "Failsafe: No summary available due to processing error."


@pytest.mark.unit
@patch(
    "src.agents.dspy_programs.role_specific_summary_generator.RoleSpecificSummaryGenerator._generate_l2_template_fallback"
)
@patch("src.agents.dspy_programs.role_specific_summary_generator.L1SummaryGenerator")
@patch("src.agents.dspy_programs.role_specific_summary_generator.L2SummaryGenerator")
def test_l2_template_fallback_when_all_dspy_fails(
    mock_l2_generator: MagicMock,
    mock_l1_generator: MagicMock,
    mock_template_fallback: MagicMock,
    summary_data: dict[str, str],
) -> None:
    """Test that L2 summarization uses template fallback when all DSPy approaches fail."""
    mock_l2_instance = MagicMock()
    mock_l2_generator.return_value = mock_l2_instance
    mock_l2_instance.generate_summary.side_effect = Exception("DSPy L2 summarization failed")
    mock_template_fallback.return_value = f"L2 Summary (Fallback): Agent {summary_data['agent_role']} consolidated summaries. Goals: {summary_data['agent_goals']}. Mood Trend: {summary_data['overall_mood_trend']}."

    with patch("src.agents.dspy_programs.role_specific_summary_generator.logger") as mock_logger:
        generator = RoleSpecificSummaryGenerator()
        generator.l1_predictors = {}
        generator.l2_predictors = {}
        with patch.object(
            generator.fallback_l2_generator,
            "generate_summary",
            side_effect=Exception("Fallback L2 failed"),
        ):
            with patch.object(
                generator.fallback_l2_generator,
                "_fallback_generate_summary",
                side_effect=Exception("Fallback L2 fallback failed"),
            ):
                result = generator.generate_l2_summary(
                    agent_role=summary_data["agent_role"],
                    l1_summaries_context=summary_data["l1_summaries_context"],
                    overall_mood_trend=summary_data["overall_mood_trend"],
                    agent_goals=summary_data["agent_goals"],
                )
                mock_logger.info.assert_any_call(
                    f"No role-specific L2 summarizer available for {summary_data['agent_role']}, using fallback"
                )
                mock_logger.error.assert_any_call(ANY)
                assert result == "Failsafe: No summary available due to processing error."


@pytest.mark.unit
def test_l1_template_fallback_extracts_keywords() -> None:
    """Test that L1 template fallback correctly extracts keywords."""
    test_events = """
    The agent analyzed algorithms for optimization.
    The agent implemented recursive algorithms.
    The agent discussed optimization techniques with the team.
    The agent presented results about algorithms and optimization.
    """
    keywords = _extract_keywords(test_events)
    expected_keywords = ["algorithms", "optimization", "agent", "techniques"]
    matching_keywords = [
        kw for kw in expected_keywords if any(kw in actual_kw for actual_kw in keywords)
    ]
    assert (
        len(matching_keywords) > 0
    ), f"None of expected keywords {expected_keywords} found in {keywords}"

    generator = RoleSpecificSummaryGenerator()
    result = generator._generate_l1_template_fallback(
        agent_role="Innovator", recent_events=test_events, current_mood="curious"
    )
    assert "L1 Summary (Fallback):" in result
    assert "Innovator" in result
    assert "processed" in result
    assert "events" in result
    assert "Key topics" in result
    assert any(
        keyword in result.lower() for keyword in expected_keywords
    ), f"None of the expected keywords found in result: {result}"


@pytest.mark.unit
def test_l1_template_fallback_without_mood(summary_data: dict[str, str]) -> None:
    """Test that L1 template fallback works without mood information."""
    generator = RoleSpecificSummaryGenerator()
    result = generator._generate_l1_template_fallback(
        agent_role=summary_data["agent_role"],
        recent_events=summary_data["recent_events"],
        current_mood=None,
    )
    assert "L1 Summary (Fallback):" in result
    assert summary_data["agent_role"] in result
    assert " with " not in result


@pytest.mark.unit
def test_l2_template_fallback_without_optional_fields(summary_data: dict[str, str]) -> None:
    """Test that L2 template fallback works without optional fields."""
    generator = RoleSpecificSummaryGenerator()
    result = generator._generate_l2_template_fallback(
        agent_role=summary_data["agent_role"],
        l1_summaries_context=summary_data["l1_summaries_context"],
        overall_mood_trend=None,
        agent_goals=None,
    )
    assert "L2 Summary (Fallback):" in result
    assert summary_data["agent_role"] in result
    assert "Goals:" not in result
    assert "mood" not in result.lower()


@pytest.mark.unit
@patch("src.agents.dspy_programs.role_specific_summary_generator.L1SummaryGenerator")
@patch("src.agents.dspy_programs.role_specific_summary_generator.L2SummaryGenerator")
def test_normal_operation_uses_fallbacks_properly(
    mock_l2_generator: MagicMock,
    mock_l1_generator: MagicMock,
    summary_data: dict[str, str],
) -> None:
    """Test that the generator uses fallback predictors when role-specific ones are absent."""
    mock_l1_instance = MagicMock()
    mock_l1_instance.generate_summary.return_value = "Fallback L1 summary"
    mock_l1_generator.return_value = mock_l1_instance

    mock_l2_instance = MagicMock()
    mock_l2_instance.generate_summary.return_value = "Fallback L2 summary"
    mock_l2_generator.return_value = mock_l2_instance

    generator = RoleSpecificSummaryGenerator()
    generator.l1_predictors = {}
    generator.l2_predictors = {}

    with patch("src.agents.dspy_programs.role_specific_summary_generator.logger") as mock_logger:
        l1_result = generator.generate_l1_summary(
            agent_role="UnknownRole",
            recent_events=summary_data["recent_events"],
            current_mood=summary_data["current_mood"],
        )
        assert l1_result == "Fallback L1 summary"
        mock_logger.info.assert_any_call(
            "No role-specific L1 summarizer available for UnknownRole, using fallback"
        )

        l2_result = generator.generate_l2_summary(
            agent_role="UnknownRole",
            l1_summaries_context=summary_data["l1_summaries_context"],
            overall_mood_trend=summary_data["overall_mood_trend"],
            agent_goals=summary_data["agent_goals"],
        )
        assert l2_result == "Fallback L2 summary"
        mock_logger.info.assert_any_call(
            "No role-specific L2 summarizer available for UnknownRole, using fallback"
        )
