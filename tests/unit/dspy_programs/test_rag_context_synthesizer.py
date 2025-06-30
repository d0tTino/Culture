"""
Unit tests for the RAGContextSynthesizer class.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.dspy_programs.rag_context_synthesizer import RAGContextSynthesizer

dspy = pytest.importorskip("dspy")


@pytest.fixture
def synthesizer() -> RAGContextSynthesizer:
    """Fixture to create a RAGContextSynthesizer with a patched configuration."""
    with patch("src.agents.dspy_programs.rag_context_synthesizer.configure_dspy_with_ollama"):
        return RAGContextSynthesizer()


@pytest.mark.unit
@patch("src.agents.dspy_programs.rag_context_synthesizer.Path.exists")
def test_initialization_without_compiled_program(mock_exists: MagicMock):
    """Test that the class initializes correctly when no compiled program exists."""
    mock_exists.return_value = False
    with patch(
        "src.agents.dspy_programs.rag_context_synthesizer.configure_dspy_with_ollama"
    ) as mock_configure:
        synthesizer = RAGContextSynthesizer()
        mock_configure.assert_called()
        assert synthesizer.dspy_program is not None


@pytest.mark.unit
@patch("src.agents.dspy_programs.rag_context_synthesizer.Path.exists")
@patch("dspy.Predict.load")
def test_initialization_with_compiled_program(mock_load: MagicMock, mock_exists: MagicMock):
    """Test that the class initializes correctly when a compiled program exists."""
    mock_exists.return_value = True
    test_path = "dummy/path.json"
    synthesizer = RAGContextSynthesizer(compiled_program_path=test_path)
    mock_exists.assert_called_with(test_path)
    assert synthesizer.dspy_program is not None


@pytest.mark.unit
def test_clean_output_with_bracket_prefix(synthesizer: RAGContextSynthesizer):
    """Test that _clean_output removes the ']]' prefix from text."""
    test_cases = [
        "]] This is a test answer.",
        "]]This is a test answer.",
        "]]] This is a test answer with extra bracket.",
        "]]  This is a test answer with extra spaces.",
        " ]] This is a test answer with leading space.",
    ]

    for test_input in test_cases:
        cleaned = synthesizer._clean_output(test_input)
        assert cleaned.startswith("This")
        assert "]]" not in cleaned


@pytest.mark.unit
def test_clean_output_without_bracket_prefix(synthesizer: RAGContextSynthesizer):
    """Test that _clean_output handles text without the prefix correctly."""
    test_cases = [
        "This is a clean test answer.",
        "  This is a test answer with leading spaces.",
        "This is a test answer with a ]] in the middle.",
        "",  # Empty string
    ]

    for test_input in test_cases:
        cleaned = synthesizer._clean_output(test_input)
        assert cleaned == test_input.strip()


@pytest.mark.unit
def test_synthesize_success(synthesizer: RAGContextSynthesizer):
    """Test the synthesize method with a successful prediction."""
    mock_prediction = MagicMock()
    mock_prediction.synthesized_answer = "]] This is a synthesized answer."
    synthesizer.dspy_program = MagicMock()
    synthesizer.dspy_program.return_value = mock_prediction

    result = synthesizer.synthesize(context="Test context", question="Test question")

    assert result == "This is a synthesized answer."
    synthesizer.dspy_program.assert_called_with(context="Test context", question="Test question")


@pytest.mark.unit
def test_synthesize_error_handling(synthesizer: RAGContextSynthesizer):
    """Test that synthesize handles errors gracefully."""
    synthesizer.dspy_program = MagicMock()
    synthesizer.dspy_program.side_effect = Exception("Test error")

    result = synthesizer.synthesize(context="Test context", question="Test question")

    assert result == "Failsafe: No answer available due to processing error."
