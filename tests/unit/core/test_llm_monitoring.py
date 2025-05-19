"""
Test script for verifying LLM call performance monitoring.
"""

import os
import sys
import time

import pytest

# Set up proper paths for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

from pydantic import BaseModel, ConfigDict, Field

from src.infra.llm_client import analyze_sentiment, generate_structured_output, generate_text


class LLMTestStructure(BaseModel):
    """A simple test structure for testing structured output."""

    model_config = ConfigDict(extra="forbid")
    title: str = Field(..., json_schema_extra={"description": "The title of the test"})
    content: str = Field(..., json_schema_extra={"description": "The content of the test"})
    score: float = Field(
        ..., json_schema_extra={"description": "A numerical score between 0 and 10"}
    )


@pytest.mark.unit
def test_llm_monitoring():
    """Run a series of LLM calls to verify monitoring functionality."""
    print("Starting LLM monitoring tests...")
    # Test 1: Simple text generation
    print("Test 1: Simple text generation")
    result1 = generate_text(
        prompt="Write a short paragraph about artificial intelligence.",
        model="mistral:latest",
        temperature=0.7,
    )
    print(f"Test 1 Result: {result1[:50]}...\n")
    time.sleep(1)
    # Test 2: Sentiment analysis
    print("Test 2: Sentiment analysis")
    result2 = analyze_sentiment(
        text="I really enjoyed working with this team, they are fantastic!", model="mistral:latest"
    )
    print(f"Test 2 Result: {result2}\n")
    time.sleep(1)
    # Test 3: Structured output
    print("Test 3: Structured output")
    result3 = generate_structured_output(
        prompt="Create a test structure with a title related to AI, some content, and a score of 8.5.",
        response_model=LLMTestStructure,
        model="mistral:latest",
        temperature=0.2,
    )
    print(f"Test 3 Result: {result3}\n")
    time.sleep(1)
    # Test 4: Force an error (unknown model)
    print("Test 4: Forcing an error with unknown model")
    try:
        result4 = generate_text(
            prompt="This should fail with an unknown model.",
            model="nonexistent-model:latest",
            temperature=0.7,
        )
        print(f"Test 4 Result (unexpected success): {result4}\n")
    except Exception as e:
        print(f"Test 4 Result (expected error): {e}\n")
    print("LLM monitoring tests completed.")
    print("Check logs/llm_performance.log for detailed monitoring results.")
