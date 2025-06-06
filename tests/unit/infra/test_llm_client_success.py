import pytest
from pydantic import BaseModel

from src.infra import llm_client
from tests.utils.mock_llm import MockLLM


class DummyOutput(BaseModel):
    thought: str
    message_content: str | None = None
    message_recipient_id: str | None = None
    action_intent: str
    requested_role_change: str | None = None


@pytest.mark.unit
def test_generate_text_mock() -> None:
    prompt = "hello"
    expected = "hi there"
    with MockLLM({prompt: expected}):
        result = llm_client.generate_text(prompt)
        assert result == expected


@pytest.mark.unit
def test_analyze_sentiment_mock() -> None:
    text = "any text"
    with MockLLM():
        result = llm_client.analyze_sentiment(text)
        assert result == 0.0


@pytest.mark.unit
def test_generate_structured_output_mock() -> None:
    mock_structured = {
        "thought": "T",
        "message_content": "M",
        "message_recipient_id": "user",
        "action_intent": "continue_collaboration",
        "requested_role_change": None,
    }
    with MockLLM({"structured_output": mock_structured}):
        result = llm_client.generate_structured_output(
            "prompt",
            DummyOutput,
        )
        assert result == mock_structured
