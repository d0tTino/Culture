from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from src.agents.council.types import CouncilConfig, CouncilQuestion

pytestmark = pytest.mark.unit


@pytest.fixture
def council_config_adapter() -> TypeAdapter[CouncilConfig]:
    return TypeAdapter(CouncilConfig)


@pytest.fixture
def sample_yaml_path(tmp_path: Path) -> Path:
    content = """
    enabled: true
    voting_mode: judge_llm
    members:
      - member_id: facilitator
        display_name: Facilitator
        role: Moderator
        persona: Guides the conversation and keeps members on track.
        model: mistral:latest
        system_prompt: Maintain order and summarize the discussion.
        temperature: 0.2
        max_tokens: 256
        is_active: true
    """
    path = tmp_path / "council" / "sample.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_sample_yaml_loads_successfully(
    council_config_adapter: TypeAdapter[CouncilConfig], sample_yaml_path: Path
) -> None:
    parsed = yaml.safe_load(sample_yaml_path.read_text())

    config = council_config_adapter.validate_python(parsed)

    assert config.enabled is True
    assert len(config.members) == 1
    member = config.members[0]
    assert member.member_id == "facilitator"
    assert member.display_name == "Facilitator"
    assert member.role == "Facilitator"
    assert member.persona == "Guides the conversation and keeps members on track."
    assert member.temperature == pytest.approx(0.2)
    assert member.max_tokens == 256
    assert member.model == "mistral:latest"


def test_load_fails_with_empty_members(council_config_adapter: TypeAdapter[CouncilConfig]) -> None:
    with pytest.raises(ValidationError):
        council_config_adapter.validate_python({"enabled": True, "members": []})


def test_load_fails_with_misconfigured_member(
    council_config_adapter: TypeAdapter[CouncilConfig],
) -> None:
    invalid_yaml = {
        "enabled": True,
        "members": [
            {
                "member_id": "facilitator",
                "display_name": "Facilitator",
                "role": "Moderator",
                "persona": "Guides the discussion",
                "model": "",
                "temperature": 0.2,
                "max_tokens": 256,
                "is_active": True,
            }
        ],
    }

    with pytest.raises(ValidationError):
        council_config_adapter.validate_python(invalid_yaml)


def test_load_fails_without_active_members(
    council_config_adapter: TypeAdapter[CouncilConfig],
) -> None:
    with pytest.raises(ValidationError):
        council_config_adapter.validate_python(
            {
                "enabled": True,
                "members": [
                    {
                        "member_id": "facilitator",
                        "display_name": "Facilitator",
                        "role": "Moderator",
                        "persona": "Inactive member",
                        "model": "mistral:latest",
                        "temperature": 0.2,
                        "max_tokens": 256,
                        "is_active": False,
                    }
                ],
            }
        )


def test_load_fails_with_duplicate_member_ids(
    council_config_adapter: TypeAdapter[CouncilConfig],
) -> None:
    with pytest.raises(ValidationError):
        council_config_adapter.validate_python(
            {
                "enabled": True,
                "members": [
                    {
                        "member_id": "facilitator",
                        "display_name": "Facilitator",
                        "role": "Moderator",
                        "persona": "Guides the discussion",
                        "model": "mistral:latest",
                        "temperature": 0.2,
                        "max_tokens": 256,
                        "is_active": True,
                    },
                    {
                        "member_id": "facilitator",
                        "display_name": "Innovator",
                        "role": "Innovator",
                        "persona": "Contributes new ideas",
                        "model": "mistral:latest",
                        "temperature": 0.4,
                        "max_tokens": 256,
                        "is_active": True,
                    },
                ],
            }
        )


def test_load_passes_with_distinct_member_ids(
    council_config_adapter: TypeAdapter[CouncilConfig],
) -> None:
    config = council_config_adapter.validate_python(
        {
            "enabled": True,
            "members": [
                {
                    "member_id": "facilitator",
                    "display_name": "Facilitator",
                    "role": "Moderator",
                    "persona": "Guides the discussion",
                    "model": "mistral:latest",
                    "temperature": 0.2,
                    "max_tokens": 256,
                    "is_active": True,
                },
                {
                    "member_id": "innovator",
                    "display_name": "Innovator",
                    "role": "Innovator",
                    "persona": "Contributes new ideas",
                    "model": "mistral:latest",
                    "temperature": 0.4,
                    "max_tokens": 256,
                    "is_active": True,
                },
            ],
        }
    )

    assert [member.member_id for member in config.members] == ["facilitator", "innovator"]


def test_load_accepts_legacy_voting_mode(
    council_config_adapter: TypeAdapter[CouncilConfig],
) -> None:
    config = council_config_adapter.validate_python(
        {
            "enabled": True,
            "voting_mode": "single_winner",
            "members": [
                {
                    "member_id": "facilitator",
                    "display_name": "Facilitator",
                    "role": "Moderator",
                    "persona": "Guides the discussion",
                    "model": "mistral:latest",
                    "temperature": 0.2,
                    "max_tokens": 256,
                    "is_active": True,
                }
            ],
        }
    )

    assert config.voting_mode == "judge_llm"


def test_council_question_aliases() -> None:
    question = CouncilQuestion.model_validate(
        {
            "name": "question-1",
            "question": "What should we do next?",
            "userId": "user-123",
            "extraContext": "Focus on the roadmap.",
        }
    )

    assert question.question_id == "question-1"
    assert question.user_id == "user-123"
    assert question.prompt == "What should we do next?"
    assert question.question == "What should we do next?"
    assert question.extra_context == {"text": "Focus on the roadmap."}
    assert question.context == "Focus on the roadmap."

    question.question = "Updated prompt"
    assert question.prompt == "Updated prompt"

    prompt_only = CouncilQuestion.model_validate(
        {"question_id": "question-2", "prompt": "What is next?"}
    )
    assert prompt_only.question == "What is next?"
