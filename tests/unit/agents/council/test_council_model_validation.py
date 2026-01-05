import logging

import pytest

from src.agents.council import orchestrator

pytestmark = pytest.mark.unit


def _stub_get_config(mapping: dict[str, str | None]):
    def _get_config(key: str | None = None):
        if key is None:
            return mapping
        return mapping.get(key)

    return _get_config


def test_resolve_default_model_rejects_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        orchestrator, "get_config", _stub_get_config({"DEFAULT_LLM_MODEL": "https://api.openai.com"})
    )

    with pytest.raises(
        ValueError,
        match="Remote model 'https://api.openai.com' detected for council default;",
    ):
        orchestrator._resolve_default_model(allowed_prefixes=["http://localhost"], allow_remote=False)


def test_resolve_default_model_allows_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        orchestrator,
        "get_config",
        _stub_get_config({"DEFAULT_LLM_MODEL": "mistral:latest", "LLM_API_BASE": "http://localhost"}),
    )

    resolved = orchestrator._resolve_default_model()

    assert resolved == "mistral:latest"


def test_resolve_default_model_rejects_remote_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        orchestrator, "get_config", _stub_get_config({"DEFAULT_LLM_MODEL": "openai/gpt-4o"})
    )

    with pytest.raises(
        ValueError,
        match="Remote model 'openai/gpt-4o' detected for council default;",
    ):
        orchestrator._resolve_default_model(allowed_prefixes=["http://localhost"], allow_remote=False)


def test_build_context_accepts_local_default_with_allowed_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "_get_allowed_local_model_prefixes", lambda: ["http://localhost"])
    monkeypatch.setattr(
        orchestrator,
        "get_config",
        _stub_get_config({"DEFAULT_LLM_MODEL": "http://localhost:11434/llama3"}),
    )
    monkeypatch.setattr(
        orchestrator,
        "load_council_config",
        lambda: {"members": [{"member_id": "member-1", "max_tokens": 128}]},
    )

    context = orchestrator._build_council_context()

    assert context.member_model == "http://localhost:11434/llama3"
    assert context.judge_model == "http://localhost:11434/llama3"
    assert context.config.members[0].model == "http://localhost:11434/llama3"


def test_build_context_rejects_remote_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "_get_allowed_local_model_prefixes", lambda: ["http://localhost"])
    monkeypatch.setattr(
        orchestrator,
        "get_config",
        _stub_get_config({"DEFAULT_LLM_MODEL": "https://api.remote.com/llm"}),
    )
    monkeypatch.setattr(
        orchestrator,
        "load_council_config",
        lambda: {"members": [{"member_id": "member-1", "max_tokens": 128}]},
    )

    with pytest.raises(
        ValueError,
        match="Remote model 'https://api.remote.com/llm' detected for council default;",
    ):
        orchestrator._build_council_context()


def test_member_remote_models_warn_when_allowed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    raw_config = {
        "allow_remote_models": True,
        "members": [
            {
                "member_id": "m1",
                "display_name": "Member 1",
                "model": "openai/gpt-4o",
                "temperature": 0.2,
                "max_tokens": 128,
            }
        ],
    }
    monkeypatch.setattr(orchestrator, "load_council_config", lambda: raw_config)
    monkeypatch.setattr(
        orchestrator,
        "get_config",
        _stub_get_config({
            "DEFAULT_LLM_MODEL": "mistral:latest",
            "LLM_API_BASE": "http://localhost",
        }),
    )
    monkeypatch.setattr(orchestrator, "_get_allowed_local_model_prefixes", lambda: ["http://localhost"])

    with caplog.at_level(logging.WARNING):
        context = orchestrator._build_council_context()

    assert context.allow_remote_models is True
    assert any("Remote model" in record.message for record in caplog.records)


def test_member_remote_models_rejected_when_disallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_config = {
        "members": [
            {
                "member_id": "m1",
                "display_name": "Member 1",
                "model": "https://api.example.com/model",  # Remote URL should be rejected
                "temperature": 0.2,
                "max_tokens": 128,
            }
        ],
    }
    monkeypatch.setattr(orchestrator, "load_council_config", lambda: raw_config)
    monkeypatch.setattr(
        orchestrator,
        "get_config",
        _stub_get_config({"DEFAULT_LLM_MODEL": "mistral:latest"}),
    )
    monkeypatch.setattr(orchestrator, "_get_allowed_local_model_prefixes", lambda: ["http://localhost"])

    with pytest.raises(ValueError):
        orchestrator._build_council_context()
