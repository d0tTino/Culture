import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.infra.dspy_ollama_integration import OllamaLM


@pytest.mark.unit
def test_try_load_compiled_program_success(tmp_path: Path) -> None:
    file_path = tmp_path / "prog.json"
    file_path.write_text("{}")
    with patch("src.infra.dspy_ollama_integration.ollama.Client", MagicMock()):
        lm = OllamaLM()
        assert lm._try_load_compiled_program(str(file_path)) is True


@pytest.mark.unit
def test_try_load_compiled_program_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing_path = tmp_path / "missing.json"
    with patch("src.infra.dspy_ollama_integration.ollama.Client", MagicMock()):
        lm = OllamaLM()
        with caplog.at_level(logging.ERROR):
            result = lm._try_load_compiled_program(str(missing_path))
    assert result is False
    assert "Compiled DSPy program not found" in caplog.text
