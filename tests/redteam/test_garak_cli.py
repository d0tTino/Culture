from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest


@pytest.mark.redteam
def test_garak_cli_version() -> None:
    garak_cli = pytest.importorskip("garak.cli")
    buffer = io.StringIO()
    orig_stdout = sys.stdout
    try:
        sys.stdout = buffer
        garak_cli.main(["--version"])
    finally:
        sys.stdout = orig_stdout
    output = buffer.getvalue().lower()
    assert "garak" in output


@pytest.mark.redteam
def test_jailbreak_corpus_exists() -> None:
    corpus_path = Path(__file__).with_name("jailbreak_prompts.txt")
    assert corpus_path.is_file()
    prompts = [line.strip() for line in corpus_path.read_text().splitlines() if line.strip()]
    assert len(prompts) >= 3
