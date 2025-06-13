import importlib.metadata as im

import pytest


@pytest.mark.unit
def test_dspy_version() -> None:
    v = im.version("dspy-ai")
    major, minor, patch, *_ = map(int, v.split("."))
    assert (
        major == 2 and minor == 6 and patch >= 24 and patch < 100
    ), f"Unsupported DSPy version: {v}"
