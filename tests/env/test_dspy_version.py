import importlib.metadata as im

import pytest
from packaging.version import Version


@pytest.mark.unit
def test_dspy_version() -> None:
    """Ensure the installed DSPy version is within the supported 2.6 range."""
    v = Version(im.version("dspy-ai"))
    assert Version("2.6.24") <= v < Version("2.7.0"), f"Unsupported DSPy version: {v}"
