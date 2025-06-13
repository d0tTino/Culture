import importlib.metadata as im


def test_dspy_version():
    v = im.version("dspy-ai")
    major, minor, *_ = map(int, v.split(".")[:2])
    assert major == 2 and 24 <= minor < 7, f"Unsupported DSPy version: {v}"
