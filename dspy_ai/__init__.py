"""shim package to expose src.dspy_ai at top level for tests."""

from importlib import import_module

# Re-export from the actual implementation in src
module = import_module("src.dspy_ai")
for attr in getattr(module, "__all__", []):
    globals()[attr] = getattr(module, attr)
__all__ = getattr(module, "__all__", [])
