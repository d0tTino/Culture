from importlib import import_module

try:
    from . import classes  # noqa: F401
except Exception:
    classes = import_module('stubs.weaviate.classes')
