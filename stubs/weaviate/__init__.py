from importlib import import_module

try:
    from . import classes
except Exception:
    classes = import_module('stubs.weaviate.classes')
