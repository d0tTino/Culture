# ruff: noqa: ANN101
from typing import Any

class Signature:
    pass

class LM:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class InputField:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class OutputField:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
