"""
Pytest fixtures for use across test files.
"""

import json
import os
import shutil
import socket
import sys
import types
from collections.abc import Generator
from pathlib import Path
from typing import Callable, Optional, AsyncGenerator, cast
from unittest.mock import MagicMock, patch

try:
    import numpy as np
except Exception:
    np = types.SimpleNamespace(float_=float)  # type: ignore[assignment]

# Ensure the project root is on sys.path before importing test utilities
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.paths import ensure_dir, get_temp_dir  # noqa: E402
from tests.utils.dummy_chromadb import setup_dummy_chromadb  # noqa: E402

# Ensure np.float_ exists for libraries expecting NumPy <2.0
if not hasattr(np, "float_"):
    np.float_ = float  # type: ignore[attr-defined]

try:
    import dspy  # type: ignore

    if not hasattr(dspy, "LM"):

        class DummyLM:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def __call__(self, *args: object, **kwargs: object) -> str:
                return ""

        dspy.LM = DummyLM  # type: ignore[attr-defined]

    if not hasattr(dspy, "Signature"):

        class Signature:
            pass

        class InputField:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

        class OutputField:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

        dspy.Signature = Signature  # type: ignore[attr-defined]
        dspy.InputField = InputField  # type: ignore[attr-defined]
        dspy.OutputField = OutputField  # type: ignore[attr-defined]
except Exception:
    pass

import pytest  # noqa: E402
from pytest import FixtureRequest, MonkeyPatch  # noqa: E402

# Add the project root to path to allow importing src modules
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.infra.warning_filters import configure_warning_filters  # noqa: E402
from src.shared.llm_mocks import patch_ollama_functions  # noqa: E402
from tests.utils.mock_llm import MockLLM  # noqa: E402

# # Add these imports and filter # Removed for now
# import warnings
# from pydantic import PydanticDeprecatedSince20
#
# warnings.filterwarnings(
#     "error",
#     category=PydanticDeprecatedSince20,
#     message=r".*extra keys: 'required'.*") # Broadened regex


configure_warning_filters()  # Apply filters
setup_dummy_chromadb()
if "weaviate" not in sys.modules:
    weaviate_mod = types.ModuleType("weaviate")

    class Client:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    weaviate_mod.Client = Client
    classes_mod = types.ModuleType("weaviate.classes")

    # Provide query stubs for Filter and MetadataQuery used in tests
    class _Filter:
        def __init__(self, target: str, value: object) -> None:
            self.operator = types.SimpleNamespace(name="EQUAL")
            self.target = target
            self.value = value

        def __and__(self, other: "_Filter") -> "_Filter":
            return self

        @classmethod
        def by_property(cls, prop: str) -> "_FilterBuilder":
            return _FilterBuilder(prop)

    class _FilterBuilder:
        def __init__(self, prop: str) -> None:
            self.prop = prop

        def equal(self, value: object) -> _Filter:
            return _Filter(self.prop, value)

    class _MetadataQuery:
        def __init__(self, distance: bool = False) -> None:
            self.distance = distance

    query_mod = types.SimpleNamespace(Filter=_Filter, MetadataQuery=_MetadataQuery)

    # Provide a minimal config stub so code importing weaviate.classes.config works
    class _Vectorizer:
        @staticmethod
        def none() -> str:
            return "none"

    class _Configure:
        Vectorizer = _Vectorizer()

    class _DataType:
        TEXT = "text"
        NUMBER = "number"

    class _Property:
        def __init__(self, name: str, data_type: str) -> None:
            self.name = name
            self.data_type = data_type

    config_mod = types.SimpleNamespace(
        Configure=_Configure, DataType=_DataType, Property=_Property
    )
    classes_mod.config = config_mod
    classes_mod.query = query_mod
    sys.modules["weaviate.classes.config"] = config_mod
    sys.modules["weaviate.classes.query"] = query_mod
    weaviate_mod.classes = classes_mod
    sys.modules["weaviate"] = weaviate_mod
    sys.modules["weaviate.classes"] = classes_mod


if "fastapi" not in sys.modules:
    try:  # pragma: no cover - optional dependency
        import fastapi  # noqa: F401
    except ImportError:
        fastapi_mod = types.ModuleType("fastapi")

        class FastAPI:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def get(self, *args: object, **kwargs: object):
                def decorator(fn: Callable[..., object]) -> Callable[..., object]:
                    return fn

                return decorator

            def websocket(self, *args: object, **kwargs: object):
                def decorator(fn: Callable[..., object]) -> Callable[..., object]:
                    return fn

                return decorator

        class Request:
            async def is_disconnected(self) -> bool:  # type: ignore[override]
                return True

        class Response:  # pragma: no cover - simple stub
            def __init__(self, content: object = "", *args: object, **kwargs: object) -> None:
                self.body = json.dumps(content).encode("utf-8")

        from starlette.websockets import WebSocket, WebSocketDisconnect

        fastapi_mod.FastAPI = FastAPI
        fastapi_mod.Request = Request
        fastapi_mod.Response = Response
        fastapi_mod.WebSocket = WebSocket
        fastapi_mod.WebSocketDisconnect = WebSocketDisconnect
        responses_mod = types.ModuleType("fastapi.responses")
        responses_mod.JSONResponse = Response
        sys.modules["fastapi"] = fastapi_mod
        sys.modules["fastapi.responses"] = responses_mod

if "sse_starlette.sse" not in sys.modules:
    sse_mod = types.ModuleType("sse_starlette.sse")
    from starlette.responses import Response

    class EventSourceResponse(Response):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__("", *args, **kwargs)

    sse_mod.EventSourceResponse = EventSourceResponse
    pkg = types.ModuleType("sse_starlette")
    pkg.sse = sse_mod
    sys.modules["sse_starlette"] = pkg
    sys.modules["sse_starlette.sse"] = sse_mod


@pytest.fixture
def mock_llm_client() -> Generator[MagicMock, None, None]:
    """Fixture to provide a mocked LLMClient"""
    with patch("src.infra.llm_client.LLMClient") as MockLLMClient:
        mock_client = MockLLM()
        MockLLMClient.return_value = mock_client
        yield MockLLMClient


@pytest.fixture(autouse=True)
def mock_ollama_by_default(request: FixtureRequest, monkeypatch: MonkeyPatch) -> None:
    """
    Automatically mock Ollama functions unless the test is explicitly marked with require_ollama.

    This prevents 404 errors when Ollama is not running.
    """
    if request.node.get_closest_marker("require_ollama"):
        # If marked with require_ollama but server isn't running, the test will be skipped
        # by the decorator.
        return

    patch_ollama_functions(monkeypatch)


@pytest.fixture(scope="session")
def chroma_test_dir(request: FixtureRequest) -> Generator[Path, None, None]:
    """Create a temporary directory for ChromaDB tests."""
    temp_dir = get_temp_dir() / "chroma_tests"
    ensure_dir(temp_dir)

    def finalizer() -> None:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    request.addfinalizer(finalizer)
    yield temp_dir


# Ensure DSPy Predict is always available as a dummy class
@pytest.fixture(autouse=True)
def ensure_dspy_predict(monkeypatch: MonkeyPatch) -> None:
    """Ensure dspy.Predict exists, creating a dummy if needed."""
    try:
        import dspy  # type: ignore

        if not hasattr(dspy, "Predict"):

            class DummyPredict:
                def __init__(self, *args: object, **kwargs: object) -> None:
                    pass

                def __call__(self, *args: object, **kwargs: object) -> object:
                    return types.SimpleNamespace()

            monkeypatch.setattr(dspy, "Predict", DummyPredict)
    except ImportError:
        pass  # DSPy not installed, nothing to patch


@pytest.fixture(autouse=True)
def ensure_langgraph(monkeypatch: MonkeyPatch) -> None:
    """Ensure langgraph.graph.StateGraph exists, creating a dummy if needed."""
    try:
        from langgraph.graph import StateGraph  # noqa F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "langgraph.graph", MagicMock())
        monkeypatch.setitem(sys.modules, "langgraph.prebuilt", MagicMock())


@pytest.fixture(autouse=True, scope="session")
def ensure_required_env() -> None:
    """Set essential env vars if they are not already defined."""
    if "OLLAMA_API_BASE" not in os.environ:
        os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
    if "MODEL_NAME" not in os.environ:
        os.environ["MODEL_NAME"] = "mistral:latest"


@pytest.fixture(scope="session")
def ollama_running():
    import os, socket, pytest
    host, port = os.getenv("OLLAMA_HOST","127.0.0.1:11434").split(":")
    s = socket.socket()
    s.settimeout(0.5)
    if s.connect_ex((host, int(port))):
        pytest.skip("Ollama not running")
    s.close()


@pytest.fixture
def temp_file() -> Generator[Path, None, None]:
    """Provide a temporary file path that is automatically cleaned up."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / "temp_file.txt"
        yield temp_file_path
