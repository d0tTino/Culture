"""
Pytest fixtures for use across test files.
"""

import shutil
import socket
import sys
import tempfile
import types
from collections.abc import Generator
from pathlib import Path
from typing import Callable, Optional
from unittest.mock import MagicMock, patch

try:
    import numpy as np
except Exception:
    np = types.SimpleNamespace(float_=float)  # type: ignore[assignment]

# Ensure the project root is on sys.path before importing test utilities
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

        class Request:
            async def is_disconnected(self) -> bool:  # type: ignore[override]
                return True

        class Response:  # pragma: no cover - simple stub
            pass

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


def is_ollama_running() -> Optional[bool]:
    """Check if Ollama server is running by attempting to connect to localhost:11434"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.1)  # Short timeout for quick check
            s.connect(("localhost", 11434))
        return True
    except (OSError, socket.timeout):
        return False


# Mark for tests requiring Ollama
require_ollama = pytest.mark.skipif(
    not is_ollama_running(), reason="Ollama server is not running on localhost:11434"
)


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
    # Skip mocking if test is marked with require_ollama
    if request.node.get_closest_marker("require_ollama"):
        # If marked with require_ollama but server isn't running, the test will be skipped
        # by the require_ollama mark, so we don't need to do anything here
        return

    # If test is not explicitly requiring Ollama, mock all Ollama functions
    patch_ollama_functions(monkeypatch)


@pytest.fixture(scope="session")
def chroma_test_dir(request: FixtureRequest) -> Generator[str, None, None]:
    """
    Provides a unique ChromaDB test directory for each pytest-xdist worker.
    On Linux, uses /dev/shm/chroma_tests/{worker_id}/ for tmpfs speed.
    On other OSs, falls back to a temp directory.
    Auto-deletes the directory after the session.
    """
    worker_id = getattr(request.config, "workerinput", {}).get("workerid", "master")
    if sys.platform.startswith("linux") and Path("/dev/shm").exists():
        base_dir = f"/dev/shm/chroma_tests/{worker_id}"
    else:
        base_dir = tempfile.mkdtemp(prefix=f"chroma_tests_{worker_id}_")
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    yield base_dir
    # Teardown: remove the directory after the session
    try:
        shutil.rmtree(base_dir)
    except Exception as e:
        print(f"Warning: Failed to remove Chroma test dir {base_dir}: {e}")


@pytest.fixture(autouse=True)
def ensure_dspy_predict(monkeypatch: MonkeyPatch) -> None:
    """Provide a simple dspy.Predict stub if DSPy is unavailable."""
    try:
        import dspy
    except Exception:
        return

    if hasattr(dspy, "Predict"):
        return

    class DummyPredict:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __call__(self, *args: object, **kwargs: object) -> object:
            return type("Result", (), {"intent": "CONTINUE_COLLABORATION"})()

    monkeypatch.setattr(dspy, "Predict", DummyPredict, raising=False)


@pytest.fixture(autouse=True)
def ensure_langgraph(monkeypatch: MonkeyPatch) -> None:
    """Provide minimal langgraph stubs when the package is missing."""
    if "langgraph" in sys.modules:
        return
    import types

    langgraph_mod = types.ModuleType("langgraph")
    graph_mod = types.ModuleType("langgraph.graph")
    graph_mod.StateGraph = object
    graph_mod.START = "START"
    graph_mod.END = "END"
    langgraph_mod.graph = graph_mod
    sys.modules["langgraph"] = langgraph_mod
    sys.modules["langgraph.graph"] = graph_mod
