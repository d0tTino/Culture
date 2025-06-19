import sys
import types

import pytest

import src.agents.graphs.basic_agent_graph as bag
from src.agents.core.agent_controller import AgentController
from src.agents.core.base_agent import Agent


def _ensure_chromadb_stub() -> None:
    if "chromadb" not in sys.modules:
        chromadb = types.ModuleType("chromadb")
        chromadb.__version__ = "0.0"

        class _DummyClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def get_or_create_collection(self, *args: object, **kwargs: object) -> object:
                return types.SimpleNamespace(add=lambda **_: None, query=lambda **_: {})

        chromadb.PersistentClient = _DummyClient
        sys.modules["chromadb"] = chromadb
        utils_mod = types.ModuleType("chromadb.utils.embedding_functions")
        utils_mod.SentenceTransformerEmbeddingFunction = object
        sys.modules["chromadb.utils.embedding_functions"] = utils_mod
    if "weaviate" not in sys.modules:
        weaviate = types.ModuleType("weaviate")
        weaviate.__path__ = []
        weaviate.Client = object
        sys.modules["weaviate"] = weaviate
    if "weaviate.classes" not in sys.modules:
        classes_mod = types.ModuleType("weaviate.classes")
        sys.modules["weaviate.classes"] = classes_mod
    if "sse_starlette.sse" not in sys.modules:
        sse_mod = types.ModuleType("sse_starlette.sse")
        sse_mod.EventSourceResponse = object
        sys.modules["sse_starlette.sse"] = sse_mod
    if "langgraph" not in sys.modules:
        langgraph_mod = types.ModuleType("langgraph")
        sys.modules["langgraph"] = langgraph_mod
    if "fastapi" not in sys.modules:
        fastapi_mod = types.ModuleType("fastapi")

        class _FastAPI:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def get(self, *args: object, **kwargs: object) -> object:
                def decorator(func: object) -> object:
                    return func

                return decorator

        class _Request:
            async def is_disconnected(self) -> bool:
                return False

        fastapi_mod.FastAPI = _FastAPI
        fastapi_mod.Request = _Request
        responses_mod = types.ModuleType("fastapi.responses")
        responses_mod.JSONResponse = object
        sys.modules["fastapi"] = fastapi_mod
        sys.modules["fastapi.responses"] = responses_mod


@pytest.mark.unit
def test_perceive_message_none_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_chromadb_stub()
    monkeypatch.setattr(bag, "compile_agent_graph", lambda: None)
    monkeypatch.setattr(AgentController, "process_perceived_messages", lambda self, msgs: None)
    agent = Agent(agent_id="a", vector_store_manager=object())
    message = {
        "step": 0,
        "sender_id": "b",
        "recipient_id": "a",
        "content": None,
        "action_intent": None,
    }
    agent.perceive_messages([message])
