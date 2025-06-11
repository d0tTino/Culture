import sys
import types


def setup_dummy_chromadb() -> None:
    """Install a lightweight stub of the ``chromadb`` package into ``sys.modules``."""

    chromadb = types.ModuleType("chromadb")
    utils = types.ModuleType("chromadb.utils")
    emb = types.ModuleType("chromadb.utils.embedding_functions")

    class SentenceTransformerEmbeddingFunction:
        def __init__(
            self, model_name: str | None = None
        ) -> None:  # pragma: no cover - simple stub
            pass

    emb.SentenceTransformerEmbeddingFunction = SentenceTransformerEmbeddingFunction
    utils.embedding_functions = emb
    chromadb.utils = utils

    class DummyCollection:
        def __init__(self) -> None:
            self.docs: list[dict[str, object]] = []

        def add(
            self, documents: list[str], metadatas: list[dict[str, object]], ids: list[str]
        ) -> None:
            for doc, meta, _id in zip(documents, metadatas, ids):
                self.docs.append({"id": _id, "metadata": meta, "document": doc})

        def get(self, where: dict | None = None, include: list[str] | None = None) -> dict:
            ids: list[str] = []
            metas: list[dict[str, object]] = []
            conditions = where.get("$and", [where]) if where else []
            for item in self.docs:
                meta = item["metadata"]
                match = True
                for cond in conditions:
                    for key, value in cond.items():
                        if meta.get(key) != value:
                            match = False
                            break
                    if not match:
                        break
                if match:
                    ids.append(item["id"])
                    metas.append(meta)
            return {"ids": ids, "metadatas": metas}

    class DummyClient:
        def __init__(self, path: str | None = None) -> None:
            self.collections: dict[str, DummyCollection] = {}

        def get_or_create_collection(
            self, name: str, embedding_function: object | None = None
        ) -> DummyCollection:
            if name not in self.collections:
                self.collections[name] = DummyCollection()
            return self.collections[name]

    chromadb.PersistentClient = DummyClient
    chromadb.__version__ = "0.0"

    exc = types.ModuleType("chromadb.exceptions")
    exc.ChromaDBException = Exception
    chromadb.exceptions = exc

    sys.modules["chromadb"] = chromadb
    sys.modules["chromadb.utils"] = utils
    sys.modules["chromadb.utils.embedding_functions"] = emb
    sys.modules["chromadb.exceptions"] = exc
