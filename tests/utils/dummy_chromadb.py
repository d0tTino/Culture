import sys
import types


def setup_dummy_chromadb() -> None:
    """Install a lightweight stub of the ``chromadb`` package into ``sys.modules``."""

    for mod in [
        "chromadb",
        "chromadb.utils",
        "chromadb.utils.embedding_functions",
        "chromadb.exceptions",
    ]:
        sys.modules.pop(mod, None)

    chromadb = types.ModuleType("chromadb")
    utils = types.ModuleType("chromadb.utils")
    emb = types.ModuleType("chromadb.utils.embedding_functions")

    class SentenceTransformerEmbeddingFunction:
        def __init__(
            self, model_name: str | None = None
        ) -> None:  # pragma: no cover - simple stub
            pass

        def __call__(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 384 for _ in texts]

    emb.SentenceTransformerEmbeddingFunction = SentenceTransformerEmbeddingFunction
    utils.embedding_functions = emb
    chromadb.utils = utils

    class DummyCollection:
        def __init__(self) -> None:
            self.docs: list[dict[str, object]] = []

        def add(
            self,
            documents: list[str],
            metadatas: list[dict[str, object]],
            ids: list[str],
            embeddings: list[list[float]] | None = None,
        ) -> None:
            for doc, meta, _id in zip(documents, metadatas, ids):
                self.docs.append({"id": _id, "metadata": meta, "document": doc})

        def get(
            self,
            where: dict | None = None,
            include: list[str] | None = None,
            ids: list[str] | None = None,
            limit: int | None = None,
        ) -> dict:
            result_ids: list[str] = []
            metas: list[dict[str, object]] = []
            docs: list[str] = []
            conditions = where.get("$and", [where]) if where else []
            items = [d for d in self.docs if ids is None or d["id"] in ids]
            for item in items:
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
                    result_ids.append(item["id"])
                    metas.append(meta)
                    docs.append(item["document"])
            result = {"ids": result_ids[:limit], "metadatas": metas[:limit]}
            docs = docs[:limit] if limit is not None else docs
            if include is None or "documents" in include:
                result["documents"] = docs
            if include is not None and "embeddings" in include:
                ids_len = len(result["ids"])
                result["embeddings"] = [[0.0] * 1 for _ in range(ids_len)]
            return result

        def query(
            self,
            query_embeddings: list[list[float]] | None = None,
            n_results: int | None = None,
            where: dict | None = None,
            include: list[str] | None = None,
        ) -> dict:
            """Return a basic query result ignoring embeddings."""
            res = self.get(where=where, include=include, limit=n_results)
            ids = res.get("ids", [])
            # Chroma's query returns nested lists
            return {
                "ids": [ids],
                "metadatas": [res.get("metadatas", [])],
                "documents": [res.get("documents", [])],
                "distances": [[0.0 for _ in ids]],
            }

        def update(self, ids: list[str], metadatas: list[dict[str, object]]) -> None:
            """Update metadata for existing documents."""
            id_to_meta = dict(zip(ids, metadatas))
            for doc in self.docs:
                if doc["id"] in id_to_meta:
                    doc["metadata"].update(id_to_meta[doc["id"]])

        def delete(self, ids: list[str] | None = None) -> None:
            if ids is None:
                self.docs.clear()
            else:
                self.docs = [d for d in self.docs if d["id"] not in ids]

        def count(self) -> int:
            return len(self.docs)

    class _DummyClient:
        def __init__(self, path: str | None = None) -> None:
            self.collections: dict[str, DummyCollection] = {}

        def get_or_create_collection(
            self, name: str, embedding_function: object | None = None
        ) -> DummyCollection:
            if name not in self.collections:
                self.collections[name] = DummyCollection()
            return self.collections[name]

    chromadb.PersistentClient = _DummyClient
    chromadb.__version__ = "0.0"

    exc = types.ModuleType("chromadb.exceptions")
    exc.ChromaDBException = Exception
    chromadb.exceptions = exc

    sys.modules["chromadb"] = chromadb
    sys.modules["chromadb.utils"] = utils
    sys.modules["chromadb.utils.embedding_functions"] = emb
    sys.modules["chromadb.exceptions"] = exc
