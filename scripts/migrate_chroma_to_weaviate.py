import argparse
import logging
import sys

from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.agents.memory.weaviate_vector_store_manager import WeaviateVectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migrate_chroma_to_weaviate")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate agent memories from ChromaDB to Weaviate."
    )
    parser.add_argument("--chroma_dir", default="./chroma_db", help="Path to ChromaDB directory")
    parser.add_argument(
        "--weaviate_url", default="http://localhost:8080", help="Weaviate endpoint URL"
    )
    parser.add_argument(
        "--chroma_collection", default="agent_memories", help="ChromaDB collection name"
    )
    parser.add_argument("--weaviate_class", default="AgentMemory", help="Weaviate class name")
    args = parser.parse_args()

    # Initialize Chroma and Weaviate managers
    chroma = ChromaVectorStoreManager(persist_directory=args.chroma_dir)
    weaviate = WeaviateVectorStoreManager(
        url=args.weaviate_url, collection_name=args.weaviate_class
    )

    # Fetch all memories from Chroma
    logger.info(f"Fetching all memories from ChromaDB collection '{args.chroma_collection}'...")
    try:
        # Chroma: get all IDs
        all_ids = chroma.collection.get()["ids"]
        logger.info(f"Found {len(all_ids)} memories in ChromaDB.")
        batch_size = 64
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i : i + batch_size]
            batch = chroma.collection.get(
                ids=batch_ids, include=["documents", "metadatas", "embeddings"]
            )
            texts = batch["documents"]
            metadatas = batch["metadatas"]
            vectors = batch["embeddings"]
            # Ensure UUID is present in metadata for Weaviate
            for meta, id_ in zip(metadatas, batch_ids):  # type: ignore[arg-type]
                if "uuid" not in meta:
                    meta["uuid"] = id_  # type: ignore[index]  # meta is Mapping, but we need dict
            weaviate.add_memories(texts, metadatas, vectors)  # type: ignore[arg-type]
            logger.info(f"Migrated {i + len(batch_ids)}/{len(all_ids)} memories...")
        logger.info("Migration complete.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
