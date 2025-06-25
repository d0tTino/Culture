# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-07-01
### Added
- Asynchronous DSPy program manager enabling concurrent LLM calls.
- MUS-based memory pruning and configurable L2 summary aging.
- OpenTelemetry logging via `ENABLE_OTEL` and `OTEL_EXPORTER_ENDPOINT`.
- Optional Redpanda event logging controlled by `ENABLE_REDPANDA` and `REDPANDA_BROKER`.
- Weaviate vector store support (`VECTOR_STORE_BACKEND`, `WEAVIATE_URL`).
- NumPy added as a runtime dependency.

## [0.1.0] - 2025-06-20
### Added
- Initial open-source release.
- Modular agent architecture using LangGraph.
- Hierarchical memory system with ChromaDB vector store.
- Retrieval Augmented Generation to inject relevant memories.
- Shared Knowledge Board for collaborative idea exchange.
- Discord bot integration for real-time simulation output.
- Basic resource management (Influence Points and Data Units).
- Dynamic roles and project affiliation system.
