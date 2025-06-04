# Testing in Culture.ai

This document explains how to run, select, and optimize tests in the Culture.ai codebase.


## Installing Test Dependencies

To run the tests, install dependencies from both `requirements.txt` and `requirements-dev.txt`:
```bash
pip install -r requirements.txt -r requirements-dev.txt
```
The development requirements include `pytest-asyncio` and `requests` for asynchronous tests and HTTP utilities.

## Test Markers and Suite Structure

Tests are categorized using pytest markers:

- `unit`: Fast, self-contained tests (default selection)
- `integration`: Multi-component or external-service tests
- `dspy_program`: Tests that require DSPy/Ollama/LLM
- `slow`: Tests that take >5s or require heavy resources
- `memory`, `vector_store`, `hierarchical_memory`, `mus`, etc.: Specialized subsystems

### Default vs. Full Suite

- **Default run** (`pytest`): Runs only unit tests (excludes `slow`, `dspy_program`, `integration`)
- **Full run** (`pytest -m "slow or dspy_program or integration"`): Runs all slow, DSPy, and integration tests

## Parallelization

Culture.ai uses [pytest-xdist](https://pytest-xdist.readthedocs.io/) for parallel test execution:

- All tests are run in parallel by default (`-n auto`)
- Tests in the same file are kept on the same worker (`--dist loadscope`)
- This reduces wall-clock time from ~40 min to ≤5 min on modern CPUs

## ChromaDB Test DB Optimization (tmpfs)

- On Linux, ChromaDB test collections are stored in `/dev/shm/chroma_tests/{worker_id}/` (RAM-backed tmpfs)
- On other OSs, a temp directory is used
- Each pytest-xdist worker gets its own DB directory (no data races)
- Directories are auto-deleted after the test session

## Running Tests Locally

- **Fast unit tests only:**
  ```bash
  pytest -v
  ```
- **Full suite (all slow/integration/DSPy):**
  ```bash
  pytest -m "slow or dspy_program or integration" -v -n auto
  ```
- **Top 10 slowest tests:**
  ```bash
  pytest --durations=10 -v
  ```

## CI Workflow

- Fast unit tests run on every push/PR
- Full suite runs nightly and on main branch merges
- See `.github/workflows/tests.yml` for details

## Adding/Updating Markers

- Add `@pytest.mark.slow`, `@pytest.mark.integration`, or `@pytest.mark.dspy_program` to slow or external-service tests
- Update `pytest.ini` to register new markers
- Document new marker usage in this file

## Troubleshooting

- If you see ChromaDB errors or data races, ensure each test uses the `chroma_test_dir` fixture
- On Windows/Mac, tmpfs is not used; performance may be lower
- For DSPy/Ollama tests, ensure Ollama is running or use mocks

## Example: Marking a Slow Test

```python
import pytest

@pytest.mark.slow
def test_big_vector_store():
    ...
```

## References
- [pytest-xdist docs](https://pytest-xdist.readthedocs.io/)
- [pytest markers](https://docs.pytest.org/en/stable/example/markers.html)

## Benchmark

### Quick Unit Suite

```
pytest -q
25 passed, 4 warnings in 58.98s
```

### Full Parallel Suite (slow, dspy, integration)

```
pytest -m "slow or dspy or integration" -v -n auto --dist loadscope --durations=10
8 passed, 32 warnings in 61.73s (0:01:01)

Slowest 10 durations:
27.18s call     tests/unit/core/test_agent_state.py::test_agent_state
7.49s call     tests/integration/memory/test_memory_pruning_mus.py::TestMUSBasedMemoryPruning::test_a_l1_mus_calculation
6.63s call     tests/integration/test_hierarchical_memory_persistence.py::TestHierarchicalMemoryPersistence::test_hierarchical_memory_persistence
6.21s call     tests/integration/memory/test_memory_utility_score.py::TestMemoryUtilityScore::test_memory_utility_score_calculation
6.16s call     tests/integration/test_memory_usage_tracking.py::TestMemoryUsageTracking::test_memory_usage_tracking
1.56s call     tests/integration/memory/test_memory_pruning_mus.py::TestMUSBasedMemoryPruning::test_z_memory_deletion
1.55s call     tests/integration/memory/test_memory_pruning_mus.py::TestMUSBasedMemoryPruning::test_b_l2_mus_calculation
0.15s call     tests/integration/test_memory_usage_tracking.py::TestMemoryUsageTracking::test_retrieve_memory_ids
```

All tests pass green. Wall-clock time for full suite: **~1 minute** on Ryzen 7 workstation. 

---

# Weaviate Local Setup for Development/Testing

## Running Weaviate Locally (for Vector Store Development)

Culture.ai supports Weaviate as an alternative vector store. To run Weaviate locally for development or testing:

### 1. Docker Compose (no vectorizer, for external embeddings)

Save this as `docker-compose.yml`:

```yaml
version: '3.4'
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.30.3
    ports:
      - 8080:8080
      - 50051:50051
    restart: on-failure:0
    volumes:
      - weaviate_data:/var/lib/weaviate
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      CLUSTER_HOSTNAME: 'node1'
volumes:
  weaviate_data:
```

Start Weaviate:
```bash
docker compose up -d
```

Weaviate will be available at `http://localhost:8080`.

### 2. Python Client

Install the client:
```bash
pip install weaviate-client
```

Connect in Python:
```python
import weaviate
client = weaviate.connect_to_local()
print(client.is_ready())  # Should print True
client.close()
```

### 3. Using External Embeddings
- When defining a class (collection), set `vectorizer: 'none'` in the schema.
- When adding objects, provide your own embedding vector via the `vector` argument.

See `src/agents/memory/weaviate_vector_store_manager.py` for a reference implementation.

### 4. Switching Vector Store Backends (Chroma/Weaviate)

To use Weaviate as the vector store backend:

1. Set the following environment variables (in your shell or .env file):
   ```env
   VECTOR_STORE_BACKEND=weaviate
   WEAVIATE_URL=http://localhost:8080
   ```
2. Restart your application or test run. The agent will use WeaviateVectorStoreManager for all memory operations.

To switch back to ChromaDB, set:
   ```env
   VECTOR_STORE_BACKEND=chroma
   VECTOR_STORE_DIR=./chroma_db
   ```

### 5. Migrating Data from ChromaDB to Weaviate

A migration script is provided:

```bash
python scripts/migrate_chroma_to_weaviate.py --chroma_dir ./chroma_db --weaviate_url http://localhost:8080
```

- This will copy all memories from the ChromaDB collection to the Weaviate class, preserving UUIDs and metadata.
- You can specify collection/class names with `--chroma_collection` and `--weaviate_class` if needed.
- See `scripts/migrate_chroma_to_weaviate.py` for details. 

# Code Quality & Compliance (2025-05)

- The codebase is **Ruff and Mypy strict-compliant** (with justified exceptions for third-party APIs).
- All obsolete/one-off scripts have been archived or removed. Only canonical cleanup and migration scripts remain (see `scripts/README.md`).
- All new scripts and helpers must pass Ruff and Mypy strict mode before merging. 
