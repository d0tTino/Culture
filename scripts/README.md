# Utility Scripts for Culture.ai

This directory contains utility scripts for managing and maintaining the Culture.ai project.

## Available Scripts

### `cleanup_temp_db.py`

Cleans up temporary ChromaDB directories that are created during tests.

Usage:
```bash
# Show directories that would be removed without removing them
python scripts/cleanup_temp_db.py --dry-run

# Actually remove the directories (will prompt for confirmation)
python scripts/cleanup_temp_db.py
```

### `query_agent_memory.py`

Queries an agent's ChromaDB memory store and synthesizes a short answer.

Usage:
```bash
PYTHONPATH=. python scripts/query_agent_memory.py \
    --agent_id <agent_id> \
    --query_file path/to/query.txt \
    [--chroma_dir ./chroma_db] [--ollama_model llama3:8b] \
    [--max_context_items 10]
```

# Culture.ai Scripts

## Code Quality & Compliance
- The codebase is **Ruff and Mypy strict-compliant** (with justified exceptions for third-party APIs).
- All obsolete/one-off scripts have been archived or removed. Only canonical cleanup and migration scripts remain.
- All scripts and helpers are type-annotated and linted.

## Linting & Type Checking
To check and auto-fix code style and type issues:

```sh
ruff check . --fix
mypy scripts --strict
```

## Workflow
- All new scripts must pass Ruff and Mypy strict mode before merging.
- Use PRs for all changes; CI will enforce code quality gates.

## Canonical Maintenance Scripts
- `final_cleanup.py`: Aggressive file/directory cleanup and archiving.
- `cleanup_temp_db.py`: Remove temporary ChromaDB directories.
- `migrate_chroma_to_weaviate.py`: Migrate agent memories from ChromaDB to Weaviate.
- `fix_test_imports.py`, `update_memory_imports.py`: Automated import path fixers.

## Onboarding
- See `/docs/coding_standards.md` and `/docs/testing.md` for full onboarding and contribution guidelines. 
