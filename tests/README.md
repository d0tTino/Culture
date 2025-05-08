# Tests for Culture.ai

This directory contains various tests for the Culture.ai framework.

## Directory Structure

- `integration/`: Integration tests that test multiple components working together
- `unit/`: Unit tests for individual components
- `data/`: Test data and fixtures

## Running Tests

### Integration Tests

To run the hierarchical memory persistence test:

```bash
python -m tests.integration.test_hierarchical_memory_persistence
```

To run the memory consolidation test:

```bash
python -m tests.integration.test_memory_consolidation
```

To run the level 2 memory consolidation test:

```bash
python -m tests.integration.run_level2_memory_test
```

To run the RAG (Retrieval Augmented Generation) test:

```bash
python -m tests.integration.test_rag
```

To run the resource constraints test:

```bash
python -m tests.integration.test_resource_constraints
```

To run the role-based data unit generation test:

```bash
python -m tests.integration.test_role_du_generation
```

## Test Logs

Test logs are stored in `data/logs/` directory. Each test generates its own log file. 