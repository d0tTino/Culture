# Culture.ai Testing Strategy

This document outlines the testing strategy for the Culture.ai project, including the use of pytest markers, optimization techniques, and best practices.

## Test Categories

Our test suite is organized into several categories:

- **Unit Tests**: Fast, isolated tests that verify the behavior of individual components.
- **Integration Tests**: Tests that verify interactions between components.

## Pytest Markers

We use pytest markers to categorize tests and make it easy to run specific subsets of the test suite. The following markers are available:

### Test Types

- `unit`: Marks tests as unit tests (fast, isolated)
- `integration`: Marks tests as integration tests (slower, may involve multiple components)

### Component Areas

- `memory`: Marks tests related to agent memory systems
- `dspy_program`: Marks tests related to DSPy programs
- `agent_graph`: Marks tests involving the BasicAgentGraph and its nodes
- `simulation`: Marks tests involving the main Simulation loop or environment
- `vector_store`: Marks tests related to vector store functionality
- `hierarchical_memory`: Marks tests for hierarchical memory systems
- `mus`: Marks tests related to Memory Utility Score (MUS)
- `core`: Marks tests related to core agent functionality

### Performance and Priority

- `fast`: Marks tests that are very quick (e.g., smoke tests)
- `slow`: Marks tests that are known to be particularly time-consuming
- `critical`: Marks tests covering the most critical functionalities
- `critical_path`: Marks tests covering essential, core functionality that must always pass

## Running Tests

### Running Tests by Marker

You can run tests using specific markers:

```bash
# Run all unit tests
python -m pytest -m unit

# Run all integration tests
python -m pytest -m integration

# Run all critical path tests (essential functionality)
python -m pytest -m critical_path

# Run memory tests that aren't slow
python -m pytest -m "memory and not slow"

# Run all fast tests
python -m pytest -m fast
```

### Running Tests from Specific Directories

You can also run tests from specific directories:

```bash
# Run tests from the unit/dspy directory
python -m pytest tests/unit/dspy/

# Run tests from multiple directories
python -m pytest tests/unit/core/ tests/unit/dspy/
```

### Parallel Test Execution

For faster test execution, you can run tests in parallel using pytest-xdist:

```bash
# Run tests in parallel with auto-detection of CPU cores
python -m pytest -n auto

# Run tests in parallel with a specific number of workers
python -m pytest -n 4
```

### Profiling Tests

To identify slow tests, you can use pytest-profiling:

```bash
# Profile tests to identify bottlenecks
python -m pytest --profile

# Generate SVG profile visualizations
python -m pytest --profile-svg
```

## Best Practices

### Adding New Tests

When adding new tests:

1. Add appropriate markers to categorize the test.
2. Ensure tests are properly isolated and don't depend on global state.
3. Use mocking for external dependencies to improve test speed and reliability.
4. If a test is slow, mark it with `@pytest.mark.slow` and add a comment explaining why.
5. If a test is critical, consider marking it with `@pytest.mark.critical_path`.

### Optimizing Slow Tests

If your test is running slowly:

1. Use profiling to identify bottlenecks.
2. Minimize setup/teardown operations.
3. Consider using fixtures with appropriate scopes.
4. Mock external services and I/O operations.
5. For database operations, use test-specific databases or in-memory options.
6. Clean up resources properly to avoid test interference.
7. If a test must be slow due to its nature, document why in a comment.

## Warning Management

The project employs warning filters to manage warnings from third-party dependencies. See `docs/warning_management.md` for details. 
