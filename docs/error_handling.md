# Error Handling and Resilience Strategy in Culture.ai

## Overview

Culture.ai implements robust error handling and resilience mechanisms across all critical system components. This ensures the simulation is stable, debuggable, and can gracefully handle common runtime issues.

## Key Principles
- **EAFP (Easier to Ask for Forgiveness than Permission):** Prefer try-except over pre-checks.
- **Specific Exception Handling:** Catch specific exceptions (e.g., `requests.exceptions.RequestException`, `chromadb.exceptions.ChromaDBException`, `pydantic.ValidationError`, `KeyError`, `TypeError`, `AttributeError`, `FileNotFoundError`, `json.JSONDecodeError`).
- **Comprehensive Logging:** All error paths log with timestamp, module/function, error type, message, and traceback (using `exc_info=True`).
- **Graceful Degradation:** On failure, return safe defaults or error objects, and allow the simulation to continue where possible.
- **Fail Fast for Critical Config:** If essential configuration is missing, the application exits with a clear error message.
- **Retries with Backoff:** LLM client network calls use retries with exponential backoff for transient errors.
- **Global Exception Handler:** The main simulation loop is wrapped in a global exception handler to log and exit gracefully on unhandled errors.

## Module-Specific Strategies

### LLM Client (`src/infra/llm_client.py`)
- Retries LLM API/network calls (3 attempts, exponential backoff).
- Catches and logs `requests.exceptions.RequestException`, `litellm.exceptions.APIError`, `pydantic.ValidationError`, and malformed responses.
- Returns error objects or degraded state on repeated failure.

### Memory Operations (`src/agents/memory/vector_store.py`)
- Catches and logs `chromadb.exceptions.ChromaDBException`, `OSError`, `json.JSONDecodeError`, `pydantic.ValidationError`.
- Returns empty list/None on read failure; logs and continues on write failure.

### Agent Graph Execution (`src/agents/graphs/basic_agent_graph.py`)
- Catches and logs errors in node execution (`KeyError`, `TypeError`, `AttributeError`, `pydantic.ValidationError`).
- Returns error state for the agent's turn, allowing simulation to continue.

### Simulation Loop (`src/sim/simulation.py` and `src/app.py`)
- Wraps agent processing in try-except; logs and skips failed agents/steps.
- Global exception handler logs and exits gracefully on unhandled errors.

### Configuration (`src/infra/config.py`)
- Catches and logs `FileNotFoundError`, `json.JSONDecodeError`, `KeyError`.
- Exits with error if critical config is missing.

### External Interfaces (`src/interfaces/discord_bot.py`)
- Catches and logs API/network exceptions; degrades gracefully if Discord is unavailable.

## Logging Standards
- All error logs include `exc_info=True` for tracebacks.
- Logs include module, function, error type, and context variables (if safe).

## Testing and Verification
- All error handling is verified by running the full test suite (22 tests).
- Manual fault injection is used to verify logging and fallback behavior.

## References
- [Effective Error Handling in Python](https://medium.com/@divyansh9144/effective-error-handling-in-python-navigating-best-practices-and-common-pitfalls-c8f1680611c5)
- [Python Exception Handling (GeeksforGeeks)](https://www.geeksforgeeks.org/python-exception-handling/)

---

This document will be updated as new error handling patterns are introduced. 
