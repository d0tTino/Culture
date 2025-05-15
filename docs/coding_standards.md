# Culture.ai Coding Standards

## 1. Introduction

This document defines the coding standards for the Culture.ai project. Following these standards ensures consistency, readability, and maintainability across the codebase. All contributors should adhere to these guidelines for new code and when modifying existing code.

## 2. Code Formatting

### 2.1. Line Length
- Maximum line length is 99 characters.
- For docstrings and comments, limit line length to 80 characters for better readability.
- Use line continuation within parentheses, brackets, and braces for logical breaks.

### 2.2. Indentation
- Use 4 spaces for indentation, never tabs.
- Continuation lines should align with the element being continued or use a hanging indent of 4 spaces.

### 2.3. Blank Lines
- Use 2 blank lines to separate top-level functions and classes.
- Use 1 blank line to separate methods within a class.
- Use blank lines sparingly within functions to indicate logical sections.

### 2.4. Import Order
Organize imports in the following order, with a blank line between each group:
1. Standard library imports
2. Related third-party imports
3. Local application/library specific imports

Example:
```python
import os
import sys
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from langchain import LangChain

from src.infra import config
from src.agents.core.agent_state import AgentState
```

### 2.5. Quotes
- Use double quotes for docstrings (triple double quotes for multi-line).
- Use single quotes for string literals unless the string contains single quotes.
- Be consistent within a file.

### 2.6. Whitespace
- Avoid extraneous whitespace in the following situations:
  - Inside parentheses, brackets, or braces
  - Between a trailing comma and a closing parenthesis
  - Before a comma, semicolon, or colon
- Use whitespace around operators (=, +, -, etc.) following standard Python style.

## 3. Naming Conventions

### 3.1. Modules and Packages
- Use lowercase names with underscores for modules: `vector_store.py`, `agent_state.py`.
- Use lowercase names for packages: `agents`, `infra`, `utils`.

### 3.2. Classes
- Use CapWords (PascalCase) convention: `BaseAgent`, `ChromaVectorStoreManager`.
- Acronyms should be all capitalized: `LLMClient`, `RAGRetriever`.

### 3.3. Functions and Methods
- Use lowercase with underscores (snake_case): `get_agent_state()`, `retrieve_memories()`.
- Instance methods that are not intended for public use should start with a single underscore: `_calculate_mus()`.

### 3.4. Variables
- Use lowercase with underscores: `agent_id`, `memory_type`, `current_mood`.
- For constants, use uppercase with underscores: `MAX_PROJECT_MEMBERS`, `DEFAULT_MODEL`.

### 3.5. Type Variables
- Use CapWords preferably with a short, descriptive name: `T`, `AgentT`, `ResponseT`.

## 4. Docstrings

### 4.1. Format
Use Google-style docstrings for all public modules, classes, functions, and methods:

```python
def retrieve_relevant_memories(
    query_text: str, 
    limit: int = 5, 
    metadata_filters: Optional[Dict[str, Any]] = None
) -> List[Dict]:
    """
    Retrieves memories relevant to the given query text.
    
    Args:
        query_text: The text query to search for relevant memories
        limit: Maximum number of memories to retrieve
        metadata_filters: Optional filters to apply to the metadata
        
    Returns:
        A list of memory dictionaries containing content and metadata
        
    Raises:
        ConnectionError: If unable to connect to the vector store
    """
```

### 4.2. Requirements
- All public modules, classes, functions, and methods must have docstrings.
- Private methods should have docstrings when their purpose is not immediately clear.
- Include the following sections as applicable:
  - Brief one-line summary
  - Extended description (if needed)
  - Args (parameters)
  - Returns
  - Raises (exceptions)
  - Examples (for complex functions)

## 5. Type Hinting

### 5.1. Usage
- Use type hints for all function signatures.
- Use type hints for complex variables where the type is not obvious.
- Import types from the `typing` module: `List`, `Dict`, `Optional`, etc.

### 5.2. Examples
```python
def add_memory(
    content: str, 
    metadata: Dict[str, Any], 
    agent_id: str
) -> str:
    """Adds a memory to the store."""
    # Implementation
    
def get_agent_by_id(agent_id: str) -> Optional[Agent]:
    """Returns an agent by ID or None if not found."""
    # Implementation
```

### 5.3. Complex Types
- Use `TypedDict` for dictionaries with specific structures.
- Use `Union` for values that could be one of several types.
- Use `Optional[Type]` rather than `Union[Type, None]`.

Example:
```python
class AgentTurnState(TypedDict):
    """Represents the state passed into and modified by the agent's graph turn."""
    agent_state: AgentState
    perceptions: List[Dict[str, Any]]
    current_step: int
    collective_ip: Optional[float]
    collective_du: Optional[float]
```

## 6. Error Handling

### 6.1. Exceptions
- Use specific exception types rather than broad exception clauses.
- Document expected exceptions in docstrings.
- Prefer multiple `except` blocks for different exception types.

```python
try:
    result = ollama_client.chat(
        model=model,
        messages=messages,
        temperature=temperature
    )
except ConnectionError as e:
    logger.error(f"Connection error when calling Ollama API: {e}")
    return None
except ValueError as e:
    logger.error(f"Value error when parsing Ollama response: {e}")
    return None
except Exception as e:
    logger.error(f"Unexpected error when calling Ollama API: {e}")
    return None
```

### 6.2. Logging
- Use the `logging` module instead of print statements.
- Choose appropriate log levels:
  - DEBUG: Detailed information for debugging
  - INFO: Confirmation of expected operation
  - WARNING: Unexpected behavior that can be recovered from
  - ERROR: Errors that prevent operation
  - CRITICAL: Critical errors requiring immediate attention

## 7. Comments

### 7.1. Block Comments
- Use block comments to explain complex algorithms or design decisions.
- Start with a `#` followed by a space, then the comment text.
- Keep line length to 80 characters for comments.

### 7.2. Inline Comments
- Sparingly use inline comments to explain non-obvious code.
- Place inline comments on the same line as the code they describe, separated by at least two spaces.
- Start with a `#` followed by a space, then the comment text.

### 7.3. TODO Comments
- Use `# TODO: description` for code that needs further work.
- Include a brief description of what needs to be done.

## 8. Testing Practices

### 8.1. Test Structure
- Write unit tests for all non-trivial functions and methods.
- Structure tests in the `tests/` directory, mirroring the structure of the main codebase.
- Use `unittest` for test cases or a compatible framework.

### 8.2. Test Naming
- Use the pattern `test_{feature}_{scenario}` for test method names.
- Make test names descriptive of what is being tested.

```python
def test_memory_retrieval_with_filters():
    """Test that memory retrieval correctly applies metadata filters."""
    # Test implementation
    
def test_agent_turn_with_empty_perceptions():
    """Test that agent turn handles empty perceptions gracefully."""
    # Test implementation
```

### 8.3. Test Categories
- Place unit tests in `tests/unit/`.
- Place integration tests in `tests/integration/`.
- Store test data in `tests/data/`.

## 9. Logging

### 9.1. Setup
- Configure logging at the module level.
- Use a dedicated logger for each module: `logger = logging.getLogger(__name__)`.

### 9.2. Message Format
- Make log messages clear and specific.
- Include relevant variables in log messages to aid debugging.
- For errors, include exception information.

```python
logger.debug(f"Retrieved {len(memories)} memories for query: {query_text[:30]}...")
logger.error(f"Failed to connect to Ollama service on port {port}: {str(e)}")
```

### 9.3. Level Usage
- DEBUG: Detailed information useful during development
- INFO: Regular operational messages
- WARNING: Unexpected but handled situations
- ERROR: Operation-preventing errors
- CRITICAL: System-critical failures

## 10. Version Control

### 10.1. Commit Messages
- Use descriptive, imperative-mood commit messages.
- Follow the conventional commits format when possible: `type(scope): subject`
  - `feat`: New feature
  - `fix`: Bug fix
  - `docs`: Documentation changes
  - `style`: Formatting, missing semicolons, etc; no code change
  - `refactor`: Refactoring production code
  - `test`: Adding tests, refactoring tests; no production code change
  - `chore`: Updating build tasks, package manager configs, etc; no production code change

Examples:
```
feat(memory): Add MUS-based pruning for L1 memories
fix(agent): Correct resource calculation for project creation
docs: Update architecture documentation
refactor(config): Centralize configuration management
```

### 10.2. Branching
- For feature development, create branches named `feature/description`.
- For bug fixes, create branches named `fix/description`.
- For documentation, create branches named `docs/description`.

## 11. Tooling

The following tools are integrated to enforce these standards:

### 11.1. Formatting
- **Black**: Automatic code formatter that enforces a consistent style by reformatting your code
- **isort**: Import statement organizer that automatically sorts and groups imports

### 11.2. Linting
- **Flake8**: Code style enforcement that checks your code against PEP 8 guidelines
- **Mypy**: Static type checker that helps catch type-related errors before runtime

### 11.3. Testing
- **Pytest**: Test framework
- **Coverage**: Code coverage measurement

### 11.4. Running Linters and Formatters

To ensure code consistency and catch potential errors, you can run all checks and formatters locally using the provided script:

```bash
# On Linux/Mac:
./scripts/lint.sh

# On Windows:
scripts\lint.bat
```

This script will:

1. Format code using Black
2. Sort imports using isort
3. Check for PEP 8 compliance and other style issues using Flake8
4. Perform static type checking using Mypy

It's recommended to run this script before committing changes.

#### Individual Tool Usage

You can also run each tool individually for specific files or directories:

**Black** (code formatting):
```bash
black src/ tests/          # Format all files in src/ and tests/
black path/to/specific.py  # Format a specific file
```

**isort** (import sorting):
```bash
isort src/ tests/          # Sort imports in all files in src/ and tests/
isort path/to/specific.py  # Sort imports in a specific file
```

**Flake8** (style checking):
```bash
flake8 src/ tests/         # Check style in all files in src/ and tests/
flake8 path/to/specific.py # Check style in a specific file
```

**Mypy** (type checking):
```bash
mypy src/ tests/           # Check types in all files in src/ and tests/
mypy path/to/specific.py   # Check types in a specific file
```

## 12. Related Processes

### 12.1. Code Review

All significant code changes in the Culture.ai project undergo code review to ensure quality, maintain standards, and share knowledge. The code review process complements these coding standards by providing human verification of both technical correctness and adherence to project conventions.

For detailed information about our code review process, including scope, responsibilities, and best practices, see [Code Review Process](./code_review_process.md).

## 13. Conclusion

These coding standards provide a foundation for consistent, high-quality code in the Culture.ai project. As the project evolves, these standards may be refined and expanded. All team members should strive to follow these guidelines while writing clear, efficient, and maintainable code. 