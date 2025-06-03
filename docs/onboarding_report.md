# Culture.ai - New Dev Lead Onboarding Report

## 1. Introduction

Welcome to the Culture.ai project! This report provides a concise overview of the current project status, architecture, and key development practices to help you get up to speed quickly.

Culture.ai is an AI Genesis Engine designed to simulate multi-agent interactions within a dynamic social environment. Autonomous agents, equipped with distinct roles (Innovator, Analyzer, Facilitator), communicate, collaborate, and evolve over time.

## 2. Project Status

The project is actively under development. Core functionalities for agent simulation, memory management, and decision-making are implemented. Recent efforts have focused on:

*   **Dynamic Role Changing:** Enabling agents to adapt their roles based on evolving simulation contexts.
*   **Advanced Memory Pruning:** Refining strategies (L1/L2 summaries, Memory Utility Score) for efficient and relevant memory retention.
*   **DSPy Integration:** Leveraging DSPy for more robust and maintainable LLM-driven logic, particularly in action selection and summary generation.
*   **Testing Framework:** Continuous improvement of unit and integration tests to ensure code quality and stability.

Key areas for upcoming development include:
*   Further refinement of conflict resolution mechanisms between agents.
*   Expansion of agent capabilities and role-specific behaviors.
*   Performance optimization for large-scale simulations.

## 3. High-Level Architecture

Culture.ai employs a modular architecture. The main components are:

*   **Agent Core (`src/agents/core/`):** Defines foundational classes for agent state (`AgentState`), basic agent structure (`BaseAgent`), and roles (`roles.py`). This system manages an agent's identity, relationships, mood, resources (Influence Points, Data Units), and project affiliations.
*   **Agent Memory System (`src/agents/memory/`):** Provides sophisticated memory capabilities.
    *   **Vector Stores:** `ChromaVectorStoreManager` and `WeaviateVectorStoreManager` handle storage and retrieval of agent memories using vector embeddings. Selection is configuration-driven.
    *   **Hierarchical Memory:** Organizes memories into Raw Memories, L1 Summaries (step-based), and L2 Summaries (chapter-based).
    *   **Memory Utility Score (MUS):** Intelligently evaluates memory importance based on retrieval frequency, relevance, and recency.
    *   **Pruning:** Employs age-based and MUS-based strategies to manage memory lifecycle and efficiency.
*   **Agent Decision Logic (`src/agents/graphs/basic_agent_graph.py`):** Implements turn-based reasoning and decision-making using LangGraph.
    *   The agent's turn is a graph with nodes for perception analysis, memory retrieval (RAG), thought generation, action selection, and state updates.
    *   DSPy is increasingly used for core logic within these nodes, such as `L1SummaryGenerator` and `action_intent_selector`.
*   **Simulation Environment (`src/sim/simulation.py`):** Manages the overall simulation, agent scheduling, and interactions with shared resources like the `KnowledgeBoard`.
*   **Infrastructure (`src/infra/`):** Contains supporting modules for configuration (`config.py`), LLM client interaction (`llm_client.py`, `dspy_ollama_integration.py`), logging (`logging_config.py`), and LLM mocking for tests (`llm_mock_helper.py`).
*   **DSPy Programs (`src/agents/dspy_programs/`):** Houses DSPy programs for various agent tasks like summary generation, thought generation, and action/intent selection. Compiled/optimized versions are often used.

## 4. Coding Standards & Practices

Adherence to coding standards is crucial for maintaining a clean and collaborative codebase. Key highlights include:

*   **Formatting:**
    *   Max line length: 99 characters (80 for docstrings/comments).
    *   Indentation: 4 spaces.
    *   Import order: Standard library, third-party, local, with blank lines between groups.
    *   Quotes: Double for docstrings, single for string literals (consistency within a file).
*   **Naming Conventions:**
    *   Modules/Packages: `lowercase_with_underscores`.
    *   Classes: `CapWords` (PascalCase).
    *   Functions/Methods/Variables: `snake_case`.
    *   Constants: `UPPERCASE_WITH_UNDERSCORES`.
*   **Docstrings:**
    *   Google-style for all public modules, classes, functions, and methods.
    *   Must include Args, Returns, and Raises sections where applicable.
*   **Type Hinting:**
    *   Mandatory for all function signatures and complex variables.
    *   Utilize types from the `typing` module.
*   **Error Handling:**
    *   Use specific exception types.
    *   Document exceptions in docstrings.
    *   Employ the `logging` module (configured per module) instead of `print` statements.
*   **Comments:**
    *   Use block comments for complex logic and `# TODO:` for pending work.
    *   Inline comments should be used sparingly.
*   **Testing:**
    *   Unit tests (`tests/unit/`) and integration tests (`tests/integration/`) are vital.
    *   Follow naming conventions like `test_{feature}_{scenario}`.
    *   `unittest` is the primary framework, with mocking utilities in `tests/utils/mock_llm.py` and `src/infra/llm_mock_helper.py`.
*   **Git & Version Control:** (Assumed standard practices, but a `code_review_process.md` exists in `docs/`)

## 5. Key Files & Directories

*   **`src/`**: Main application code.
    *   **`src/agents/`**: Core agent logic, memory systems, DSPy programs, and graph-based decision logic.
    *   **`src/infra/`**: Configuration, LLM clients, logging.
    *   **`src/sim/`**: Simulation orchestration and shared environment components.
    *   **`src/utils/`**: Utility functions.
    *   **`app.py`**: Main application entry point or orchestrator.
*   **`tests/`**: Unit and integration tests.
    *   **`tests/integration/test_dynamic_role_change.py`**: Good example of an integration test for a core feature.
*   **`docs/`**: Project documentation.
    *   **`architecture.md`**: Detailed system architecture.
    *   **`coding_standards.md`**: Development guidelines.
    *   **`testing_strategy.md`**: Overview of testing approaches.
    *   Various design proposals and READMEs for specific features (e.g., memory pruning, DSPy programs).
*   **`.gitignore`**: Specifies intentionally untracked files (e.g., `__pycache__/`, `chroma_db_test*/`, `.vscode/`).

## 6. Getting Started

1.  **Environment Setup:** Ensure Python environment is set up with dependencies from `requirements.txt` (or similar dependency management file - *please verify which is used*).
2.  **Configuration:** Review `.env.example` and set up a local `.env` file for necessary configurations (API keys, database paths, LLM model settings).
3.  **Run Tests:** Execute the test suite to ensure the environment is correctly configured and to see examples of how components are used.
4.  **Explore `app.py`:** Understand the main execution flow.
5.  **Dive into `src/agents/core/base_agent.py` and `src/agents/graphs/basic_agent_graph.py`**: These are central to agent behavior.
6.  **Review recent integration tests** in `tests/integration/` for practical examples of system operation.

## 7. Points of Contact / Key Contributors

*   (Please fill in with relevant team members)

This report should serve as a solid starting point. Don't hesitate to ask questions and explore the codebase further. Welcome aboard! 