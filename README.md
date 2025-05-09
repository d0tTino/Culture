# Culture: An AI Genesis Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Repository:** [https://github.com/d0tTino/Culture](https://github.com/d0tTino/Culture)

## Vision: The Crucible of Emergent AI

**Culture: An AI Genesis Engine** is an ambitious open-source research project dedicated to creating a dynamic and persistent simulated environment where autonomous AI agents can evolve, interact, and develop complex emergent behaviors. Our primary vision is to build a digital "crucible" – a platform to observe and study the potential genesis of novel AI personalities, dynamic social roles, unique communication styles, AI-driven creativity, and ultimately, rudimentary forms of AI-driven societies and cultures.

This project aims to move beyond task-oriented agents towards a deeper understanding of how sophisticated AI, powered by Large Language Models (LLMs), might develop and interact when placed in a persistent world with shared context, memory, and resource dynamics.

## Core Goals

* **Simulate Emergence:** Foster and study emergent phenomena arising from complex agent interactions.
* **Evolving Agents:** Enable agents to develop and exhibit:
    * Evolving personalities and internal states.
    * Dynamic role allocation and adaptation.
    * Emergent communication patterns and potentially novel language use.
    * AI-driven creativity (e.g., generating ideas, narratives).
    * Complex social structures (groups, alliances, conflicts).
* **Research Platform:** Serve as an experimental platform for AI research, including a "Red Teaming Playground" to test AI resilience, ethics, and alignment in complex social simulations.

## Target Application

The primary target application for the engine is to power an **Experimental AI Social Sandbox**. This could manifest, for example, as an interactive Discord channel where multiple distinct AI characters live, interact with each other and human users, form relationships, and evolve over extended periods based on their experiences and shared knowledge.

## Current Status (as of May 2025)

The "Culture: An AI Genesis Engine" project has established a robust foundational framework. Key implemented and validated components include:

* **Core Agent Architecture:** Agents are orchestrated using LangGraph, allowing for complex internal decision-making flows.
* **Hierarchical Memory System:** Agents possess a two-level memory system:
    * **Level 1 (Session Summaries):** Short-term memories are consolidated into session summaries.
    * **Level 2 (Chapter Summaries):** Level 1 summaries are further consolidated into longer-term chapter summaries.
    * **Persistence & Retrieval:** Both memory levels are persisted in a ChromaDB vector store and are retrievable via RAG, with dedicated test suites (`test_hierarchical_memory_persistence.py`) validating this functionality.
* **Retrieval Augmented Generation (RAG):** Agents utilize RAG to inject relevant past memories and knowledge board content into their context for decision-making.
* **Shared Knowledge Board (v1):** A central repository (`src/sim/knowledge_board.py`) where agents can post ideas and information, which is then perceived by other agents.
* **Resource Management (IP/DU):** Agents manage and utilize Influence Points (IP) and Data Units (DU) for actions like posting to the knowledge board, proposing projects, and changing roles. Resource constraint handling is implemented and tested (`test_resource_constraints.py`).
* **Relationship Dynamics:** Agents form and evolve dyadic relationships with other agents based on interaction sentiment, influencing their behavior.
* **Collective Metrics:** The simulation tracks collective IP and DU, and agents perceive these global metrics. Validated by `test_collective_metrics.py`.
* **Dynamic Roles & Basic Goals:** Agents can be assigned roles (Innovator, Analyzer, Facilitator) that influence their behavior and can dynamically request role changes.
* **Basic Group/Project Affiliation:** Agents can propose, create, join, and leave projects.
* **Initial Discord Output:** A read-only Discord bot interface (`src/interfaces/discord_bot.py`) provides real-time visibility into simulation events.
* **Robust Testing:** Multiple test scripts validate core functionalities. File logging for tests is now operational.

**Ongoing Work (as of this README update):**
* **Task 66a: Debug Memory Pruning Logic:** The initial implementation of Strategy 1 for memory pruning (hierarchical L1 decay) is undergoing debugging to ensure it functions as intended.

## Key Features

### Implemented
* Hierarchical Agent Memory (L1/L2 Consolidation, ChromaDB Vector Storage & Retrieval)
* Retrieval Augmented Generation (RAG) for Contextual Awareness
* Dynamic Agent Roles & Basic Goal-Driven Behavior
* Inter-Agent Relationship Modeling & Dynamics
* Resource Management System (Influence Points & Data Units) with Action Costs & Constraints
* Shared Knowledge Board (v1 - Text-Based) for Collective Information
* Basic Group/Project Affiliation Mechanisms
* LangGraph-based Agent Orchestration for Complex Turn Logic
* Pydantic-based Structured LLM Output
* Collective Resource Metrics (IP/DU) Tracking and Agent Perception
* Initial Read-Only Discord Bot Interface

### Planned (Medium & Long Term)
* **Advanced Memory Management:**
    * Robust Memory Pruning (validating Strategy 1, potentially implementing relevance-based strategies).
    * Further refinements to memory consolidation.
* **LLM & Agent Enhancements:**
    * LLM Call Performance Monitoring.
    * Improved LLM Directive Following & Reliability.
    * Evolving Personalities & Dynamic Trait Systems.
    * Emergent Communication & Language.
    * AI-driven Creativity (idea generation, narrative contributions).
* **Social & Environmental Dynamics:**
    * Complex AI Societies, Group Dynamics & Governance.
    * Dynamic Environmental Cycles ("Seasons") affecting resources and agent behavior.
    * Spatial Simulation / Agent Embodiment in a virtual environment.
* **Knowledge Board Evolution:**
    * Structured content (typed entries, rich metadata, semantic tagging).
    * Enhanced agent interaction (querying, referencing, voting).
    * Potential backing by a **Graph Database** for semantic links and complex queries.
    * Visualization of Knowledge Board content and evolution.
* **User Interaction & Observability:**
    * Full Interactive Discord Integration (bidirectional communication).
    * User Interaction as "Ecosystem Shapers" (Deity Mode).
    * Advanced Visualization Layer for simulation dynamics, agent interactions, and Knowledge Board.
    * Observability and analysis tools for emergent phenomena.
* **Agent Lifecycle & Legacy:**
    * Agent Legacy & Artifacts on the Knowledge Board.
    * Mechanisms for agent "death" or succession.

## Technology Stack

* **Core Language:** Python 3.10+
* **Agent Orchestration:** LangChain / LangGraph
* **LLM Hosting/Access:** Ollama (primarily for local LLMs like Mistral, Llama 3.2 variants)
* **Vector Storage:** ChromaDB
* **Embeddings:** Sentence Transformers
* **State/Cache (Planned/Optional):** Redis
* **Discord Integration:** discord.py
* **Data Validation:** Pydantic
* **Configuration:** Python-based (`config.py`), `.env` files
* **Testing:** `unittest` (Python standard library)

**Future Technology Considerations:**
* **Efficient LLM Inference:** Monitoring developments like **`microsoft/BitNet`** (1-bit LLMs) for potential future integration to run more powerful agents on resource-constrained hardware.
* **Graph Databases:** For advanced Knowledge Board implementation (e.g., Neo4j, Memgraph, ArangoDB).

## Architecture Overview

The project follows a modular design, broadly organized into:
* `src/agents/`: Core logic for agent definition, state, roles, and their decision-making graphs (LangGraph).
* `src/infra/`: Foundational services like LLM client interaction, configuration management, and memory systems (vector store).
* `src/sim/`: The simulation engine, including the main simulation loop, knowledge board, and overall environment management.
* `src/interfaces/`: Connectors to external platforms, currently featuring the Discord bot.
* `src/tests/`: Contains integration and unit tests for various components.

The simulation operates on a turn-based loop, where each agent perceives the environment (including other agents' states, messages, and the knowledge board), processes this information using its LLM-driven graph, decides on an action, and then executes it, affecting its own state and the shared environment.

## Getting Started

This project is primarily a research and development platform.

1.  **Prerequisites:**
    * Python 3.10 or higher.
    * Git.
    * Ollama installed and running (refer to [Ollama GitHub](https://github.com/ollama/ollama)).
    * At least one LLM pulled via Ollama (e.g., `ollama pull mistral`).

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/d0tTino/Culture.git](https://github.com/d0tTino/Culture.git)
    cd Culture
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration:**
    * Copy `config/.env.example` to `config/.env`.
    * Edit `config/.env` to set your `OLLAMA_API_BASE` (if not default `http://localhost:11434`) and `DISCORD_BOT_TOKEN` (if using the Discord interface).
    * Review default settings in `src/infra/config.py` and adjust if necessary (e.g., default LLM model, memory pruning settings).

## Running the Simulation & Tests

* **Main Simulation:**
    ```bash
    python src/app.py
    ```
    (This is the typical entry point; consult `app.py` for command-line arguments if any are added.)

* **Running Tests:**
    Tests are crucial for validating functionality. Key test suites include:
    * `tests/integration/test_hierarchical_memory_persistence.py`
    * `tests/integration/test_resource_constraints.py`
    * `tests/integration/test_collective_metrics.py`
    * `tests/integration/test_memory_pruning.py` (once Task 66a is complete)
    * And others like `test_memory_consolidation.py`.

    To run a specific test module (assuming it's in `tests/integration/` and your `PYTHONPATH` is set or you are in the root `Culture.ai` directory):
    ```bash
    python -m tests.integration.test_hierarchical_memory_persistence
    ```
    To discover and run all tests (standard `unittest` discovery, ensure tests are structured correctly for this):
    ```bash
    python -m unittest discover -s src/tests -p "test_*.py" 
    # Or adjust path if tests are in root
    python -m unittest discover -s . -p "test_*.py" 
    ```

## Project Philosophy

* **Iterative Development:** Building complex features incrementally with continuous testing and refinement.
* **Focus on Emergence:** Designing systems that allow for, rather than explicitly script, complex agent behaviors and societal patterns.
* **Open Experimentation:** The platform is intended to be flexible for trying out different AI models, agent architectures, and simulation parameters.
* **Resource Consciousness:** While ambitious, there's an underlying awareness of resource constraints, driving interest in efficient LLMs and memory management techniques.

## Roadmap & Future Work

The project's direction is guided by the detailed **Development Log** (specifically the "Timeline Considerations" section). Key future work includes:

* **Medium-Term:**
    * Validating and refining Memory Pruning (Task 66a and beyond).
    * Implementing LLM Call Performance Monitoring (based on Task 61 research).
    * Research and experimentation to improve LLM directive following (Task 49).
    * Further refinements to the agent memory system (Task 56 continuation).
* **Long-Term (Wishlist & Grand Vision):**
    * Developing richer agent personalities and enabling their evolution.
    * Fostering emergent communication and AI-driven creativity.
    * Simulating complex AI societies with governance and unique cultures.
    * Introducing dynamic environmental factors ("Seasons") and spatial dimensions.
    * Creating advanced user interaction modes ("Ecosystem God Mode") and comprehensive visualization tools.
    * Exploring agent legacy through persistent artifacts on an evolved, potentially graph-based, Knowledge Board.

## Contributing

Currently, "Culture: An AI Genesis Engine" is primarily a solo research project by [d0tTino](https://github.com/d0tTino). However, the project is open-source, and insights or discussions are welcome via GitHub issues. As the project matures, contribution guidelines may be established.

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project draws inspiration from various fields including Agent-Based Modeling (ABM), Multi-Agent Systems (MAS), artificial life, cognitive science, and the rapidly evolving landscape of Large Language Models. Specific acknowledgements to the open-source communities behind Python, LangChain, Ollama, ChromaDB, and other libraries that make this work possible.
Inspiration for UI/UX and dynamic conversation/knowledge exploration is also drawn from innovative projects in the AI space like `liminalbardo/liminal_backrooms`.
