# Culture: An AI Genesis Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Repository:** [https://github.com/d0tTino/Culture](https://github.com/d0tTino/Culture)

## Summary
- [Vision](#vision-the-crucible-of-emergent-ai)
- [Setup](#installation)
- [Running Tests](#running-tests)

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
    * **Persistence & Retrieval:** Both memory levels are persisted in a ChromaDB vector store and are retrievable via RAG, with dedicated test suites validating this functionality.
* **Retrieval Augmented Generation (RAG):** Agents utilize RAG to inject relevant past memories and knowledge board content into their context for decision-making.
* **Shared Knowledge Board (v1):** A central repository where agents can post ideas and information, which is then perceived by other agents.
* **Resource Management (IP/DU):** Agents manage and utilize Influence Points (IP) and Data Units (DU) for actions like posting to the knowledge board, proposing projects, and changing roles.
* **Relationship Dynamics:** Agents form and evolve dyadic relationships with other agents based on interaction sentiment, influencing their behavior.
* **Collective Metrics:** The simulation tracks collective IP and DU, and agents perceive these global metrics.
* **Dynamic Roles & Basic Goals:** Agents can be assigned roles (Innovator, Analyzer, Facilitator) that influence their behavior and can dynamically request role changes.
* **Basic Group/Project Affiliation:** Agents can propose, create, join, and leave projects.
* **Initial Discord Output:** A read-only Discord bot interface provides real-time visibility into simulation events.
* **DSPy Integration:** Advanced prompt optimization using DSPy with local Ollama models.
* **LLM Performance Monitoring:** Comprehensive monitoring of LLM call performance metrics.
* **Memory Pruning System:** Sophisticated pruning to maintain optimal performance while preserving critical information.

## Key Features

### Implemented
* **Agent Architecture**: Modular agent design using LangGraph for thought generation and decision-making
* **Memory System**: Hierarchical memory system with short-term, session (Level 1), and chapter (Level 2) summaries
* **Memory Pruning**: Sophisticated pruning system to maintain optimal performance while preserving critical information
* **Broadcast System**: Communication mechanism allowing agents to share messages with others
* **Knowledge Board**: Shared repository for important ideas and proposals
* **Intent-Based Actions**: Framework for different types of agent interactions
* **Sentiment Analysis**: Ability to analyze emotional tone of messages and adjust agent mood accordingly
* **Project Affiliation**: System for agents to create, join, and leave collaborative projects
* **Simulation Engine**: Customizable simulation environment with round-robin agent activation
* **Scenario Framework**: Support for focused, goal-oriented simulation scenarios
* **Discord Integration**: Enhanced message formatting for Discord with embeds for different event types
* **Resource Management**: Agents manage Influence Points (IP) and Data Units (DU) as resources for actions
* **Role System**: Dynamic role system allowing agents to serve as Innovator, Analyzer, or Facilitator
* **Relationship Dynamics**: Non-linear relationship system affecting agent interactions and decision-making
* **DSPy Integration**: Advanced prompt optimization using DSPy with local Ollama models
* **LLM Performance Monitoring**: Comprehensive monitoring of LLM call performance and statistics

### Planned (Medium & Long Term)
* **Advanced Memory Management:**
    * Further refinements to memory consolidation and pruning strategies.
* **LLM & Agent Enhancements:**
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

## Requirements

- Python 3.10+
- Ollama (for local LLM inference)
- Required Python packages listed in `requirements.txt`
- Additional development and testing dependencies in `requirements-dev.txt` (required for the full test suite)
- `pydantic` is required for both runtime and development

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/d0tTino/Culture.git
   cd Culture
   cp .env.example .env  # create local environment file
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate.bat (or .venv\Scripts\activate.bat)
   ```

3. Install all required dependencies (including development packages):
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
   The project currently supports **DSPy 2.6.27**; `requirements.txt` pins `dspy-ai==2.6.27`.
   You **must** install the development requirements before running `pytest`.

4. Install Ollama following the [official instructions](https://ollama.ai/download)

5. Pull the required models:
   ```bash
   ollama pull mistral:latest
   ```

6. Configuration:
   * Copy `.env.example` to `.env` in the project root
   * Edit `.env` to customize your simulation settings, including:
     * API keys (if any)
     * LLM model settings
     * Memory pruning thresholds
     * Resource costs/awards
     * Role change parameters
   * See `docs/configuration_system.md` for detailed configuration documentation

Once Ollama is running you can launch a basic simulation:
```bash
python -m src.app --steps 5 --discord
```
See [docs/windows_setup.md#run-the-simulation](docs/windows_setup.md#run-the-simulation) for a step-by-step guide on Windows.


## Code Linting and Formatting

This project uses **Ruff** and **Black** for linting and formatting. Ruff handles
import sorting and style checks (replacing Flake8 and isort), while Black
provides opinionated code formatting. **Mypy** is used for static type checking.

To install development dependencies (including linting tools), run:
```bash
pip install -r requirements-dev.txt
```
These development requirements now include `numpy>=2` alongside tools like Ruff,
Black, and Mypy.

To run the linters and type checker locally, use the helper scripts:
```bash
# Linux/Mac
./scripts/lint.sh --format  # omit --format to only check
# Windows
scripts\lint.bat --format
```
These scripts execute `ruff check`, `black`, and `mypy` to match the CI pipeline.
You can still run the commands manually:
```bash
ruff check .
black --check .
ruff format
```

See `docs/coding_standards.md` for detailed information about our coding standards and linting setup.

## Development Practices

### Code Review

All significant code changes undergo review to ensure quality and maintainability. Our lightweight code review process helps maintain standards, share knowledge, and catch issues early.

For more information, see `docs/code_review_process.md`.

## Usage

Run a simulation with the default parameters:

```bash
python -m src.app
```

Run a simulation with Discord integration:

```bash
python -m src.app --discord
```

Start the optional HTTP dashboard backend (for streaming events via SSE):

```bash
python -m src.http_app
```

### Configuring a Simulation Scenario

You can modify the `DEFAULT_SCENARIO` constant in `src/app.py` to define a specific context and goal for your agents:

```python
DEFAULT_SCENARIO = "The team's objective is to collaboratively design a specification for a decentralized communication protocol suitable for autonomous AI agents operating in a resource-constrained environment. Key considerations are efficiency, security, and scalability."
```

## Project Structure

```
Culture.ai/
├── requirements.txt           # Project dependencies
├── README.md                  # This file
├── data/                      # Data files and logs
│   └── logs/                  # Log files from app and tests
├── docs/                      # Documentation files
├── examples/                  # Usage examples and small scripts
│   ├── minimal_repro.py
│   └── test_synthesizer.py
├── scripts/                   # Utility scripts for project management
│   └── cleanup_temp_db.py     # Script to clean up temporary ChromaDB directories
├── src/                       # Source code
│   ├── app.py                 # Main application entry point
│   ├── agents/                # Agent implementation
│   │   ├── core/              # Core agent functionality
│   │   │   ├── __init__.py
│   │   │   ├── base_agent.py  # Base agent class
│   │   │   ├── agent_state.py # Pydantic model for agent state
│   │   │   └── roles.py       # Role definitions and behaviors
│   │   ├── dspy_programs/     # DSPy-based components
│   │   │   ├── __init__.py
│   │   │   ├── l1_summary_generator.py     # DSPy L1 summary generation
│   │   │   ├── l1_summary_examples.py      # Examples for L1 summary training
│   │   │   ├── role_thought_generator.py   # Role-based thought generation
│   │   │   ├── action_intent_selector.py   # Action intent selection
│   │   │   └── rag_context_synthesizer.py  # RAG context processing
│   │   ├── graphs/            # Agent cognitive graphs
│   │   │   ├── agent_graph_builder.py  # Build LangGraph workflows
│   │   │   ├── graph_nodes.py          # Individual graph nodes
│   │   │   ├── interaction_handlers.py # Interaction handlers
│   │   │   └── basic_agent_graph.py    # Coordinator tying nodes and handlers
│   │   └── __init__.py
│   ├── infra/                 # Infrastructure code
│   │   ├── __init__.py
│   │   ├── config.py          # Application configuration
│   │   ├── llm_client.py      # LLM client with monitoring
│   │   ├── dspy_ollama_integration.py  # Integration for DSPy with Ollama
│   │   ├── logging_config.py  # Logging configuration
│   │   └── memory/            # Memory infrastructure
│   │       ├── __init__.py
│   │       └── vector_store.py  # ChromaDB integration for memories
│   ├── interfaces/            # External interface implementations
│   │   ├── __init__.py
│   │   └── discord_bot.py     # Discord bot integration
│   ├── utils/                 # Utility functions and helpers
│   │   └── __init__.py
│   └── sim/                   # Simulation environment
│       ├── __init__.py
│       ├── simulation.py      # Simulation engine
│       └── knowledge_board.py # Shared repository for agent ideas
└── tests/                     # Tests for the project
    ├── data/                  # Test data and fixtures
    ├── integration/           # Integration tests
    │   ├── test_memory_pruning.py         # Tests for memory pruning system
    │   ├── test_collective_metrics.py     # Tests for collective metrics
    │   └── ... (other test files)
    └── unit/                  # Unit tests
```

## Architecture

### Agents

Each agent in Culture.ai is implemented as an instance of the `Agent` class, containing:

- A unique ID
- An internal state dictionary (including mood, memory, etc.)
- A LangGraph-based cognitive system
- Project affiliations

### Agent Cognition

Agent thought processes use a graph workflow:
1. **Sentiment Analysis**: Analyze perceived broadcasts and update mood
2. **Prepare Relationship Prompt**: Adjust communication based on agent relationships
3. **Generate Action Output**: Generate thoughts, broadcasts, and select an action intent
4. **Handle Intent**: Process the selected intent (propose_idea, ask_clarification, etc.)
5. **Update State**: Update internal state and memory

### Action Intents

Agents can select from different action intents:
- **propose_idea**: Suggest a formal idea to be added to the Knowledge Board
- **ask_clarification**: Request more information about something unclear
- **continue_collaboration**: Standard contribution to ongoing discussion
- **idle**: No specific action, continue monitoring
- **perform_deep_analysis**: Conduct thorough analysis of a proposal or situation
- **create_project**: Create a new project for collaboration
- **join_project**: Join an existing project
- **leave_project**: Leave a project

### Project Affiliation System

The project affiliation system allows agents to:
- Create new projects with custom names and descriptions (costs IP and DU)
- Join existing projects created by other agents (costs IP and DU)
- Leave projects they are currently affiliated with (free)
- See all available projects and their current members
- Collaborate more closely with project members

### Simulation Loop

The simulation proceeds in discrete steps:
1. Agents perceive broadcasts from the previous step and the current Knowledge Board
2. Each agent takes a turn to process perceptions, generate thoughts, and select an action intent
3. The Knowledge Board is updated with new entries
4. Broadcasts are collected for the next step

## Customization

To customize the simulation:

- Adjust the number of agents in `src/app.py`
- Modify the agent's cognitive process in `src/agents/graphs/basic_agent_graph.py`
- Change initialization parameters in `src/app.py`
- Add new agent capabilities by extending the base classes
- Define a specific simulation scenario in `src/app.py`
- Configure project system parameters in `src/infra/config.py`

## Development

### Adding New Features

1. **Enhanced Agent Capabilities**: Extend the `Agent` class or modify the cognition graph
2. **New Environment Features**: Add to the `Simulation` class in `src/sim/simulation.py`
3. **Better LLM Integration**: Enhance the `llm_client.py` for more sophisticated interactions
4. **New Action Intents**: Add new intent types and handlers to expand agent behaviors

### Future Directions

- More complex social structures
- Visualization tools for agent interactions
- Advanced emotional models
- Goal-oriented agent behaviors
- Enhanced Knowledge Board functionality
- Advanced project collaboration mechanics

## Running Tests

### Test Suite Setup
1. *(Optional)* Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install both runtime and development dependencies:
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
   The development requirements include tools such as `pytest-xdist`,
   `pytest-asyncio`, `requests`, and `numpy>=2` which are required for the full test suite.
   They also install **pip-tools**, which provides `pip-compile`.
   Run `scripts/check_requirements.sh` after modifying dependencies to ensure
   `requirements.txt` matches `requirements.in`.
   You can also run `scripts/setup_test_env.sh` to automate these steps.
   Optional packages like `chromadb`, `weaviate-client`, and `langgraph` are
   included so tests won't be skipped unexpectedly.

Run tests using the Python module format:

```bash
# Run memory pruning test
python -m tests.integration.test_memory_pruning

# Run collective metrics test
python -m tests.integration.test_collective_metrics

# Run resource constraint test
python -m tests.integration.test_resource_constraints
```

Test logs are stored in the `data/logs/` directory.

## Project Philosophy

* **Iterative Development:** Building complex features incrementally with continuous testing and refinement.
* **Focus on Emergence:** Designing systems that allow for, rather than explicitly script, complex agent behaviors and societal patterns.
* **Open Experimentation:** The platform is intended to be flexible for trying out different AI models, agent architectures, and simulation parameters.
* **Resource Consciousness:** While ambitious, there's an underlying awareness of resource constraints, driving interest in efficient LLMs and memory management techniques.

## Roadmap & Future Work

The project's direction is guided by the future directions listed above. Key future work includes:

* **Medium-Term:**
    * Validating and refining Memory Pruning.
    * Implementing LLM Call Performance Monitoring.
    * Research and experimentation to improve LLM directive following.
    * Further refinements to the agent memory system.
* **Long-Term (Wishlist & Grand Vision):**
    * Developing richer agent personalities and enabling their evolution.
    * Fostering emergent communication and AI-driven creativity.
    * Simulating complex AI societies with governance and unique cultures.
    * Introducing dynamic environmental factors ("Seasons") and spatial dimensions.
    * Creating advanced user interaction modes ("Ecosystem God Mode") and comprehensive visualization tools.
    * Exploring agent legacy through persistent artifacts on an evolved, potentially graph-based, Knowledge Board.

## License


This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

## How to Cite

Citation metadata is provided in [CITATION.cff](CITATION.cff). Most reference managers can read this file directly. If you use Culture in your research, please cite the latest release using this metadata.

## Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph) for agent cognition framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [Discord.py](https://discordpy.readthedocs.io/) for Discord integration
- [DSPy](https://github.com/stanfordnlp/dspy) for prompt optimization
- [ChromaDB](https://www.trychroma.com/) for vector storage

This project draws inspiration from various fields including Agent-Based Modeling (ABM), Multi-Agent Systems (MAS), artificial life, cognitive science, and the rapidly evolving landscape of Large Language Models.

## Recent Updates

### DSPy Memory Summarization Integration

The project now leverages DSPy for generating both Level 1 (L1) and Level 2 (L2) summaries in the agent's cognitive cycle, marking a significant improvement over the previous direct LLM call approach:

- **Enhanced L1 Summaries**: More concise, relevant, and coherent session-level summaries through DSPy's structured approach to prompting
- **Enhanced L2 Summaries**: Higher-quality chapter-level summaries that synthesize multiple L1 summaries into comprehensive insights
- **Context-Aware Processing**: Takes into account the agent's current role, mood trends, and goals to produce contextually appropriate summaries
- **Robust Implementation**: Includes fallback mechanisms when DSPy is unavailable
- **Future Optimization Ready**: Contains example infrastructure for future optimization using DSPy's learning capabilities

This implementation significantly improves the quality of memory summarization at both levels of the hierarchical memory system, resulting in better long-term memory representation and more relevant information retrieval during agent cognition.

### Memory Pruning Improvements

L2 summary pruning functionality has been implemented to manage long-term growth of the memory system:

- **Automatic Cleanup**: Removes older L2 summaries based on configurable age thresholds
- **Configurable Parameters**: Added control settings in the configuration system
- **Preservation of Recent Information**: Ensures only truly outdated information is removed while preserving important recent memory

### DSPy Action Intent Selection Experiment

The framework leverages DSPy for optimizing agent action intent selection:

- **Experimental Design**: Created a signature and test protocol for agents to select appropriate action intents based on role, goals, and situation
- **BootstrapFewShot Optimization**: Optimized action intent selection using BootstrapFewShot to learn from examples
- **Role-Appropriate Actions**: Demonstrated that optimized decision-making resulted in actions aligned with agent roles (Facilitator, Analyzer, Innovator)
- **Result Validation**: Verified that optimized selections consistently produced justifications showing understanding of role, goals, and current situation
### Asynchronous DSPy Program Management (AsyncDSPyManager)

All DSPy program calls (for memory summarization, action intent selection, and relationship updating) are now managed asynchronously via the `AsyncDSPyManager`. This enables non-blocking, parallel DSPy execution for all agents, with robust timeout and error handling—if a DSPy call is slow or fails, a failsafe output is returned and the simulation continues smoothly.

Agent and graph methods that invoke DSPy programs are now `async def` and must be awaited. The main simulation loop is fully asynchronous, using `asyncio.run()`. This pattern significantly improves simulation responsiveness, stability, and scalability, especially as agent populations grow or LLM calls become slow or unreliable.

For more details, see [docs/architecture.md](docs/architecture.md#61-asynchronous-dspy-program-management-asyncdspymanager).

## Testing

Culture.ai uses pytest with marker-based test selection and parallelization for fast feedback:

- **Default run** (`pytest`): Runs only unit tests (fast, no external dependencies)
- **Full suite** (`pytest -m "slow or dspy_program or integration" -v -n auto`): Runs all slow, DSPy, and integration tests in parallel
- ChromaDB test DBs are stored in RAM (tmpfs) on Linux for speed; see `docs/testing.md` for details

See [docs/testing.md](docs/testing.md) for full instructions, marker definitions, and troubleshooting.

## Quickstart for Developers

### Prerequisites
- **Python 3.10+**
- **Ollama** (for local LLM inference): [Install Ollama](https://ollama.ai/download)
- **Docker** (for Weaviate vector store, optional but recommended)

### Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/d0tTino/Culture.git
   cd Culture
   cp .env.example .env  # create local environment file
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Linux/Mac:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate.bat
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
   Alternatively you can run `scripts/codex_setup.sh` to install everything and
   set up pre-commit hooks in one step.
4. **Set up Ollama and pull the required model:**
   ```bash
   ollama pull mistral:latest
   ```
   Alternatively, you can run the model with vLLM. Start the server with the
   `--swap-space` option to avoid out-of-memory errors when running more than
   ten agents:
   ```bash
   scripts/start_vllm.sh  # defaults to port 8001 (override with VLLM_PORT)
   ```
5. **Run Weaviate (for vector store, optional):**
   ```bash
   docker compose up -d  # See docs/testing.md for details
   # Or use ChromaDB (default, file-based, no extra setup needed)
   ```
6. **Configure environment variables:**
   - Copy `.env.example` to `.env` and edit as needed:
    - `OLLAMA_API_BASE` (e.g., http://localhost:11434)
    - `OLLAMA_REQUEST_TIMEOUT` (request timeout in seconds)
    - `WEAVIATE_URL` (e.g., http://localhost:8080)
    - `VECTOR_STORE_BACKEND` ("chroma" or "weaviate")
    - `DISCORD_BOT_TOKEN` and `DISCORD_CHANNEL_ID` (for Discord integration)
   - `DISCORD_TOKENS_DB_URL` for Postgres storage of additional bot tokens
     (`postgresql://user:pass@localhost/dbname`). Initialize the table with
     `scripts/init_discord_tokens.sql`:
     ```sql
     CREATE TABLE IF NOT EXISTS discord_tokens (
         agent_id TEXT PRIMARY KEY,
         token TEXT NOT NULL
     );
     ```
   - `ENABLE_OTEL=1` to activate OpenTelemetry log export
   - `ENABLE_REDPANDA=1` to log events to Redpanda
   - `REDPANDA_BROKER` (e.g., localhost:9092) address of the Redpanda broker

   - See `.env.example`, `docs/testing.md`, `docs/redpanda_setup.md`,
     and `docs/opa_setup.md` for details.

### Windows / WSL2 Notes

Running on Windows requires the WSL2 build of **Ollama** (version 0.1.34 or
newer). Expose port `11434` to your host when launching Ollama so the Python
services can reach it. Configure the connection with the `OLLAMA_API_BASE` and
`OLLAMA_REQUEST_TIMEOUT` variables in your `.env` (see `.env.example`). GPU
acceleration is only available when running Ollama through Docker or WSL2.
For step-by-step instructions, see
[docs/windows_setup.md](docs/windows_setup.md).
The provided `scripts\vertical_slice.bat` detects both `venv` and `.venv` virtual environments when activating the demo.

### Running the Simulation
Run a basic simulation (default parameters):
```bash
python -m src.app --steps 5
```

Display the installed version:
```bash
python -m src.app --version
python -m src.http_app --version
```

Start the HTTP dashboard backend (optional):
```bash
python -m src.http_app
```

The application initializes logging using `setup_logging()` from
`src.infra.logging_config`. Log files are written to the `logs/` directory by
default. Adjust the log level or path as needed by customizing this function.

For deterministic event logging and replay, install Redpanda as described in
[docs/redpanda_setup.md](docs/redpanda_setup.md). Set `ENABLE_REDPANDA` and
`REDPANDA_BROKER` in your `.env` to activate this feature.

### Prometheus Metrics
The simulation exposes Prometheus metrics on port 8000 when `src.interfaces.metrics` is imported.
Metrics include `llm_latency_ms`, `llm_calls_total`, `knowledge_board_size`, and `active_agent_count`. You can scrape them with a Prometheus server and
check the latest values with the `!stats` Discord command.

For routine operations and troubleshooting, see [docs/runbook.md](docs/runbook.md).

### Walking Vertical Slice
To verify your local setup with actual LLM calls, run the minimal demo script:
```bash
python -m examples.walking_vertical_slice
```
This spins up three agents for a few steps using your local Ollama instance and
persists their memories to ChromaDB. See
[docs/walking_vertical_slice.md](docs/walking_vertical_slice.md) for details.

You can also launch the demo using Make:
```bash
make local-slice
```
This command activates `.venv` if available, installs the required packages, and
executes `scripts/vertical_slice.sh` (or the Windows `.bat` version).

### Running Tests
Run the full test suite (after installing development dependencies):
```bash
python -m pytest tests/
```
`pytest-xdist` is required for this command because the default `pytest.ini` uses `-n auto`.
Install it via `requirements-dev.txt` if you haven't already.
These tests also rely on optional packages (`chromadb`, `weaviate-client`, `langgraph`) which are included in `requirements.txt` and installed in CI.
Generate a coverage report:
```bash
python -m pytest --cov=src --cov-report=term-missing tests/
```
CI enforces `--cov-fail-under=90` for overall coverage.
CI also uploads `coverage.xml` as a GitHub Actions artifact. Open the workflow run
and download the file from the **Artifacts** section.

### Project Structure (Key Directories)
- `src/` — Main source code (agents, graphs, memory, infra, simulation)
- `tests/` — Unit and integration tests
- `docs/` — Documentation (architecture, runbook, testing, coding standards)
- `scripts/` — Utility and migration scripts
- `examples/` — Example and experimental scripts
- `archives/` — Historical documents (e.g., [README_archives.md](archives/README_archives.md))

### Contributing
We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code style, review, and testing.

For advanced testing, parallelization, and CI details, see [docs/testing.md](docs/testing.md).
CI workflows are skipped when a commit only modifies documentation (`*.md` files or files under `docs/`) or contains only code comments. A dedicated `changes` job detects comment-only changes and prevents unnecessary runs.
Outdated runs on the same branch are automatically canceled, and heavy test suites run on a self-hosted Linux runner.
See [docs/ci_status.md](docs/ci_status.md) for tips on checking CI status with the GitHub interface or the `gh` CLI. Because this repository has no remote configured by default, you'll need to add your GitHub remote before checking statuses.

## Code Quality and Type Safety

As of 2025-06-11 the repository no longer relies on project-wide `mypy` or `ruff` ignores. The codebase is checked in strict mode and suppressions are used only where necessary:

- `src/infra/llm_client.py` – fallback classes for optional dependencies use `type: ignore[no-redef]` and `no-any-unimported` annotations.
- DSPy program modules in `src/agents/dspy_programs/` – `dspy.Signature` is dynamic, so each file keeps `mypy: ignore-errors` and ruff `noqa` directives for long example strings.
- Graph and core modules under `src/agents/graphs/` and `src/agents/core/` – rely on runtime graph construction and maintain `mypy: ignore-errors`.
- `src/interfaces/discord_bot.py` – optional `discord` imports require `mypy: ignore-errors` and an `ANN401` suppression.
- `src/interfaces/dashboard_backend.py` – uses `type: ignore[no-any-unimported]` for `EventSourceResponse`.
- `src/infra/dspy_ollama_integration.py` – dynamic LM patching requires `mypy: ignore-errors` and annotation ignores.
- `src/infra/warning_filters.py` – overrides `warnings.showwarning` using `type: ignore[assignment]`.
- `src/shared/llm_mocks.py` – redefines stub classes with `type: ignore[no-redef]`.
- `src/sim/simulation.py` – retains `mypy: ignore-errors` and ruff `RUF006` for asynchronous initialization.

All other modules pass Ruff and Mypy without suppressions. DSPy integration remains fully typed with async management via `AsyncDSPyManager`. See the development log for details on the compliance process and any remaining edge cases.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes.
