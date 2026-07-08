# Culture: An AI Genesis Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Repository:** [https://github.com/d0tTino/Culture](https://github.com/d0tTino/Culture)

## Summary
- [Vision](#vision-the-crucible-of-emergent-ai)
- [Getting Started](#getting-started)
- [Setup](#installation)
- [Windows / WSL2 Setup Checklist](docs/windows_setup.md#quick-setup-checklist)
- [Running Tests](#running-tests)
- [Start vLLM](#start-vllm)
- [Signature Demo](#signature-demo)
- [culture-ui Frontend](#culture-ui-frontend)
- [Extensions and Plug-ins](#extensions-and-plug-ins)
- [Council Mode (Experimental)](#council-mode-experimental)
- [Roadmap](#roadmap)
- [Exporting Simulation Traces](#exporting-simulation-traces)

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
* **Semantic Memory:** Higher-level summaries are stored in Neo4j via the SemanticMemoryManager.
* **Retrieval Augmented Generation (RAG):** Agents utilize RAG to inject relevant past memories and knowledge board content into their context for decision-making.
* **Shared Knowledge Board (v1):** A central repository where agents can post ideas and information, which is then perceived by other agents.
* **Resource Management (IP/DU):** Agents manage and utilize Influence Points (IP) and Data Units (DU) for actions like posting to the knowledge board, proposing projects, and changing roles.
* **Relationship Dynamics:** Agents form and evolve dyadic relationships with other agents based on interaction sentiment, influencing their behavior.
* **Collective Metrics:** The simulation tracks collective IP and DU, and agents perceive these global metrics.
* **Dynamic Roles & Basic Goals:** Agents can be assigned roles (Innovator, Analyzer, Facilitator) that influence their behavior and can dynamically request role changes.
* **Basic Group/Project Affiliation:** Agents can propose, create, join, and leave projects.
* **Initial Discord Output:** A Discord bot interface provides real-time visibility into simulation events and routes any user messages through the shared event queue (see `Simulation._handle_human_command`).
* **DSPy Integration:** Advanced prompt optimization using DSPy with local Ollama models.
* **LLM Performance Monitoring:** Comprehensive monitoring of LLM call performance metrics.
* **Memory Pruning System:** Sophisticated pruning to maintain optimal performance while preserving critical information.
* **Semantic Memory:** Consolidation of episodic memories into topic-based summaries stored in Neo4j.
* **AsyncDSPyManager:** Concurrency layer allowing parallel DSPy calls without blocking the event loop.

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
* **AsyncDSPyManager**: Asynchronous DSPy execution for concurrent LLM calls
* **Role System**: Dynamic role system allowing agents to serve as Innovator, Analyzer, or Facilitator
* **Relationship Dynamics**: Non-linear relationship system affecting agent interactions and decision-making
* **DSPy Integration**: Advanced prompt optimization using DSPy with local Ollama models
* **LLM Performance Monitoring**: Comprehensive monitoring of LLM call performance and statistics

### Planned (Medium & Long Term)
* **Advanced Memory Management:**
    * Continued tuning of MUS-based pruning and hierarchical summaries.
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

## Council Mode (Experimental)

Culture ships with an opt-in, experimental **Council Mode** where you can temporarily promote a panel of specialized agents to debate, critique, or ratify pivotal simulation actions before they execute. Because this workflow is still evolving, it is disabled by default—enable it only when you are ready to iterate on council prompts and guardrails. Refer to the [Council Mode design doc](docs/council_mode_design.md) for the latest setup instructions, capabilities, and caveats.

To pose a one-off question to the council without running a full simulation, use the Makefile helper and provide your prompt via `Q`:

```bash
make council Q="Should we prioritize the supply-chain audit?"
```

You can also call the underlying Typer CLI for richer telemetry and RAG context:

```bash
python -m scripts.council_cli \
  "Should we prioritize the supply-chain audit?" \
  --context "Procurement stalled last sprint" \
  --rag-doc "Incident INC-2045" \
  --question-id "audit-priority-check"
```

The Typer CLI is now the canonical entry point for Council Mode and replaces the
legacy `scripts/run_council_cli.py` helper.

Configuration flags that influence council behavior include:

- `USE_COUNCIL_MODE` to enable the feature gate.
- `COUNCIL_CONFIG_PATH` to point at a roster file (defaults to `config/council.yml`).
- `COUNCIL_MAX_CONCURRENT_CALLS` and `DU_BUDGET_PER_QUESTION` to bound LLM usage per council run.
- `ROLE_DU_GENERATION` for persona-specific DU generation budgets.

For a themed loadout, the [PewDiePie-style roster example](docs/council_mode_design.md#pewdiepie-style-roster-youtube-friendly-experiment) shows how to swap in a creator-inspired persona pack via `COUNCIL_CONFIG_PATH` and return to the defaults afterward.

## Technology Stack

* **Core Language:** Python 3.11+
* **Agent Orchestration:** LangChain / LangGraph
* **LLM Hosting/Access:** vLLM (primary) with Ollama as a fallback backend
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

- Python 3.11+
- vLLM for local LLM inference (Ollama is supported as a fallback)
- Required Python packages listed in `requirements.txt`
- Runtime dependencies now include `numpy>=2`
- Additional development and testing dependencies in `requirements-dev.txt` (required for the full test suite)
- `pydantic` is required for both runtime and development

## Getting Started

Follow these steps to run the example simulation locally:

> **Five-minute walkthrough**
> ```bash
> scripts/quickstart.sh
> ```
> This script installs dependencies, launches a local vLLM server, runs the vertical slice simulation, and connects to a Discord channel for interaction.

1. **Clone the repository and create a virtual environment**
   ```bash
   git clone https://github.com/d0tTino/Culture.git
   cd Culture
   python3.11 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate.bat
   ```
2. **Install the dependencies**
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
3. **Copy the example environment file and adjust settings**
   ```bash
   cp .env.example .env
   ```
   Edit `LLM_API_BASE` if your LLM server runs on a
   different URL. Set `DISCORD_BOT_TOKEN` and `DISCORD_CHANNEL_ID` if you plan
   to use the Discord bot.
4. **Install an LLM backend**
  ```bash
  # vLLM (preferred backend)
  pip install vllm
  VLLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2" VLLM_PORT=8001 scripts/start_vllm.sh
  export VLLM_API_BASE="http://localhost:$VLLM_PORT"
  ```
  To use Ollama as a fallback:
  ```bash
  curl https://ollama.ai/install.sh | sh
  ollama pull mistral:latest
  ollama serve &
  ```

### Start vLLM
Launch the OpenAI-compatible API and point the application to it:

```bash
VLLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2" VLLM_PORT=8001 scripts/start_vllm.sh
export VLLM_API_BASE="http://localhost:$VLLM_PORT"
export LLM_API_BASE="$VLLM_API_BASE"   # overrides Ollama when set
```

Requests now use vLLM instead of Ollama. Unset `VLLM_API_BASE` to fall back to
Ollama. See [docs/runbook.md](docs/runbook.md#start-vllm) for additional details.

To compare latency and throughput between vLLM and Ollama, use the benchmarking helper:

```bash
python scripts/benchmark_llm.py "Hello" --runs 3 --model mistral:latest --vllm_base "http://localhost:$VLLM_PORT" --output results.json
```

Example results:

| Backend | Avg latency (s) | Throughput (req/s) |
|---------|----------------:|-------------------:|
| vLLM    | 0.25            | 4.00               |
| Ollama  | 1.20            | 0.83               |

Results will vary depending on hardware and models.

5. **Run the vertical slice demo**
   ```bash
   make local-slice
   ```
   This launches a short simulation with three agents and persists their
   memories in ChromaDB.
6. **Start the optional dashboard**
   ```bash
 python -m src.http_app
  ```
7. **Connect Discord (optional)**
   ```bash
   scripts/start_discord_slice.sh
  ```
   This script loads environment variables from `.env` and launches the vertical
   slice with Discord enabled so you can chat with the agents immediately.

### Scenario Catalog

The repository ships with curated YAML scenarios that you can run via:

```bash
python src/app.py --scenario <path-to-scenario>
```

| Scenario | File | When to use it |
| --- | --- | --- |
| Demo warm-up | `scenarios/demo.yaml` | Quick smoke test with two agents when validating a fresh installation. |
| Planning project | `scenarios/planning_project.yaml` | Longer-form collaboration with role-specific prompts for proposal → critique → vote → deliverable workflows. |
| Signature demo | `scenarios/signature_demo.yaml` | Evaluation showcase with sentiment, coalition, and collective IP/DU instrumentation for regression testing. |
| Crisis response | `scenarios/crisis_response.yaml` | Incident-management drill that stresses alignment during alert, triage, stabilization, and recovery beats. |
| Research sprint | `scenarios/research_sprint.yaml` | Time-boxed discovery sprint emphasizing experiment logs, synthesis, and publication of findings. |

### Signature Demo

Run the evaluation-focused scenario and export reference artifacts with:

```bash
python scripts/run_signature_demo.py
```

The entry point executes `scenarios/signature_demo.yaml`, resets
`results/signature_demo/`, and captures the evaluation hooks used in continuous
integration. Install `matplotlib` (for example, `pip install matplotlib`) to
render the PNG plots. To stream events to Redpanda during the run, set
`ENABLE_REDPANDA=1` and `REDPANDA_BROKER` in your environment before launching
the script.

Each run produces fresh outputs in `results/signature_demo/`, including:

- `event_log.jsonl`, `metrics.json`, and `traces.jsonl`.
- A `snapshots/` directory containing `snapshot_<step>.json` files and the
  deterministic `replay_0_<step>.jsonl` slice.
- `signature_demo_bundle.zip` for sharing or replaying the run end-to-end.
- Optional `plots/` PNGs and an updated `README.md` summarizing the artifact
  locations.

For deeper guidance on evaluation hooks, exports, and replaying the bundle, see
[docs/scenario_hooks.md](docs/scenario_hooks.md#running-and-replaying-signature_demo).

### Quick Start

Run the vertical slice with Discord enabled using the default settings:

```bash
./scripts/vertical_slice.sh --discord
```

This command reads your `.env` file, activates any available virtual environment,
and starts the demo so you can talk with the agents right away.

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
   source venv/bin/activate  # On Windows run `venv\Scripts\activate.bat` or `.venv\Scripts\activate.bat`
   ```

3. Install all required dependencies (including development packages):
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
   The project currently supports **DSPy 2.6.27**; `requirements.txt` pins `dspy-ai==2.6.27`.
   You **must** install the development requirements before running `pytest`.

4. Install Ollama following the [official instructions](https://ollama.ai/download)

  On Windows the project is designed to run inside WSL2. Follow the
  [quick setup checklist](docs/windows_setup.md#quick-setup-checklist) to enable
  GPU support, install Ollama, and activate the virtual environment.

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
See [the Windows setup checklist](docs/windows_setup.md#quick-setup-checklist) for a step-by-step guide on Windows.


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

Set `DASHBOARD_API_TOKEN` to require an access token for state-changing (POST,
PUT, PATCH, DELETE) endpoints. Clients must include
`Authorization: Bearer <token>` when calling those APIs.

You can then consume events using any SSE-capable client. Here's a minimal
example using `httpx`:

```python
import asyncio
import httpx

async def main() -> None:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("GET", "http://localhost:8000/stream/events") as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    print(line.removeprefix("data: "))

asyncio.run(main())
```

The server also exposes `/api/map` for real-time world map updates via SSE:

```bash
curl http://localhost:8000/api/map
```

### Discord Commands
When running with Discord integration you can issue slash commands directly in
your channel.

```text
/broadcast Hello from the outside
```
Sends a broadcast message to all agents.

```text
/pause
```
Pauses the simulation.

```text
/resume
```
Resumes the simulation.

```text
/speed 2.0
```
Adjusts the simulation tick speed.

```text
/kb Add multi-agent architecture diagram to the KB
```
Creates a new entry on the shared Knowledge Board.

```text
/propose_law Agents must document major decisions
```
Proposes a new law and triggers an on-chain vote.

```text
/vote "Agents must document major decisions" true
```
Casts a yes/no vote on the given proposal.

#### Usage and Gating Rules

- The first message you post in an agent's channel binds your Discord user to
  that agent until you chat in another agent's channel.
- Each message consumes Influence Points (IP) and Decision Units (DU) from the
  mapped agent according to the simulation's ledger settings.
- Messages are rejected with an "Insufficient IP/DU" notice if the ledger shows
  the agent lacks the required resources.

See `/status` and `/stats` for ephemeral information about agent resources and
latency.

### Troubleshooting Permission Errors
If the bot fails to respond to commands:
- Confirm the bot's role allows **Send Messages**, **Read Message History**, and
  **Use Application Commands** in the channel.
- Double-check `DISCORD_CHANNEL_ID` is correct and the bot has access to that
  channel.
- Re-invite the bot with the `applications.commands` scope if slash commands do
  not appear.

### Configuring a Simulation Scenario

You can modify the `DEFAULT_SCENARIO` constant in `src/app.py` to define a specific context and goal for your agents:

```python
DEFAULT_SCENARIO = "The team's objective is to collaboratively design a specification for a decentralized communication protocol suitable for autonomous AI agents operating in a resource-constrained environment. Key considerations are efficiency, security, and scalability."
```

## culture-ui Frontend

The `culture-ui` folder provides a lightweight React + TypeScript web UI managed
through **pnpm** workspaces. Install all workspace dependencies from the project
root:

```bash
pnpm install
```

Launch the UI in development mode with:

```bash
pnpm --filter culture-ui dev
```

Once the server is running, visit `http://localhost:5173/memory` to explore agent
memories using the **Memory Explorer** page.
Visit `/storyboard` to view the live Storyboard showing agent actions.

Real-time updates are streamed from `/stream/events` using Server-Sent Events with a WebSocket fallback. See [Stream Events and WebSocket Fallback](docs/culture_ui_requirements.md#streamevents-and-websocket-fallback) for a minimal subscriber example.
Prettier formatting is configured via `.prettierrc`. Format UI code with:

```bash
pnpm --filter culture-ui format
```

A Husky pre-commit hook runs `pnpm lint` and `pnpm type-check` automatically.

See [culture-ui/README.md](culture-ui/README.md) for additional details.
UI requirements are summarized in [docs/culture_ui_requirements.md](docs/culture_ui_requirements.md).
Steps for launching the Memory Explorer are in [docs/memory_explorer.md](docs/memory_explorer.md).


## User Value KPI Reference

The canonical user-value KPI contract is exposed in two places:

- the dashboard KPI Card page (`/kpi-card`) for product-facing reference, and
- this README for engineering-facing implementation details.

The `/api/user_value_metrics` payload is currently versioned as `payload_version = 2`.
Consumers should branch on `payload_version` before assuming metric semantics.

### KPI definitions

| Field | Definition |
| --- | --- |
| `payload_version` | Schema/semantics version for the KPI payload. Increment this whenever field meaning or alert behavior changes. |
| `thresholds.min_novelty_score` | Minimum acceptable `novelty_score` before raising `low_novelty`. |
| `thresholds.min_interaction_diversity` | Minimum acceptable `cross_agent_interaction_diversity` before raising `low_interaction_diversity`. |
| `thresholds.max_repetitive_intents_ratio` | Maximum acceptable `repetitive_intents_ratio` before raising `repetitive_intents`. |
| `thresholds.min_social_graph_change_count` | Minimum acceptable `social_graph_change_count` before raising `no_social_graph_change`. |
| `narrative_continuity_score` | Average of knowledge-board continuation coverage and contiguous event-step continuity. |
| `unresolved_conflict_count` | Conflict entries without linked resolution entries on the knowledge board. |
| `cross_agent_interaction_diversity` | Observed directed interaction pairs divided by the total possible directed agent pairs. |
| `user_intervention_rate` | Fraction of sampled events triggered by `human_command`. |
| `return_session_continuity` | Continuity of snapshot/resume progression across snapshot steps. |
| `novelty_score` | Unique action intents divided by total action intents. |
| `repetitive_intents_ratio` | Frequency of the dominant action intent divided by total action intents. |
| `social_graph_change_count` | Count of events mentioning relationship, coalition, ally, or rival changes. |
| `stagnation_alerts` | Independent alerts emitted when a single KPI crosses its dedicated threshold. |

### Stagnation alerts

The stagnation alerts are intentionally independent so dashboards can explain *which* signal regressed:

- `low_novelty` → `novelty_score < thresholds.min_novelty_score`
- `low_interaction_diversity` → `cross_agent_interaction_diversity < thresholds.min_interaction_diversity`
- `repetitive_intents` → `repetitive_intents_ratio > thresholds.max_repetitive_intents_ratio`
- `no_social_graph_change` → `social_graph_change_count < thresholds.min_social_graph_change_count`

## Extensions and Plug-ins

Culture exposes simple hooks for registering dashboard widgets and agent behaviors.
Use `scripts/create_plugin.py <name>` to scaffold a minimal plug-in package:

```bash
python scripts/create_plugin.py my_plugin
```

See [docs/plugins.md](docs/plugins.md) for details and [docs/plugin_guide.md](docs/plugin_guide.md) for a full walkthrough.

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
│   ├── init_agents.py       # Initialize agents with seed memories
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
  included so tests won't be skipped unexpectedly. Use
  `scripts/install_optional_deps.sh` to install them separately if needed.

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

* **Resource Consciousness:** While ambitious, there's an underlying
  awareness of resource constraints, driving interest in efficient LLMs and
  memory management techniques.

## Roadmap

See [docs/community_roadmap.md](docs/community_roadmap.md) for an overview of
the current milestones and ways plug-in authors can participate.

## Roadmap & Future Work

The project's direction is guided by the future directions listed above. Key future work includes:

* Reviewing the architectural layers summarized in the [Blueprint Memo](docs/blueprint_memo.md).
* Exercising the [Signature Demo](#signature-demo) scenario to validate
  evaluation hooks and artifact generation.

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
- **Full suite** (`pytest -m "slow or dspy or integration" -v -n auto`): Runs all slow, DSPy, and integration tests in parallel
- ChromaDB test DBs are stored in RAM (tmpfs) on Linux for speed; see `docs/testing.md` for details

`pytest` reads settings from `pytest.ini`. If you run tests with `-c /dev/null` or
without the required plugins installed, you may see "unknown mark" warnings.
Use `scripts/run_tests.py` to automatically adjust options based on the
available plugins, or invoke `pytest -c pytest.ini` directly.

Set `SKIP_DEP_INSTALL=1` to skip dependency installation when running
`scripts/run_tests.py` if your environment already has the required
packages.

See [docs/testing.md](docs/testing.md) for full instructions, marker definitions, and troubleshooting.

## Quickstart for Developers

### Prerequisites
- **Python 3.11+**
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
   Alternatively, you can run the model with vLLM. The helper script
   `scripts/start_vllm.sh` launches the server with sensible defaults and the
   `--swap-space` option to avoid out-of-memory errors when running more than
   ten agents:
   ```bash
   # Optionally override the model or port used by vLLM (defaults to port 8001)
   VLLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2" VLLM_PORT=8001 \
   scripts/start_vllm.sh
   # Point the application to the vLLM server
   export LLM_API_BASE="http://localhost:$VLLM_PORT"
   ```
5. **Run Weaviate (for vector store, optional):**
   ```bash
   docker compose up -d  # See docs/testing.md for details
   # Or use ChromaDB (default, file-based, no extra setup needed)
   ```
6. **Configure environment variables:**
   - Copy `.env.example` to `.env` and edit as needed:
   - `LLM_API_BASE` (e.g., http://localhost:11434 or the vLLM base URL)
    - `OLLAMA_REQUEST_TIMEOUT` (request timeout in seconds)
    - `VLLM_MODEL` (e.g., mistralai/Mistral-7B-Instruct-v0.2) for the vLLM backend
    - `VLLM_PORT` (e.g., 8001) for the vLLM backend
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
  - Run with a comma-separated token list when you don't use a database:
    ```bash
    DISCORD_BOT_TOKEN="token1,token2" python -m src.app --discord
    ```
  - When `DISCORD_TOKENS_DB_URL` is set the application loads tokens from the
    database and assigns them to agents by `agent_id` at startup.
  - `ENABLE_OTEL=1` to activate OpenTelemetry log export
  - `OTEL_EXPORTER_ENDPOINT` to override the OTLP log endpoint
  - `ENABLE_REDPANDA=1` to log events to Redpanda
  - `REDPANDA_BROKER` (e.g., localhost:9092) address of the Redpanda broker
  - `REDPANDA_TOPIC` to override the default `culture.events` topic name
  - `SNAPSHOT_COMPRESS=1` to compress simulation snapshots


7. **Initialize agent memories (optional):**
   ```bash
   PYTHONPATH=. python scripts/init_agents.py --n 3 "Hello world"
   ```

   This pre-populates the ChromaDB store with seed memories for `agent_1` through `agent_3`.

    - See `.env.example`, `docs/testing.md`, `docs/redpanda_setup.md`,
      and `docs/opa_setup.md` for details.

### Windows / WSL2 Notes

Running on Windows requires the WSL2 build of **Ollama** (version 0.1.34 or
newer). Expose port `11434` to your host when launching Ollama so the Python
services can reach it. Configure the connection with the `LLM_API_BASE` and
`OLLAMA_REQUEST_TIMEOUT` variables in your `.env` (see `.env.example`).
GPU acceleration is only available when Ollama runs inside WSL2 or Docker.
Install the NVIDIA drivers for WSL2 and run all Python commands from your WSL2
shell. For step-by-step instructions, see
[the Windows setup checklist](docs/windows_setup.md#quick-setup-checklist).
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
You can install Redpanda quickly with:
```bash
curl -1s https://raw.githubusercontent.com/redpanda-data/redpanda/master/install.sh | bash
docker compose -f docker-compose.redpanda.yml up -d
```
Events are written to the `culture.events` topic by default and can be consumed using the `rpk` CLI. Override the topic name with the `REDPANDA_TOPIC` environment variable if desired.

### Prometheus Metrics
The simulation exposes Prometheus metrics on port 8000 when `src.interfaces.metrics` is imported.
Metrics include `llm_latency_ms`, `llm_calls_total`, `knowledge_board_size`, and `active_agent_count`. You can scrape them with a Prometheus server and
check the latest values with the `!stats` Discord command.

For routine operations and troubleshooting, see [docs/runbook.md](docs/runbook.md).
When running against a local vLLM server, set `VLLM_API_BASE` or `LLM_API_BASE` to its base URL. Unset this variable or point `LLM_API_BASE` back to Ollama to switch back.

### Starting the vLLM Server
`scripts/start_vllm.sh` launches the vLLM OpenAI-compatible API with sensible defaults.
Install the package first if it isn't already available:
```bash
pip install vllm
```
Set the following environment variables to customize the launch:

- `VLLM_MODEL` – Hugging Face model name (defaults to `mistralai/Mistral-7B-Instruct-v0.2`)
- `VLLM_PORT` – port for the server (default `8001`)
- `VLLM_SWAP_SPACE` – swap space in GB (default `16`)

Run the script from the project root, overriding values as needed:

```bash
VLLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2" VLLM_PORT=8001 scripts/start_vllm.sh
```

After the server is running, point the application to it:

```bash
export VLLM_API_BASE="http://localhost:$VLLM_PORT"
export LLM_API_BASE="$VLLM_API_BASE"  # overrides Ollama when set
```
Unset `VLLM_API_BASE` to switch back to Ollama, or set `LLM_API_BASE` to your
Ollama URL.

### Walking Vertical Slice
To verify your local setup with actual LLM calls, run the minimal demo script:
```bash
python -m examples.walking_vertical_slice
```
This spins up three agents for a few steps using your local Ollama instance and
persists their memories to ChromaDB. See
[docs/walking_vertical_slice.md](docs/walking_vertical_slice.md) for details.

To quickly try Discord interaction, run:
```bash
scripts/start_discord_slice.sh
```
This loads environment variables from `.env` and starts the same demo with
`--discord` enabled.

You can also launch the demo using Make:
```bash
make local-slice
```
This command activates `.venv` if available, installs the required packages, and
executes `scripts/vertical_slice.sh` (or the Windows `.bat` version).

### Running Tests
Run the full test suite after installing development dependencies and starting an LLM backend.

1. **Launch Ollama or vLLM**
   ```bash
   # Ollama
   ollama serve &
   # or vLLM
   scripts/start_vllm.sh
   ```
2. **Run linters**
   ```bash
   ./scripts/lint.sh --format    # Windows: scripts\lint.bat --format
   ```
3. **Execute tests**
   ```bash
   python -m pytest tests/
   ```
`pytest-xdist` enables parallel execution when you pass `-n auto` explicitly, such as in the full-suite command above.
`scripts/run_tests.py` checks whether this plugin is installed before using parallel options, so tests still run serially without it.
These tests also rely on optional packages (`chromadb`, `weaviate-client`, `langgraph`) which are included in `requirements.txt` and installed in CI.
Generate a coverage report:
```bash
python -m pytest --cov=src --cov-report=term-missing tests/
```
CI enforces `--cov-fail-under=90` for overall coverage.
CI also uploads `coverage.xml` as a GitHub Actions artifact. Open the workflow run
and download the file from the **Artifacts** section.

### Exporting Simulation Traces
Use `scripts/export_traces.py` to convert snapshots or event logs into a JSONL dataset.
```bash
python scripts/export_traces.py --snapshots snapshots/ --output data/traces.jsonl
```
The integration test [tests/integration/tools/test_export_traces.py](tests/integration/tools/test_export_traces.py) validates this export process.

### Project Structure (Key Directories)
- `src/` — Main source code (agents, graphs, memory, infra, simulation)
- `tests/` — Unit and integration tests
- `docs/` — Documentation (architecture, runbook, testing, coding standards)
- `scripts/` — Utility and migration scripts
- `examples/` — Example and experimental scripts
- `archives/` — Historical documents (e.g., [README_archives.md](archives/README_archives.md))

### Contributing
We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code style, review, and testing.

For advanced testing, parallelization, and optional heavy suites, see [docs/testing.md](docs/testing.md).
CI workflows are skipped when a commit only modifies documentation (`*.md` files or files under `docs/`) or contains only code comments. A dedicated `changes` job detects comment-only changes and prevents unnecessary runs.
Outdated runs on the same branch are automatically canceled, and heavy test suites run on a self-hosted Linux runner.
See [docs/ci_status.md](docs/ci_status.md) for tips on checking CI status with the GitHub interface or the `gh` CLI. Because this repository has no remote configured by default, you'll need to add your GitHub remote before checking statuses.

### Troubleshooting
* **LLM connection errors** – Ensure `ollama serve` or `scripts/start_vllm.sh` is running and that `LLM_API_BASE` points to the correct URL.
* **Missing dependencies** – Reinstall with `pip install -r requirements.txt -r requirements-dev.txt`.
* **Check optional packages** – Run `python scripts/check_optional_deps.py` to see which optional dependencies (e.g., `chromadb`, `fastapi`, `asyncpg`) are missing before running tests.
* **Port conflicts** – Set `VLLM_PORT` to a free port when launching the vLLM server.

## Code Quality and Type Safety

As of 2025-06-11 the repository no longer relies on project-wide `mypy` or `ruff` ignores. The codebase is checked in strict mode and suppressions are used only where necessary:

- `src/infra/llm_client.py` – fallback classes for optional dependencies use `type: ignore[no-redef]` and `no-any-unimported` annotations.
- `src/interfaces/dashboard_backend.py` – uses `type: ignore[no-any-unimported]` for `EventSourceResponse`.
- `src/infra/warning_filters.py` – overrides `warnings.showwarning` using `type: ignore[assignment]`.
- `src/shared/llm_mocks.py` – redefines stub classes with `type: ignore[no-redef]`.

All other modules pass Ruff and Mypy without suppressions. DSPy integration remains fully typed with async management via `AsyncDSPyManager`. See the development log for details on the compliance process and any remaining edge cases.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes.
