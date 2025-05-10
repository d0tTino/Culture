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

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/d0tTino/Culture.git
   cd Culture
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Ollama following the [official instructions](https://ollama.ai/download)

5. Pull the required models:
   ```bash
   ollama pull mistral:latest
   ```

6. Configuration:
   * Copy `config/.env.example` to `config/.env` if it exists.
   * Edit `config/.env` to set your `OLLAMA_API_BASE` (if not default `http://localhost:11434`) and `DISCORD_BOT_TOKEN` (if using the Discord interface).
   * Review default settings in `src/infra/config.py` and adjust if necessary (e.g., default LLM model, memory pruning settings).

## Usage

Run a simulation with the default parameters:

```bash
python -m src.app
```

Run a simulation with Discord integration:

```bash
python -m src.app --discord
```

### Configuring a Simulation Scenario

You can modify the `SIMULATION_SCENARIO` constant in `src/app.py` to define a specific context and goal for your agents:

```python
SIMULATION_SCENARIO = "The team's objective is to collaboratively design a specification for a decentralized communication protocol suitable for autonomous AI agents operating in a resource-constrained environment. Key considerations are efficiency, security, and scalability."
```

## Project Structure

```
Culture.ai/
├── requirements.txt           # Project dependencies
├── README.md                  # This file
├── data/                      # Data files and logs
│   └── logs/                  # Log files from app and tests
├── docs/                      # Documentation files
│   └── hierarchical_memory_README.md  # Documentation for the hierarchical memory system
├── experiments/               # Experiment scripts and reports
│   ├── dspy_action_intent_experiment.py  # DSPy experiment for action intent selection
│   ├── dspy_action_intent_report.md      # Report on DSPy action intent experiment
│   └── test_dspy_ollama_optimizer_integration.py  # DSPy-Ollama integration tests
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
│   │   ├── graphs/            # Agent cognitive graphs
│   │   │   └── basic_agent_graph.py  # LangGraph implementation
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
    │   ├── test_hierarchical_memory_persistence.py  # Tests for hierarchical memory persistence
    │   ├── test_memory_consolidation.py            # Tests for memory consolidation
    │   ├── test_resource_constraints.py            # Tests for resource constraints
    │   ├── test_memory_pruning.py                  # Tests for memory pruning system
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

Run tests using the Python module format:

```bash
# Run hierarchical memory persistence test
python -m tests.integration.test_hierarchical_memory_persistence

# Run memory consolidation test
python -m tests.integration.test_memory_consolidation

# Run level 2 memory consolidation test
python -m tests.integration.run_level2_memory_test

# Run resource constraint test
python -m tests.integration.test_resource_constraints

# Run RAG functionality test
python -m tests.integration.test_rag

# Run data unit generation by role test
python -m tests.integration.test_role_du_generation

# Run memory pruning test
python -m tests.integration.test_memory_pruning
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

## Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph) for agent cognition framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [Discord.py](https://discordpy.readthedocs.io/) for Discord integration
- [DSPy](https://github.com/stanfordnlp/dspy) for prompt optimization
- [ChromaDB](https://www.trychroma.com/) for vector storage

This project draws inspiration from various fields including Agent-Based Modeling (ABM), Multi-Agent Systems (MAS), artificial life, cognitive science, and the rapidly evolving landscape of Large Language Models.

## Recent Updates

### DSPy Action Intent Selection Experiment

The framework leverages DSPy for optimizing agent action intent selection:

- **Experimental Design**: Created a signature and test protocol for agents to select appropriate action intents based on role, goals, and situation
- **BootstrapFewShot Optimization**: Optimized action intent selection using BootstrapFewShot to learn from examples
- **Role-Appropriate Actions**: Demonstrated that optimized decision-making resulted in actions aligned with agent roles (Facilitator, Analyzer, Innovator)
- **Result Validation**: Verified that optimized selections consistently produced justifications showing understanding of role, goals, and current situation

Detailed documentation is available in `experiments/dspy_action_intent_report.md`.

### DSPy-Ollama Integration

The framework now includes a robust integration between DSPy and local Ollama models, enabling advanced optimizers like BootstrapFewShot to work with local LLMs:

- **OllamaLM Class**: Created a proper implementation that inherits from `dspy.LM` in `src/infra/dspy_ollama_integration.py`, with all required methods including `__init__`, `__call__`, `basic_request`, and `generate`
- **Global Configuration**: Implemented `configure_dspy_with_ollama()` function to easily set up DSPy with an Ollama model
- **Comprehensive Testing**: Developed test script in `experiments/test_dspy_ollama_optimizer_integration.py` with three test scenarios:
  - Direct calls to OllamaLM
  - Using OllamaLM with dspy.Predict
  - Using OllamaLM with DSPy's BootstrapFewShot optimizer
- **API Compatibility**: Fixed implementation issues related to parameter patterns DSPy uses when calling language models
- **Enhanced Experiments**: Updated `dspy_action_intent_experiment.py` to use the new integration

This integration resolves the "No LM is loaded" errors that previously occurred when attempting to use DSPy optimizers with local Ollama models.

### LLM Call Performance Monitoring

The framework now includes comprehensive monitoring for LLM calls to provide insights into performance:

- **Decorator-Based Monitoring**: Added a monitoring decorator that can be applied to any function making LLM calls, maintaining clean separation of concerns
- **Key Metrics Collection**: Tracks important metrics including:
  - Request latency and duration
  - Success/failure status
  - Token usage (prompt and completion)
  - Context information (model name, caller context)
- **Detailed Error Tracking**: Captures error types and messages for failed calls
- **Performance Insights**: Provides visibility into performance bottlenecks, error patterns, and resource utilization
- **Simulation Stability**: Helps identify and address failed LLM calls that might affect simulation stability

This monitoring system adds minimal computational overhead while providing valuable insights for optimizing simulation performance.

### Memory Pruning System

A sophisticated memory pruning system has been implemented to maintain optimal performance while preserving critical information:

- **Hierarchical Pruning**: Implements age-based pruning with different retention policies for each memory hierarchy level
- **Level 1 Summary Pruning**: Automatically prunes Level 1 (session) summaries after they've been consolidated into Level 2 (chapter) summaries
- **Level 2 Summary Pruning**: Prunes older Level 2 summaries based on age to prevent indefinite accumulation
- **Configurable Parameters**: Provides flexible configuration options including:
  - Maximum age for L2 summaries before pruning
  - Check interval frequency to control how often pruning is performed
  - Enable/disable toggles for each pruning level
- **Performance Benefits**: Reduces vector store size and improves retrieval performance by removing redundant memories
- **Verification Tools**: Includes scripts for checking pruning status (`check_pruning.py`) and analyzing pruning logs (`analyze_memory_pruning_log.py`)

The pruning system helps maintain manageable memory sizes as simulations run for extended periods, preventing performance degradation while ensuring critical information is preserved in higher-level memory structures.

### Relationship Dynamics Refinement

The relationship dynamics system has been significantly enhanced with the following improvements:

- **Non-Linear Relationship Updates**: Implemented a sophisticated formula that considers both sentiment and current relationship score
- **Targeted vs. Broadcast Messages**: Direct messages now have stronger impact on relationships than broadcasts
- **Relationship-Based Decision Making**: Agents' behaviors are now influenced by their relationships with others
- **Natural Relationship Decay**: Relationships gradually decay toward neutral, requiring active maintenance
- **Enhanced Prompting**: More nuanced guidance based on relationship intensity

For detailed information, see the `src/relationship_dynamics_verification.md` report.

### Agent State Refactoring

The agent state management was refactored to use Pydantic models instead of plain dictionaries. This provides:

- Type checking and validation
- Better code readability and maintainability
- Structured history tracking of various agent metrics
- Clearer interface between modules

The `AgentState` class in `src/agents/core/agent_state.py` now manages all agent state and provides proper typing for all agent attributes.

### Role Change System

The role change system now allows agents to:

- Request a role change after spending a minimum number of steps in their current role
- Pay an Influence Points cost to change roles
- Switch between Facilitator, Innovator, and Analyzer roles based on strategic considerations

### Hierarchical Memory System

The simulation now includes a sophisticated hierarchical memory system that allows agents to consolidate their experiences at different levels of abstraction:

- **Level 1 - Session Summaries**: Generated from recent short-term memories to capture immediate context and experiences
- **Level 2 - Chapter Summaries**: Generated every ~10 steps from multiple Level 1 summaries, providing higher-level views of experiences over longer time periods
- **ChromaDB Integration**: Persists memories to a vector database for long-term storage and semantic retrieval
- **Memory Filtering**: Enables retrieval of specific memory types based on metadata
- **RAG (Retrieval-Augmented Generation)**: Allows agents to incorporate relevant past experiences into current thinking

For detailed information, see the `docs/hierarchical_memory_README.md` document.

### Resource Constraint Error Handling

The agent action system now includes robust resource constraint checking for any actions that have associated costs:

- **Pre-action Resource Verification**: Checks for sufficient Influence Points (IP) and Data Units (DU) before executing costly actions
- **Role Change Constraints**: Requires sufficient IP and cooldown period before allowing role changes
- **Knowledge Board Posting Constraints**: Requires sufficient IP and DU to post ideas to the Knowledge Board
- **Detailed Clarification Constraints**: Requires sufficient DU for detailed clarification requests
- **Memory Recording**: Failed actions due to resource constraints are recorded in the agent's memory
- **Message Modification**: When an action is blocked due to insufficient resources, the agent's message is automatically modified to acknowledge the constraint
- **Action Downgrading**: When appropriate, actions are downgraded to less resource-intensive alternatives rather than being completely blocked

This system ensures that agents operate within their resource limits while providing appropriate feedback through the simulation.

## DSPy Integration for L1 Summary Generation

The project now leverages DSPy for generating Level 1 (L1) summaries in the agent's cognitive cycle. This implementation:

1. Uses the DSPy framework to create more concise, relevant, and high-quality summaries of the agent's short-term memory events
2. Resides in `src/agents/dspy_programs/l1_summary_generator.py`
3. Integrates with the agent workflow in `src/agents/graphs/basic_agent_graph.py`

The implementation includes:
- `GenerateL1SummarySignature` - A DSPy signature defining the input/output contract
- `L1SummaryGenerator` - The main class that handles the summary generation
- Support for the agent's role and mood in summary generation
- Fallback mechanisms if DSPy is unavailable

### Example Usage

```python
from src.agents.dspy_programs.l1_summary_generator import L1SummaryGenerator

# Create the generator
generator = L1SummaryGenerator()

# Generate a summary with all parameters
summary = generator.generate_summary(
    agent_role="Innovator",
    recent_events="- Step 5, Thought: I should propose a new idea.\n- Step 6, Broadcast: Message sent to all.",
    current_mood="curious"
)

# Generate without mood (optional parameter)
summary = generator.generate_summary(
    agent_role="Facilitator",
    recent_events="- Step 10, Thought: Noticed disagreement.\n- Step 11, Action: Asked for clarification."
)
```

For future optimization and testing, example summaries are provided in `src/agents/dspy_programs/l1_summary_examples.py`.
