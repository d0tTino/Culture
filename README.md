# Culture.ai

A framework for simulating multi-agent AI cultures with cognitive and social capabilities, leveraging LLMs as agent minds.

## Overview

Culture.ai creates a virtual environment where multiple AI agents can interact, communicate, and evolve together. Each agent has:

- Independent thought processes
- Memory of recent interactions
- Sentiment analysis capabilities
- Dynamic mood states
- The ability to broadcast messages to other agents
- The ability to post ideas to a shared Knowledge Board
- Action intents that determine behavior (propose_idea, ask_clarification, continue_collaboration, idle)
- Project affiliation capabilities for collaborative group work

This framework allows for the study of emergent social behaviors, agent cooperation, and cultural development in a controlled environment.

## Features

- **Agent Architecture**: Modular agent design using LangGraph for thought generation and decision-making
- **Memory System**: Hierarchical memory system with short-term, session (Level 1), and chapter (Level 2) summaries
- **Memory Pruning**: Sophisticated pruning system to maintain optimal performance while preserving critical information
- **Broadcast System**: Communication mechanism allowing agents to share messages with others
- **Knowledge Board**: Shared repository for important ideas and proposals
- **Intent-Based Actions**: Framework for different types of agent interactions
- **Sentiment Analysis**: Ability to analyze emotional tone of messages and adjust agent mood accordingly
- **Project Affiliation**: System for agents to create, join, and leave collaborative projects
- **Simulation Engine**: Customizable simulation environment with round-robin agent activation
- **Scenario Framework**: Support for focused, goal-oriented simulation scenarios
- **Discord Integration**: Enhanced message formatting for Discord with embeds for different event types
- **Resource Management**: Agents manage Influence Points (IP) and Data Units (DU) as resources for actions
- **Role System**: Dynamic role system allowing agents to serve as Innovator, Analyzer, or Facilitator
- **Relationship Dynamics**: Non-linear relationship system affecting agent interactions and decision-making
- **DSPy Integration**: Advanced prompt optimization using DSPy with local Ollama models
- **LLM Performance Monitoring**: Comprehensive monitoring of LLM call performance and statistics

## Requirements

- Python 3.8+
- Ollama (for local LLM inference)
- Required Python packages listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/d0tTino/Culture.git
   cd Culture
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install Ollama following the [official instructions](https://ollama.ai/download)

5. Pull the required models:
   ```
   ollama pull mistral:latest
   ```

## Usage

Run a simulation with the default parameters:

```
python -m src.app
```

Run a simulation with Discord integration:

```
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

## License

[MIT License](LICENSE)

## Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph) for agent cognition framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [Discord.py](https://discordpy.readthedocs.io/) for Discord integration
- [DSPy](https://github.com/stanfordnlp/dspy) for prompt optimization
- [ChromaDB](https://www.trychroma.com/) for vector storage

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
- **Configurable Delay**: Respects a configurable delay between L2 creation and L1 pruning to ensure information preservation
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

## Running Tests

Run tests using the Python module format:

```
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