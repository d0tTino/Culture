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
- **Memory System**: Short-term memory allowing agents to recall recent events, thoughts, and interactions
- **Broadcast System**: Communication mechanism allowing agents to share messages with others
- **Knowledge Board**: Shared repository for important ideas and proposals
- **Intent-Based Actions**: Framework for different types of agent interactions
- **Sentiment Analysis**: Ability to analyze emotional tone of messages and adjust agent mood accordingly
- **Project Affiliation**: System for agents to create, join, and leave collaborative projects
- **Simulation Engine**: Customizable simulation environment with round-robin agent activation
- **Scenario Framework**: Support for focused, goal-oriented simulation scenarios
- **Discord Integration**: Enhanced message formatting for Discord with embeds for different event types

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
└── src/                       # Source code
    ├── app.py                 # Main application entry point
    │   ├── core/              # Core agent functionality
    │   │   ├── __init__.py
    │   │   └── base_agent.py  # Base agent class
    │   ├── graphs/            # Agent cognitive graphs
    │   │   └── basic_agent_graph.py  # LangGraph implementation
    │   └── __init__.py
    ├── infra/                 # Infrastructure code
    │   ├── __init__.py
    │   ├── config.py          # Application configuration
    │   └── llm_client.py      # LLM client for Ollama
    ├── interfaces/            # External interface implementations
    │   ├── __init__.py
    │   └── discord_bot.py     # Discord bot integration
    └── sim/                   # Simulation environment
        ├── __init__.py
        ├── simulation.py      # Simulation engine
        └── knowledge_board.py # Shared repository for agent ideas
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

- Long-term memory mechanisms
- More complex social structures
- Visualization tools for agent interactions
- Advanced emotional models
- Goal-oriented agent behaviors
- Extended Knowledge Board functionality
- Enhanced project collaboration mechanics

## License

[Specify appropriate license here]

## Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph) for agent cognition framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [Discord.py](https://discordpy.readthedocs.io/) for Discord integration

## Recent Updates

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

The hierarchical memory system enhances agent continuity across long simulations and enables more coherent reasoning based on past experiences.

For detailed information, see the `hierarchical_memory_README.md` document.

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

To verify the agent state refactoring works correctly:

```
python -m src.test_agent_state
```

To test the role change system:

```
python -m src.test_role_change
```

To test project mechanics:

```
python -m src.test_project_mechanics
```

To test hierarchical memory persistence:

```
python test_hierarchical_memory_persistence.py
```

To test basic memory consolidation:

```
python test_memory_consolidation.py
```

To test level 2 memory consolidation:

```
python run_level2_memory_test.py
```

To test RAG functionality:

```
python test_rag.py
```

To test data unit generation by role:

```
python test_role_du_generation.py
```

To test resource constraint handling:

```
python test_resource_constraints.py
``` 