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
    ├── agents/                # Agent implementation
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