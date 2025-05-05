# Culture.ai

A framework for simulating multi-agent AI cultures with cognitive and social capabilities, leveraging LLMs as agent minds.

## Overview

Culture.ai creates a virtual environment where multiple AI agents can interact, communicate, and evolve together. Each agent has:

- Independent thought processes
- Memory of recent interactions
- Sentiment analysis capabilities
- Dynamic mood states
- The ability to broadcast messages to other agents

This framework allows for the study of emergent social behaviors, agent cooperation, and cultural development in a controlled environment.

## Features

- **Agent Architecture**: Modular agent design using LangGraph for thought generation and decision-making
- **Memory System**: Short-term memory allowing agents to recall recent events, thoughts, and interactions
- **Broadcast System**: Communication mechanism allowing agents to share messages with others
- **Sentiment Analysis**: Ability to analyze emotional tone of messages and adjust agent mood accordingly
- **Simulation Engine**: Customizable simulation environment with round-robin agent activation

## Requirements

- Python 3.8+
- Ollama (for local LLM inference)
- Required Python packages listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Culture.ai.git
   cd Culture.ai
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
    └── sim/                   # Simulation environment
        ├── __init__.py
        └── simulation.py      # Simulation engine
```

## Architecture

### Agents

Each agent in Culture.ai is implemented as an instance of the `Agent` class, containing:

- A unique ID
- An internal state dictionary (including mood, memory, etc.)
- A LangGraph-based cognitive system

### Agent Cognition

Agent thought processes use a graph workflow:
1. **Sentiment Analysis**: Analyze perceived broadcasts and update mood
2. **Thought Generation**: Generate internal thoughts based on context
3. **State Update**: Update internal state and memory

### Simulation Loop

The simulation proceeds in discrete steps:
1. Agents perceive broadcasts from the previous step
2. Each agent takes a turn to process perceptions, generate thoughts, and broadcast
3. Broadcasts are collected for the next step

## Customization

To customize the simulation:

- Adjust the number of agents in `src/app.py`
- Modify the agent's cognitive process in `src/agents/graphs/basic_agent_graph.py`
- Change initialization parameters in `src/app.py`
- Add new agent capabilities by extending the base classes

## Development

### Adding New Features

1. **Enhanced Agent Capabilities**: Extend the `Agent` class or modify the cognition graph
2. **New Environment Features**: Add to the `Simulation` class in `src/sim/simulation.py`
3. **Better LLM Integration**: Enhance the `llm_client.py` for more sophisticated interactions

### Future Directions

- Long-term memory mechanisms
- More complex social structures
- Visualization tools for agent interactions
- Advanced emotional models
- Goal-oriented agent behaviors

## License

[Specify appropriate license here]

## Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph) for agent cognition framework
- [Ollama](https://ollama.ai/) for local LLM inference 