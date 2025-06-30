# How to Add a New Agent

This guide outlines the process for adding a new type of agent to the Culture.ai simulation.

## 1. Understand the Core Agent Structure

All agents inherit from the `Agent` class in `src/agents/core/base_agent.py`. This class provides the fundamental scaffolding for an agent's lifecycle, including its state, memory, and interaction with the simulation.

The agent's core data is managed by the `AgentState` Pydantic model in `src/agents/core/agent_state.py`. This model defines all of an agent's attributes, such as its ID, name, role, resources (IP and DU), and relationships.

## 2. Define a New Agent Class (Optional)

For agents with unique behaviors, you can create a new class that inherits from `Agent`.

**Example:**

```python
# src/agents/custom/my_new_agent.py

from src.agents.core.base_agent import Agent

class MyNewAgent(Agent):
    """
    A custom agent with specialized logic.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add any custom initialization logic here

    async def take_action(self, turn_context: dict) -> dict:
        """
        Implement the agent's custom action logic.
        """
        # Custom logic goes here
        return await super().take_action(turn_context)

```

## 3. Define Custom Agent Actions

Agent actions are defined by the `AgentActionIntent` enum in `src/agents/core/agent_state.py`. To add a new action:

1.  Add a new member to the `AgentActionIntent` enum.
2.  Implement the handler for this action in `src/agents/graphs/interaction_handlers.py`.
3.  Add the new handler to the agent graph in `src/agents/graphs/agent_graph_builder.py`.

## 4. Register the New Agent in the Simulation

To make the simulation aware of your new agent, you need to instantiate it in the `create_simulation` function in `src/app.py`.

Modify the agent creation loop to include your new agent class:

```python
# src/app.py

from src.agents.custom.my_new_agent import MyNewAgent

def create_simulation(...):
    # ...
    agents = [
        MyNewAgent(
            agent_id=f"agent_{i + 1}",
            name=f"Agent_{i + 1}",
            vector_store_manager=vsm,
        )
        for i in range(num_agents)
    ]
    # ...
```

## 5. Write Tests for Your New Agent

It is crucial to add tests for your new agent to ensure it behaves as expected and does not introduce regressions.

-   **Unit Tests**: Add unit tests for any new methods or logic in your agent class to `tests/unit/agents/`.
-   **Integration Tests**: Add integration tests to `tests/integration/agents/` to verify that your agent interacts correctly with the rest of the simulation.

By following these steps, you can successfully add new and diverse agents to the Culture.ai simulation, expanding its capabilities and enabling more complex and interesting emergent behaviors. 