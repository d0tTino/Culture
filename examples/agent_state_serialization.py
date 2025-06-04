"""Example demonstrating AgentState serialization and deserialization."""

from src.agents.core.agent_state import AgentState


def main() -> None:
    """Create, serialize, and deserialize an AgentState."""
    state = AgentState(agent_id="agent1", name="ExampleAgent")
    print("Original state:", state)

    data = state.to_dict()
    print("Serialized:", data)

    restored = AgentState.from_dict(data)
    print("Deserialized:", restored)


if __name__ == "__main__":
    main()
