"""Example demonstrating AgentState serialization and deserialization."""

from __future__ import annotations

from src.agents.core.agent_state import AgentState

try:
    # Optional config class if available in this environment
    from src.infra.llm_client import LLMClientConfig
except Exception:  # pragma: no cover - fallback if config class absent
    LLMClientConfig = None  # type: ignore[misc]


def main() -> None:
    """Run a simple serialization round-trip using AgentState."""
    llm_config_instance = None
    if LLMClientConfig is not None:
        llm_config_instance = LLMClientConfig(model_name="test-model", api_key="test-key")

    agent_data_example = {
        "agent_id": "agent1",
        "name": "TestAgent",
        "current_role": "Innovator",
        "mood_level": 0.5,
        "mood_history": [(0, 0.5), (1, 0.6)],
        "relationships": {"agent2": 0.8, "agent3": -0.2},
        "relationship_history": {
            "agent2": [(0, 0.7), (1, 0.8)],
            "agent3": [(0, -0.1), (1, -0.2)],
        },
        "ip": 100.0,
        "du": 50.0,
        "llm_client_config": llm_config_instance.model_dump() if llm_config_instance else None,
    }

    agent_state = AgentState(**agent_data_example)
    print(
        f"AgentState created: {agent_state.name}, Role: {agent_state.current_role}, Mood: {agent_state.mood_level:.2f}"
    )
    if llm_config_instance:
        print(f"LLM Client from state: {agent_state.get_llm_client()}")

    serialized = agent_state.to_dict()
    print(f"Serialized state: {serialized}")

    if LLMClientConfig is not None:
        new_llm_config = LLMClientConfig(model_name="override-model", api_key="override-key")
        deserialized_state = AgentState.from_dict(
            serialized, llm_client_config_override=new_llm_config.model_dump()
        )
    else:
        deserialized_state = AgentState.from_dict(serialized)

    llm_model = (
        deserialized_state.get_llm_client().config.model_name
        if LLMClientConfig and deserialized_state.get_llm_client()
        else "None"
    )
    print(f"Deserialized state: {deserialized_state.name}, LLM Model: {llm_model}")

    agent_state.update_mood(sentiment_score=0.5)
    print(f"Mood after update: {agent_state.mood_level:.2f}, History: {agent_state.mood_history}")

    agent_state.update_relationship("agent2", sentiment_score=-0.3)
    print(
        f"Relationship with agent2: {agent_state.relationships.get('agent2'):.2f}, History: {agent_state.relationship_history.get('agent2')}"
    )

    agent_state.reset_state()
    print(f"Mood after reset: {agent_state.mood_level}, Mood History: {agent_state.mood_history}")
    print(
        f"Relationships after reset: {agent_state.relationships}, Relationship History: {agent_state.relationship_history}"
    )

    agent_state.update_dynamic_config("mood_decay_rate", 0.05)

    agent_state.process_perceived_messages(
        [
            {"sender_name": "agent2", "content": "0.9 Great idea!"},
            {"sender_name": "agent3", "content": "-0.5 I disagree."},
        ]
    )
    print(f"Conversation history: {agent_state.conversation_history}")
    print(f"Relationships with agent2 after msg: {agent_state.relationships.get('agent2'):.2f}")
    print(f"Relationships with agent3 after msg: {agent_state.relationships.get('agent3'):.2f}")


if __name__ == "__main__":  # pragma: no cover - example script
    main()
