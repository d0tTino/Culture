from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_state import AgentState


def apply_gossip_update(
    state: AgentState, other_embedding: list[float], interaction_score: float
) -> None:
    """Update role embedding and role reputation from a gossip interaction."""
    from .role_embeddings import ROLE_EMBEDDINGS

    if not state.current_role.embedding or not other_embedding:
        return
    lr = 0.1
    state.role_embedding = [
        a + lr * interaction_score * (b - a) for a, b in zip(state.role_embedding, other_embedding)
    ]
    state.current_role.embedding = list(state.role_embedding)
    role_name, similarity = ROLE_EMBEDDINGS.nearest_role_from_embedding(other_embedding)
    if not role_name:
        return

    current_reputation = state.role_reputation.get(role_name, 0.0)
    state.role_reputation[role_name] = (current_reputation + similarity * interaction_score) / 2
    state.learned_roles[role_name] = other_embedding
    if role_name == state.current_role.name:
        state.reputation_score = state.role_reputation[role_name]
    ROLE_EMBEDDINGS.update_role_vector(role_name, other_embedding)
    ROLE_EMBEDDINGS.update_reputation(role_name, similarity * interaction_score)
