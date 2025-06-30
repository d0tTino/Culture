from __future__ import annotations

# Skip self argument annotation warnings for simple methods
import math

from .embedding_utils import compute_embedding
from .roles import INITIAL_ROLES


class RoleEmbeddingManager:
    """Manage role embeddings and reputation scores."""

    def __init__(self) -> None:
        self.role_vectors: dict[str, list[float]] = {
            role: compute_embedding(role) for role in INITIAL_ROLES
        }
        self.reputation: dict[str, float] = {role: 0.0 for role in INITIAL_ROLES}

    @staticmethod
    def similarity(v1: list[float], v2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    def best_role(self, query: str, threshold: float = 0.7) -> tuple[str | None, float]:
        q_emb = compute_embedding(query)
        best_role = None
        best_sim = -1.0
        for role, vec in self.role_vectors.items():
            sim = self.similarity(q_emb, vec)
            if sim > best_sim:
                best_role = role
                best_sim = sim
        if best_sim < threshold:
            return None, best_sim
        return best_role, best_sim

    def nearest_role_from_embedding(
        self, embedding: list[float], threshold: float = 0.7
    ) -> tuple[str | None, float]:
        best_role = None
        best_sim = -1.0
        for role, vec in self.role_vectors.items():
            sim = self.similarity(embedding, vec)
            if sim > best_sim:
                best_role = role
                best_sim = sim
        if best_sim < threshold:
            return None, best_sim
        return best_role, best_sim

    def update_role_vector(self, role: str, other_vector: list[float], lr: float = 0.1) -> None:
        current = self.role_vectors.get(role)
        if current is None:
            self.role_vectors[role] = list(other_vector)
        else:
            self.role_vectors[role] = [a + lr * (b - a) for a, b in zip(current, other_vector)]

    def update_reputation(self, role: str, value: float) -> None:
        cur = self.reputation.get(role, 0.0)
        self.reputation[role] = (cur + value) / 2


ROLE_EMBEDDINGS = RoleEmbeddingManager()
