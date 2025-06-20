from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, cast

from typing_extensions import Self

from src.agents.core.agent_state import AgentState
from src.agents.core.mood_utils import get_descriptive_mood
from src.agents.dspy_programs.intent_selector import IntentSelectorProgram
from src.infra.config import get_config

logger = logging.getLogger(__name__)


class AgentController:
    """Controller that applies state update logic."""

    def __init__(self: Self, state: AgentState | None = None, lm: object | None = None) -> None:
        self.state = state
        self.intent_selector = IntentSelectorProgram(lm=lm)

    def select_intent(self: Self, state: object | None = None) -> str:
        """Return the chosen intent from the DSPy program."""
        return self.intent_selector.run()

    # ------------------------------------------------------------------
    # State mutation helpers formerly on ``AgentState``
    # ------------------------------------------------------------------
    def _require_state(self: Self) -> AgentState:
        if self.state is None:
            raise ValueError("AgentController requires an AgentState instance")
        return self.state

    def update_mood(self: Self, sentiment_score: float) -> None:
        state = self._require_state()
        current_numeric = state.mood_level
        decayed = current_numeric * (1.0 - state._mood_decay_rate)
        sentiment_float = float(sentiment_score) if sentiment_score is not None else 0.0
        change = sentiment_float * state._mood_update_rate
        new_level = max(-1.0, min(1.0, decayed + change))
        state.mood_level = new_level
        state.mood_history.append((state.step_counter, new_level))
        mood_desc = get_descriptive_mood(new_level)
        logger.debug(
            "AGENT_STATE MOOD_DEBUG (%s): initial=%.2f decayed=%.2f sentiment=%.2f change=%.2f final=%.2f desc=%s",
            state.agent_id,
            current_numeric,
            decayed,
            sentiment_float,
            change,
            new_level,
            mood_desc,
        )

    def update_relationship(
        self: Self, other_agent_id: str, sentiment_score: float, is_targeted: bool = False
    ) -> None:
        state = self._require_state()
        current_score = state.relationships.get(other_agent_id, 0.0)
        sentiment_score = float(sentiment_score) if sentiment_score is not None else 0.0
        effective = (
            sentiment_score * state._targeted_message_multiplier
            if is_targeted
            else sentiment_score
        )
        if effective > 0:
            lr = state._positive_relationship_learning_rate
        elif effective < 0:
            lr = state._negative_relationship_learning_rate
        else:
            lr = state._neutral_relationship_learning_rate
        change = effective * lr
        new_score = current_score + change
        new_score = max(
            state._min_relationship_score, min(state._max_relationship_score, new_score)
        )
        state.relationships[other_agent_id] = new_score
        if abs(new_score - current_score) > 0.01:
            state.relationship_history.setdefault(other_agent_id, []).append(
                (state.step_counter, new_score)
            )

    def change_role(self: Self, new_role: str, current_step: int) -> bool:
        state = self._require_state()
        if self.can_change_role(new_role, current_step):
            state.ip -= state._role_change_ip_cost
            logger.info(
                "AGENT_STATE (%s): Role changed from %s to %s. IP cost: %.2f. New IP: %.2f",
                state.agent_id,
                state.current_role,
                new_role,
                state._role_change_ip_cost,
                state.ip,
            )
            state.current_role = new_role
            state.steps_in_current_role = 0
            state.role_history.append((current_step, new_role))
            return True
        return False

    def can_change_role(self: Self, new_role: str, current_step: int) -> bool:
        state = self._require_state()
        if new_role == state.current_role:
            logger.debug(
                "AGENT_STATE (%s): Role change to %s denied (already current role).",
                state.agent_id,
                new_role,
            )
            return False
        if state.ip < state._role_change_ip_cost:
            logger.debug(
                "AGENT_STATE (%s): Role change to %s denied (insufficient IP: %.2f < %.2f).",
                state.agent_id,
                new_role,
                state.ip,
                state._role_change_ip_cost,
            )
            return False

        last_change_step = -1
        if len(state.role_history) > 1:
            last_change_step = state.role_history[-1][0]
        if (
            current_step - last_change_step
        ) < state._role_change_cooldown and last_change_step != -1:
            logger.debug(
                "AGENT_STATE (%s): Role change to %s denied (cooldown). current=%d last=%d cd=%d",
                state.agent_id,
                new_role,
                current_step,
                last_change_step,
                state._role_change_cooldown,
            )
            return False

        role_du_gen = get_config("ROLE_DU_GENERATION")
        if isinstance(role_du_gen, dict) and new_role not in role_du_gen:
            logger.warning(
                "AGENT_STATE (%s): Attempted role change to unrecognized role '%s'.",
                state.agent_id,
                new_role,
            )
            return False
        return True

    def update_collective_metrics(self: Self, collective_ip: float, collective_du: float) -> None:
        state = self._require_state()
        state.collective_ip = collective_ip
        state.collective_du = collective_du

    def update_dynamic_config(self: Self, key: str, value: Any) -> None:
        state = self._require_state()
        private_attr_name = f"_{key}"
        if hasattr(state, private_attr_name):
            current_val = getattr(state, private_attr_name)
            try:
                if isinstance(current_val, bool):
                    if isinstance(value, str):
                        lowered = value.strip().lower()
                        if lowered == "true":
                            casted = True
                        elif lowered == "false":
                            casted = False
                        else:
                            casted = bool(value)
                    else:
                        casted = bool(value)
                else:
                    casted = type(current_val)(value)
                setattr(state, private_attr_name, casted)
            except (ValueError, TypeError):
                logger.error(
                    "AGENT_STATE (%s): Failed to update dynamic config %s with %r",
                    state.agent_id,
                    key,
                    value,
                )
        else:
            logger.warning(
                "AGENT_STATE (%s): Attempted to update unknown config '%s'",
                state.agent_id,
                key,
            )

    def add_memory(self: Self, memory_text: str, metadata: dict[str, Any] | None = None) -> None:
        state = self._require_state()
        if not state.memory_store_manager:
            raise ValueError("MemoryStoreManager not initialized to add memory.")
        meta = metadata or {"timestamp": time.time()}
        state.memory_store_manager.add_documents([memory_text], [meta])

    async def aretrieve_relevant_memories(
        self: Self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        state = self._require_state()
        if not state.memory_store_manager:
            raise ValueError("MemoryStoreManager not initialized to retrieve memories.")
        result = await asyncio.to_thread(state.memory_store_manager.query, query, top_k)
        return cast(list[dict[str, Any]], result)

    def process_perceived_messages(self: Self, messages: list[dict[str, Any]]) -> None:
        state = self._require_state()
        for msg in messages:
            sender = msg.get("sender_name", "Unknown")
            content = msg.get("content", "")
            state.conversation_history.append(f"{sender}: {content}")
            try:
                sentiment_value = float(content.split()[0])
                self.update_relationship(sender, sentiment_value, is_targeted=True)
            except (ValueError, IndexError):
                self.update_relationship(sender, 0.0, is_targeted=True)

    def reset_state(self: Self) -> None:
        """Reset mood and relationship histories while preserving known agents."""
        state = self._require_state()
        old_keys = list(state.relationship_history.keys())
        state.relationship_history.clear()
        state.mood_history = [(0, 0.0)]
        state.relationship_history = {name: [(0, 0.0)] for name in old_keys}
