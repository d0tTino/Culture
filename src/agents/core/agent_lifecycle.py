from __future__ import annotations

from enum import Enum


class AgentLifecycleState(str, Enum):
    ACTIVE = "active"
    RETIRED = "retired"
    DECEASED = "deceased"
    ARCHIVED = "archived"


_ALLOWED_LIFECYCLE_TRANSITIONS: dict[AgentLifecycleState, set[AgentLifecycleState]] = {
    AgentLifecycleState.ACTIVE: {
        AgentLifecycleState.RETIRED,
        AgentLifecycleState.DECEASED,
        AgentLifecycleState.ARCHIVED,
    },
    AgentLifecycleState.RETIRED: {
        AgentLifecycleState.ACTIVE,
        AgentLifecycleState.ARCHIVED,
    },
    AgentLifecycleState.DECEASED: {
        AgentLifecycleState.ARCHIVED,
    },
    AgentLifecycleState.ARCHIVED: {
        AgentLifecycleState.ACTIVE,
    },
}


def is_valid_lifecycle_transition(
    from_state: AgentLifecycleState,
    to_state: AgentLifecycleState,
) -> bool:
    if from_state == to_state:
        return True
    return to_state in _ALLOWED_LIFECYCLE_TRANSITIONS.get(from_state, set())
