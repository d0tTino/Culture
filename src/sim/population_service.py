from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from src.agents.core.agent_state import AgentLifecycleState


@dataclass(slots=True)
class LifecycleTransitionResult:
    from_state: AgentLifecycleState
    to_state: AgentLifecycleState
    changed: bool
    artifacts: dict[str, Any]
    archival_policy: dict[str, Any]


class PopulationService:
    """Domain service for population lifecycle and continuity operations."""

    _ALLOWED_TRANSITIONS: ClassVar[dict[AgentLifecycleState, set[AgentLifecycleState]]] = {
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

    def __init__(
        self,
        *,
        policy_hook: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> None:
        self._policy_hook = policy_hook

    async def emit_agent_joined(self, *, simulation: Any, agent: Any, source: str) -> None:
        await self._emit_lifecycle_event(
            simulation=simulation,
            actor_id=agent.agent_id,
            event_type="agent_joined",
            content=f"Agent {agent.agent_id} joined the population.",
            metadata={"source": source},
        )

    async def transition_lifecycle(
        self,
        *,
        simulation: Any,
        agent: Any,
        to_state: AgentLifecycleState,
        reason: str,
        initiated_by: str = "system",
        permissions: set[str] | None = None,
        autonomous: bool = True,
    ) -> LifecycleTransitionResult:
        self._enforce_policy(
            action="retire" if to_state == AgentLifecycleState.RETIRED else "decease",
            initiated_by=initiated_by,
            permissions=permissions,
            autonomous=autonomous,
        )

        state = agent.state
        from_state = AgentLifecycleState(getattr(state, "lifecycle_state", "active"))
        if to_state == from_state:
            return LifecycleTransitionResult(
                from_state=from_state,
                to_state=to_state,
                changed=False,
                artifacts=dict(getattr(state, "legacy_artifacts", {}) or {}),
                archival_policy=dict(getattr(state, "memory_archival_policy", {}) or {}),
            )

        allowed = self._ALLOWED_TRANSITIONS.get(from_state, set())
        if to_state not in allowed:
            raise ValueError(f"Invalid lifecycle transition: {from_state.value} -> {to_state.value}")

        artifacts = dict(getattr(state, "legacy_artifacts", {}) or {})
        if from_state == AgentLifecycleState.ACTIVE:
            artifacts.update(
                self._build_legacy_artifacts(
                    simulation=simulation,
                    agent=agent,
                    reason=reason,
                )
            )

        archival_policy = self._memory_archival_policy(to_state)
        history = list(getattr(state, "lifecycle_history", []) or [])
        history.append(
            {
                "step": simulation.current_step,
                "from": from_state.value,
                "to": to_state.value,
                "reason": reason,
            }
        )

        state.lifecycle_state = to_state
        state.lifecycle_history = history
        state.legacy_artifacts = artifacts
        state.memory_archival_policy = archival_policy
        state.is_alive = to_state != AgentLifecycleState.DECEASED

        if from_state == AgentLifecycleState.ACTIVE and to_state != AgentLifecycleState.ACTIVE:
            state.inheritance = float(getattr(state, "ip", 0.0)) + float(getattr(state, "du", 0.0))
            state.ip = 0.0
            state.du = 0.0

        await self._fan_out_social_impact(simulation=simulation, departed_agent=agent, reason=reason)

        event_name = "agent_retired" if to_state == AgentLifecycleState.RETIRED else "agent_deceased"
        await self._emit_lifecycle_event(
            simulation=simulation,
            actor_id=agent.agent_id,
            event_type=event_name,
            content=(
                f"Agent {agent.agent_id} {('retired' if to_state == AgentLifecycleState.RETIRED else 'deceased')}"
                f" ({reason or 'unspecified_reason'})."
            ),
            metadata={
                "from_state": from_state.value,
                "to_state": to_state.value,
                "reason": reason,
                "legacy_artifacts": artifacts,
                "memory_archival_policy": archival_policy,
            },
        )

        return LifecycleTransitionResult(
            from_state=from_state,
            to_state=to_state,
            changed=True,
            artifacts=artifacts,
            archival_policy=archival_policy,
        )

    async def register_successor(
        self,
        *,
        simulation: Any,
        predecessor: Any,
        successor: Any,
        inherit_role: bool = True,
        inherit_context: bool = False,
        initiated_by: str = "system",
        permissions: set[str] | None = None,
        autonomous: bool = True,
    ) -> dict[str, Any]:
        self._enforce_policy(
            action="register_successor",
            initiated_by=initiated_by,
            permissions=permissions,
            autonomous=autonomous,
        )

        predecessor.state.successor_id = successor.agent_id
        successor.state.predecessor_id = predecessor.agent_id

        inherited: dict[str, Any] = {}
        if inherit_role:
            successor.state.current_role = predecessor.state.current_role
            successor.state.role_embedding = list(getattr(predecessor.state, "role_embedding", []))
            successor.state.reputation_score = float(
                getattr(predecessor.state, "reputation_score", 0.0)
            )
            inherited["role"] = getattr(predecessor.state.current_role, "name", "")

        if inherit_context:
            successor.state.goals = list(getattr(predecessor.state, "goals", []))
            successor.state.current_project_id = getattr(predecessor.state, "current_project_id", None)
            inherited["context"] = {
                "goals_count": len(successor.state.goals),
                "current_project_id": successor.state.current_project_id,
            }

        transferred = float(getattr(predecessor.state, "inheritance", 0.0))
        if transferred > 0:
            successor.state.ip = float(getattr(successor.state, "ip", 0.0)) + transferred
            predecessor.state.inheritance = 0.0
            inherited["inheritance"] = transferred

        payload = {
            "relationship": "successor_of",
            "predecessor_id": predecessor.agent_id,
            "successor_id": successor.agent_id,
            "inherited": inherited,
        }
        await self._emit_lifecycle_event(
            simulation=simulation,
            actor_id=predecessor.agent_id,
            event_type="successor_registered",
            content=f"Agent {successor.agent_id} registered as successor for {predecessor.agent_id}.",
            metadata=payload,
        )
        return payload


    def apply_lifecycle_transition_event(self, *, agent: Any, event: dict[str, Any]) -> None:
        """Project a persisted lifecycle transition event onto an agent state."""
        to_state = AgentLifecycleState(str(event.get("to_state", AgentLifecycleState.ACTIVE.value)))
        history = list(getattr(agent.state, "lifecycle_history", []) or [])
        history.append(
            {
                "step": int(event.get("step", 0)),
                "from": str(event.get("from_state", "active")),
                "to": to_state.value,
                "reason": str(event.get("reason", "")),
            }
        )
        agent.state.lifecycle_state = to_state
        agent.state.lifecycle_history = history
        agent.state.legacy_artifacts = dict(event.get("legacy_artifacts") or {})
        agent.state.memory_archival_policy = dict(event.get("memory_archival_policy") or {})

    def _enforce_policy(
        self,
        *,
        action: str,
        initiated_by: str,
        permissions: set[str] | None,
        autonomous: bool,
    ) -> None:
        policy_context = {
            "action": action,
            "initiated_by": initiated_by,
            "permissions": set(permissions or set()),
            "autonomous": autonomous,
        }
        if self._policy_hook is not None and not self._policy_hook(action, policy_context):
            raise PermissionError(f"Lifecycle transition blocked by policy hook: {action}")

        if autonomous:
            return
        if "admin" not in policy_context["permissions"]:
            raise PermissionError(f"Lifecycle transition requires admin permission: {action}")

    def _build_legacy_artifacts(
        self,
        *,
        simulation: Any,
        agent: Any,
        reason: str,
    ) -> dict[str, Any]:
        projects = dict(getattr(simulation, "projects", {}) or {})
        member_projects: list[dict[str, Any]] = []
        for project_id, project in projects.items():
            members = project.get("members", []) if isinstance(project, dict) else []
            if agent.agent_id in members:
                member_projects.append(
                    {
                        "project_id": project_id,
                        "project_name": str(project.get("name", project_id)),
                        "responsibility": f"Reassign open work from {agent.agent_id}",
                    }
                )

        return {
            "epitaph": (
                f"{agent.agent_id} served as {getattr(agent.state.current_role, 'name', 'unknown')} "
                f"until {reason or 'lifecycle transition'}."
            ),
            "contributions": [
                {
                    "type": "role",
                    "value": getattr(agent.state.current_role, "name", "unknown"),
                },
                {
                    "type": "projects",
                    "value": [item["project_name"] for item in member_projects],
                },
            ],
            "unresolved_obligations": member_projects,
            "inheritance_ledger": [
                {
                    "from": agent.agent_id,
                    "amount": float(getattr(agent.state, "ip", 0.0))
                    + float(getattr(agent.state, "du", 0.0)),
                    "status": "pending_distribution",
                }
            ],
        }

    async def _fan_out_social_impact(
        self,
        *,
        simulation: Any,
        departed_agent: Any,
        reason: str,
    ) -> None:
        for survivor in [a for a in simulation.agents if a.agent_id != departed_agent.agent_id]:
            survivor.state.relationships.pop(departed_agent.agent_id, None)
            history = list(survivor.state.relationship_history.get(departed_agent.agent_id, []))
            history.append((int(simulation.current_step), 0.0))
            survivor.state.relationship_history[departed_agent.agent_id] = history
            if hasattr(survivor.state, "short_term_memory"):
                survivor.state.short_term_memory.append(
                    {
                        "step": simulation.current_step,
                        "type": "lifecycle_social_impact",
                        "agent_id": departed_agent.agent_id,
                        "reason": reason,
                    }
                )

    async def _emit_lifecycle_event(
        self,
        *,
        simulation: Any,
        actor_id: str,
        event_type: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        payload = {
            "type": event_type,
            "step": simulation.current_step,
            "agent_id": actor_id,
            **metadata,
        }
        await simulation.event_kernel.emit_environment_event(payload)
        if getattr(simulation, "knowledge_board", None):
            await simulation.knowledge_board_service.post_event(
                actor_id=actor_id,
                content=content,
                event_type=event_type,
                tags=["population", "lifecycle", event_type],
                reference_metadata=metadata,
                causal_source=f"population_service.{event_type}",
            )

    def _memory_archival_policy(self, lifecycle_state: AgentLifecycleState) -> dict[str, Any]:
        if lifecycle_state == AgentLifecycleState.ACTIVE:
            return {
                "retain_summaries": True,
                "retain_raw_episodic": True,
                "compact_raw_episodic": False,
            }
        if lifecycle_state == AgentLifecycleState.RETIRED:
            return {
                "retain_summaries": True,
                "retain_raw_episodic": True,
                "compact_raw_episodic": True,
            }
        if lifecycle_state == AgentLifecycleState.DECEASED:
            return {
                "retain_summaries": True,
                "retain_raw_episodic": False,
                "compact_raw_episodic": True,
            }
        return {
            "retain_summaries": True,
            "retain_raw_episodic": False,
            "compact_raw_episodic": True,
            "archive_tier": "cold",
        }


__all__ = ["LifecycleTransitionResult", "PopulationService"]
