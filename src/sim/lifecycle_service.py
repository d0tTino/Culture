from __future__ import annotations

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


class LifecycleService:
    """Centralized lifecycle transition rules and side-effects."""

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

    def transition(
        self,
        *,
        agent: Any,
        to_state: AgentLifecycleState,
        step: int,
        reason: str = "",
        projects: dict[str, dict[str, Any]] | None = None,
    ) -> LifecycleTransitionResult:
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
            artifacts.update(self._build_legacy_artifacts(agent=agent, step=step, projects=projects))

        archival_policy = self._memory_archival_policy(to_state)
        history = list(getattr(state, "lifecycle_history", []) or [])
        history.append(
            {
                "step": step,
                "from": from_state.value,
                "to": to_state.value,
                "reason": reason,
            }
        )

        state.lifecycle_state = to_state
        state.lifecycle_history = history
        state.legacy_artifacts = artifacts
        state.memory_archival_policy = archival_policy

        if to_state == AgentLifecycleState.ACTIVE:
            state.is_alive = True
        else:
            state.is_alive = to_state != AgentLifecycleState.DECEASED
            if from_state == AgentLifecycleState.ACTIVE:
                state.inheritance = float(getattr(state, "ip", 0.0)) + float(getattr(state, "du", 0.0))
                state.ip = 0.0
                state.du = 0.0

        return LifecycleTransitionResult(
            from_state=from_state,
            to_state=to_state,
            changed=True,
            artifacts=artifacts,
            archival_policy=archival_policy,
        )

    def register_successor(
        self,
        *,
        predecessor: Any,
        successor: Any,
        inherit_role: bool = True,
        inherit_context: bool = False,
    ) -> dict[str, Any]:
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

        return {
            "relationship": "successor_of",
            "predecessor_id": predecessor.agent_id,
            "successor_id": successor.agent_id,
            "inherited": inherited,
        }

    def _build_legacy_artifacts(
        self,
        *,
        agent: Any,
        step: int,
        projects: dict[str, dict[str, Any]] | None,
    ) -> dict[str, Any]:
        state = agent.state
        kb_summary = (
            f"Legacy summary for {agent.agent_id}: role={getattr(state.current_role, 'name', 'unknown')}, "
            f"mood={float(getattr(state, 'mood_level', 0.0)):.2f}, step={step}."
        )
        relationships = dict(getattr(state, "relationships", {}) or {})
        closure_notes = [
            {
                "agent_id": other,
                "final_score": score,
                "note": f"Relationship with {other} closed at score {score:.2f}.",
            }
            for other, score in sorted(relationships.items())
        ]
        reassignment_tasks: list[dict[str, Any]] = []
        for project_id, project in (projects or {}).items():
            members = project.get("members", []) if isinstance(project, dict) else []
            if agent.agent_id in members:
                reassignment_tasks.append(
                    {
                        "type": "project_reassignment",
                        "project_id": project_id,
                        "project_name": str(project.get("name", project_id)),
                        "task": f"Reassign responsibilities held by {agent.agent_id}.",
                    }
                )
        return {
            "generated_step": step,
            "kb_summary": kb_summary,
            "relationship_closure_notes": closure_notes,
            "project_reassignment_tasks": reassignment_tasks,
        }

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


__all__ = ["LifecycleService", "LifecycleTransitionResult"]
