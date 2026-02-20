from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


class DecisionProvenance(BaseModel):
    policy_id: str
    rule_id: str
    vote_reference: str | None = None


class PolicyDecision(BaseModel):
    decision: Literal["allow", "deny", "transform", "requires_vote"]
    reason_code: str
    user_message: str | None = None
    provenance: DecisionProvenance
    transformed_updates: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyDecisionService:
    """Evaluates canonical interaction envelopes into typed policy decisions."""

    policy_id = "interaction-policy-v1"

    def decide(
        self,
        *,
        envelope: Any,
        context: Any,
        simulation: Any,
        stage: Literal["entry", "message", "knowledge_board", "control", "moderation"] = "entry",
    ) -> PolicyDecision:
        if stage == "entry":
            return self._entry_decision(envelope=envelope, context=context)
        if stage == "knowledge_board":
            return self._knowledge_board_decision(simulation=simulation)
        if stage == "message":
            return self._message_decision(envelope=envelope, context=context, simulation=simulation)
        if stage in {"control", "moderation"}:
            return self._control_decision(envelope=envelope, context=context)
        return self._allow("policy.default", "policy_allowed")

    def _entry_decision(self, *, envelope: Any, context: Any) -> PolicyDecision:
        if envelope.intent == "spawn":
            if not ({"admin", "moderator"} & set(context.permissions)):
                return self._deny("policy.auth.spawn", "unauthorized", "You are not authorized to spawn agents.")
            return self._allow("policy.auth.spawn", "authorized")
        if envelope.intent in {"moderation", "control", "inject_event"}:
            return self._control_decision(envelope=envelope, context=context)
        if envelope.intent == "human_message":
            content = str(envelope.content or "").strip()
            if content == "/broadcast":
                return self._transform(
                    "policy.normalize.human_message",
                    "normalized_human_command",
                    {"intent": "broadcast", "content": ""},
                )
            if content.startswith("/broadcast "):
                return self._transform(
                    "policy.normalize.human_message",
                    "normalized_human_command",
                    {"intent": "broadcast", "content": content[len("/broadcast ") :]},
                )
            if content.startswith("/kb "):
                return self._transform(
                    "policy.normalize.human_message",
                    "normalized_human_command",
                    {"intent": "knowledge_board", "content": content[4:]},
                )
            return self._transform(
                "policy.normalize.human_message",
                "normalized_human_command",
                {"intent": "direct_message", "content": content},
            )
        return self._allow("policy.entry", "policy_allowed")

    def _knowledge_board_decision(self, *, simulation: Any) -> PolicyDecision:
        now = time.monotonic()
        last_seen = simulation._last_kb_time
        elapsed = now - last_seen
        cooldown = simulation._kb_cooldown
        if elapsed < cooldown:
            retry = max(0.0, cooldown - elapsed)
            return self._deny(
                "policy.cooldown.knowledge_board",
                "kb_rate_limited",
                "Knowledge Board is cooling down. Please try again shortly.",
                metadata={"retry_after_seconds": retry},
            )
        simulation._last_kb_time = now
        return self._allow("policy.cooldown.knowledge_board", "cooldown_ok")

    def _message_decision(self, *, envelope: Any, context: Any, simulation: Any) -> PolicyDecision:
        now = time.monotonic()
        channel_id = context.channel_id
        relay_scope = context.sender_id if channel_id is None else f"{context.sender_id}:{channel_id}"
        last_seen = simulation._last_relay_times.get(relay_scope, 0.0)
        elapsed = now - last_seen
        if elapsed < simulation._relay_cooldown:
            retry = max(0.0, simulation._relay_cooldown - elapsed)
            return self._deny(
                "policy.cooldown.message",
                "rate_limited",
                f"Rate limited: please wait {retry:.1f}s before sending another message.",
                metadata={"retry_after_seconds": retry, "scope": relay_scope},
            )
        simulation._last_relay_times[relay_scope] = now

        if not simulation.agents:
            return self._deny(
                "policy.message.availability",
                "no_agents",
                "No agents are available to receive messages.",
            )

        if envelope.metadata.get("requires_vote") is True:
            return PolicyDecision(
                decision="requires_vote",
                reason_code="requires_vote",
                user_message="Action requires governance vote.",
                provenance=DecisionProvenance(
                    policy_id=self.policy_id,
                    rule_id="policy.governance.vote_gate",
                    vote_reference=str(envelope.metadata.get("vote_reference") or "pending"),
                ),
            )

        return self._allow("policy.message.availability", "policy_allowed")

    def _control_decision(self, *, envelope: Any, context: Any) -> PolicyDecision:
        if not ({"admin", "moderator"} & set(context.permissions)):
            return self._deny(
                "policy.auth.control",
                "unauthorized",
                "You are not authorized to run moderation commands.",
            )
        if envelope.metadata.get("requires_vote") is True:
            return PolicyDecision(
                decision="requires_vote",
                reason_code="requires_vote",
                user_message="Command requires governance vote.",
                provenance=DecisionProvenance(
                    policy_id=self.policy_id,
                    rule_id="policy.governance.control_vote",
                    vote_reference=str(envelope.metadata.get("vote_reference") or "pending"),
                ),
            )
        return self._allow("policy.auth.control", "authorized")

    def _allow(self, rule_id: str, reason_code: str) -> PolicyDecision:
        return PolicyDecision(
            decision="allow",
            reason_code=reason_code,
            provenance=DecisionProvenance(policy_id=self.policy_id, rule_id=rule_id),
        )

    def _deny(
        self,
        rule_id: str,
        reason_code: str,
        user_message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            decision="deny",
            reason_code=reason_code,
            user_message=user_message,
            provenance=DecisionProvenance(policy_id=self.policy_id, rule_id=rule_id),
            metadata=metadata or {},
        )

    def _transform(self, rule_id: str, reason_code: str, updates: dict[str, Any]) -> PolicyDecision:
        return PolicyDecision(
            decision="transform",
            reason_code=reason_code,
            provenance=DecisionProvenance(policy_id=self.policy_id, rule_id=rule_id),
            transformed_updates=updates,
        )
