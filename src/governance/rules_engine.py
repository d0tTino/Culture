from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any


@dataclass(slots=True)
class RulePenalty:
    """Penalty to apply when a rule violation is overridden instead of denied."""

    ip: float = 0.0
    du: float = 0.0
    reason: str = "rule_violation"


@dataclass(slots=True)
class GovernanceRule:
    """Executable rule materialized from an accepted governance proposal."""

    rule_id: str
    source_text: str
    action_intent: str
    effective_date: str
    decision_mode: str = "deny"
    penalty: RulePenalty = field(default_factory=RulePenalty)
    provenance: dict[str, Any] = field(default_factory=dict)
    enforcement_stats: dict[str, int] = field(
        default_factory=lambda: {
            "checked": 0,
            "accepted": 0,
            "rejected": 0,
            "overridden": 0,
            "penalties_applied": 0,
        }
    )


@dataclass(slots=True)
class RuleEvaluationResult:
    """Result of evaluating an action against active governance rules."""

    allowed: bool
    decision: str
    violated_rules: list[str]
    reason: str | None = None
    penalties: list[dict[str, Any]] = field(default_factory=list)


class RulesEngine:
    """Materializes accepted proposals into executable action constraints."""

    _BAN_PATTERN = re.compile(
        r"(?:^|\b)(?:no|ban|forbid)\s+(?P<action>[a-z_\-\s]+)",
        flags=re.IGNORECASE,
    )
    _PENALIZE_PATTERN = re.compile(
        r"(?:^|\b)(?:penalize|penalty)\s+(?P<action>[a-z_\-\s]+)\s+(?:by|with)\s+"
        r"(?P<amount>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>ip|du)",
        flags=re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._active_rules: dict[str, GovernanceRule] = {}

    @staticmethod
    def _normalize_action(action: str) -> str:
        return action.strip().lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _rule_id(text: str, effective_date: str) -> str:
        digest = sha1(f"{text}:{effective_date}".encode()).hexdigest()[:12]
        return f"rule_{digest}"

    def materialize_from_proposal(
        self,
        proposal_text: str,
        *,
        proposer_id: str,
        approved: bool,
        proposal_record: dict[str, Any] | None = None,
        effective_date: str | None = None,
    ) -> dict[str, Any]:
        """Build and register executable rules from a governance proposal."""
        decided_at = effective_date or datetime.now(timezone.utc).isoformat()
        provenance: dict[str, Any] = {
            "proposer_id": proposer_id,
            "proposal_text": proposal_text,
            "approved": bool(approved),
            "decided_at": decided_at,
        }
        if isinstance(proposal_record, dict):
            provenance.update(proposal_record)

        if not approved:
            return {
                "decision": "rejected",
                "rule_ids": [],
                "effective_date": decided_at,
                "provenance": provenance,
                "reason": "proposal_rejected_by_vote",
            }

        materialized: list[GovernanceRule] = []
        text = proposal_text.strip()

        penalize_match = self._PENALIZE_PATTERN.search(text)
        if penalize_match:
            action = self._normalize_action(penalize_match.group("action"))
            amount = float(penalize_match.group("amount"))
            unit = penalize_match.group("unit").lower()
            penalty = RulePenalty(ip=amount if unit == "ip" else 0.0, du=amount if unit == "du" else 0.0)
            rule = GovernanceRule(
                rule_id=self._rule_id(text, decided_at),
                source_text=text,
                action_intent=action,
                effective_date=decided_at,
                decision_mode="penalize",
                penalty=penalty,
                provenance=provenance,
            )
            materialized.append(rule)

        ban_match = self._BAN_PATTERN.search(text)
        if ban_match:
            action = self._normalize_action(ban_match.group("action"))
            action = action.split("_", 1)[0] if "_" in action else action
            rule = GovernanceRule(
                rule_id=self._rule_id(f"deny:{text}", decided_at),
                source_text=text,
                action_intent=action,
                effective_date=decided_at,
                decision_mode="deny",
                provenance=provenance,
            )
            materialized.append(rule)

        if not materialized:
            return {
                "decision": "accepted",
                "rule_ids": [],
                "effective_date": decided_at,
                "provenance": provenance,
                "reason": "no_executable_constraint_materialized",
            }

        for rule in materialized:
            self._active_rules[rule.rule_id] = rule

        return {
            "decision": "accepted",
            "rule_ids": [rule.rule_id for rule in materialized],
            "effective_date": decided_at,
            "provenance": provenance,
        }

    def evaluate_action(self, action_intent: str) -> RuleEvaluationResult:
        """Evaluate ``action_intent`` against active executable rules."""
        normalized_action = self._normalize_action(action_intent or "idle")
        denials: list[str] = []
        penalties: list[dict[str, Any]] = []

        for rule in self._active_rules.values():
            rule.enforcement_stats["checked"] += 1
            if rule.action_intent != normalized_action:
                continue

            if rule.decision_mode == "deny":
                rule.enforcement_stats["rejected"] += 1
                denials.append(rule.rule_id)
                continue

            if rule.decision_mode == "penalize":
                rule.enforcement_stats["overridden"] += 1
                rule.enforcement_stats["penalties_applied"] += 1
                penalties.append(
                    {
                        "rule_id": rule.rule_id,
                        "ip": rule.penalty.ip,
                        "du": rule.penalty.du,
                        "reason": rule.penalty.reason,
                    }
                )

        if denials:
            return RuleEvaluationResult(
                allowed=False,
                decision="rejected",
                violated_rules=denials,
                reason="blocked_by_governance_rule",
            )

        if penalties:
            return RuleEvaluationResult(
                allowed=True,
                decision="overridden",
                violated_rules=[p["rule_id"] for p in penalties],
                penalties=penalties,
                reason="action_allowed_with_penalty",
            )

        for rule in self._active_rules.values():
            rule.enforcement_stats["accepted"] += 1

        return RuleEvaluationResult(
            allowed=True,
            decision="accepted",
            violated_rules=[],
        )

    def active_rules_read_model(self) -> list[dict[str, Any]]:
        """Return active rules and their enforcement statistics."""
        return [
            {
                "rule_id": rule.rule_id,
                "source_text": rule.source_text,
                "action_intent": rule.action_intent,
                "decision_mode": rule.decision_mode,
                "effective_date": rule.effective_date,
                "provenance": rule.provenance,
                "enforcement_stats": dict(rule.enforcement_stats),
            }
            for rule in self._active_rules.values()
        ]


governance_rules_engine = RulesEngine()

__all__ = [
    "GovernanceRule",
    "RuleEvaluationResult",
    "RulePenalty",
    "RulesEngine",
    "governance_rules_engine",
]
