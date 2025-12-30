"""Utility script for inspecting council metrics from the stats store."""
from __future__ import annotations

from collections.abc import Iterable

from src.agents.council.orchestrator import CouncilOrchestrator


def _format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def _iter_member_lines(metrics: dict[str, list[dict[str, float]]]) -> Iterable[str]:
    for member in metrics.get("members", []):
        member_id = member.get("member_id", "unknown")
        win_rate = _format_percentage(float(member.get("win_rate", 0.0)))
        wins = int(member.get("wins", 0))
        participations = int(member.get("participations", 0))
        avg_confidence = float(member.get("avg_confidence", 0.0))
        yield (
            f"- {member_id}: {win_rate} win rate "
            f"({wins}/{participations} wins, avg confidence {avg_confidence:.2f})"
        )


def _iter_pairwise_lines(metrics: dict[str, list[dict[str, float]]]) -> Iterable[str]:
    for pair in metrics.get("pairwise", []):
        member_a = pair.get("member_a", "unknown")
        member_b = pair.get("member_b", "unknown")
        agreement_rate = _format_percentage(float(pair.get("agreement_rate", 0.0)))
        agreements = int(pair.get("agreements", 0))
        disagreements = int(pair.get("disagreements", 0))
        yield (
            f"- {member_a} vs {member_b}: {agreement_rate} agreement "
            f"({agreements} agreements / {disagreements} disagreements)"
        )


def _collect_warnings(metrics: dict[str, list[dict[str, float]]]) -> list[str]:
    warnings: list[str] = []

    members = metrics.get("members", [])
    pairwise = metrics.get("pairwise", [])

    if not members:
        warnings.append("No member statistics available.")
    else:
        for member in members:
            if int(member.get("participations", 0)) <= 0:
                member_id = member.get("member_id", "unknown")
                warnings.append(f"Member {member_id} has no recorded participations.")

    if not pairwise:
        warnings.append("No pairwise agreement metrics available.")

    return warnings


def main() -> None:
    orchestrator = CouncilOrchestrator()
    metrics = orchestrator.serialize_metrics()

    print("Member Win Rates:")
    for line in _iter_member_lines(metrics):
        print(line)

    print("\nPairwise Agreement Rates:")
    for line in _iter_pairwise_lines(metrics):
        print(line)

    warnings = _collect_warnings(metrics)
    print("\nWarnings:")
    if warnings:
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("- None")


if __name__ == "__main__":
    main()
