"""Utility script for inspecting council metrics from the stats store."""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from typing import Any

from src.agents.council.health import iter_health_warnings
from src.agents.council.orchestrator import CouncilOrchestrator


def _format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def _iter_member_lines(metrics: Mapping[str, Any]) -> Iterable[str]:
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


def _iter_pairwise_lines(metrics: Mapping[str, Any], *, show_flags: bool) -> Iterable[str]:
    for pair in metrics.get("pairwise", []):
        member_a = pair.get("member_a", "unknown")
        member_b = pair.get("member_b", "unknown")
        agreement_rate = _format_percentage(float(pair.get("agreement_rate", 0.0)))
        agreements = int(pair.get("agreements", 0))
        disagreements = int(pair.get("disagreements", 0))

        flagged = bool(
            pair.get("flagged")
            or pair.get("flagged_for_collusion")
            or pair.get("pairwise_flag")
            or pair.get("collusion_warning")
            or pair.get("warning")
        )
        label = " [FLAGGED]" if show_flags and flagged else ""
        base_line = (
            f"- {member_a} vs {member_b}: {agreement_rate} agreement{label} "
            f"({agreements} agreements / {disagreements} disagreements)"
        )

        if show_flags:
            note = pair.get("collusion_warning") or pair.get("warning")
            if note:
                yield f"{base_line} - {note}"
                continue
        yield base_line


def _collect_warnings(
    metrics: Mapping[str, Any], *, include_pairwise_flags: bool
) -> list[str]:
    warnings: list[str] = []

    raw_warnings = metrics.get("warnings", [])
    if isinstance(raw_warnings, Iterable):
        warnings.extend(str(warning) for warning in raw_warnings)

    collusion_warnings = metrics.get("collusion_warnings", [])
    if isinstance(collusion_warnings, Iterable):
        warnings.extend(str(warning) for warning in collusion_warnings)

    members = metrics.get("members", [])
    pairwise = metrics.get("pairwise", [])

    if not members:
        warnings.append("No member statistics available.")
    else:
        for member in members:
            if int(member.get("participations", 0)) <= 0:
                member_id = member.get("member_id", "unknown")
                warnings.append(f"Member {member_id} has no recorded participations.")
        warnings.extend(iter_health_warnings(members))

    if not pairwise:
        warnings.append("No pairwise agreement metrics available.")

    if include_pairwise_flags:
        for pair in pairwise:
            flagged = bool(
                pair.get("flagged")
                or pair.get("flagged_for_collusion")
                or pair.get("pairwise_flag")
            )
            if not flagged:
                continue
            member_a = pair.get("member_a", "unknown")
            member_b = pair.get("member_b", "unknown")
            reason = pair.get("collusion_warning") or pair.get("warning")
            if reason:
                warnings.append(str(reason))
            else:
                warnings.append(
                    f"Pair {member_a} vs {member_b} flagged for follow-up."
                )

    return warnings


def _print_section(title: str, lines: Iterable[str]) -> None:
    print(title)
    printed = False
    for line in lines:
        print(line)
        printed = True
    if not printed:
        print("- None")


def _render_metrics_block(metrics: Mapping[str, Any], *, show_flags: bool) -> None:
    _print_section("Member Win Rates:", _iter_member_lines(metrics))
    print("\nPairwise Agreement Rates:")
    for line in _iter_pairwise_lines(metrics, show_flags=show_flags):
        print(line)

    warnings = _collect_warnings(metrics, include_pairwise_flags=show_flags)
    print("\nWarnings:")
    if warnings:
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("- None")


def _iter_questions(
    metrics: Mapping[str, Any], question_filter: str | None
) -> Iterable[Mapping[str, Any]]:
    questions = metrics.get("questions", [])
    if not isinstance(questions, Iterable):
        return []

    if question_filter:
        return [
            question
            for question in questions
            if isinstance(question, Mapping)
            and question.get("question_id") == question_filter
        ]

    return [question for question in questions if isinstance(question, Mapping)]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect council metrics and agreements")
    parser.add_argument(
        "--question-id",
        help="Filter metrics to a specific question identifier when available.",
    )
    parser.add_argument(
        "--show-collusion-flags",
        action="store_true",
        help="Surface collusion warnings and flagged pairwise agreements.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    orchestrator = CouncilOrchestrator()
    metrics = orchestrator.serialize_metrics(question_id=args.question_id)

    print("Council Metrics")
    print("================")
    _render_metrics_block(metrics, show_flags=args.show_collusion_flags)

    for question_metrics in _iter_questions(metrics, args.question_id):
        question_id = question_metrics.get("question_id", "unknown-question")
        prompt = question_metrics.get("prompt")

        print("\n---")
        print(f"Question: {question_id}")
        if prompt:
            print(f"Prompt: {prompt}")
        _render_metrics_block(question_metrics, show_flags=args.show_collusion_flags)


if __name__ == "__main__":
    main()
