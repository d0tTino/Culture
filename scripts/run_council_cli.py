#!/usr/bin/env python
"""Run a single Council Mode deliberation from the command line."""

from __future__ import annotations

import argparse
import os
import re
import sys
import uuid
from collections.abc import Mapping
from typing import Any

# Ensure project root is on the import path when executed from scripts/
sys.path.append(str(os.path.dirname(os.path.dirname(__file__))))

from src.agents.council.orchestrator import run_council
from src.agents.council.types import CouncilOutcome, CouncilQuestion
from src.infra import config as infra_config
from src.infra.config import get_config, load_council_config


def _slugify(value: str) -> str:
    """Convert display names into predictable member identifiers."""

    lowered = value.strip().lower()
    sanitized = re.sub(r"[^a-z0-9]+", "-", lowered)
    return sanitized.strip("-") or "member"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the council runner."""

    parser = argparse.ArgumentParser(description="Run a Council Mode question")
    parser.add_argument(
        "--question",
        "-q",
        required=True,
        help="Question to ask the council",
    )
    parser.add_argument(
        "--show-all",
        "-a",
        action="store_true",
        help="Show all member answers instead of only the winner",
    )
    return parser.parse_args()


def _build_member_lookup(raw_config: Mapping[str, Any]) -> dict[str, str]:
    """Return a mapping of member_id -> display name using the shared loader rules."""

    lookup: dict[str, str] = {}
    members = raw_config.get("members")
    if not isinstance(members, list):
        return lookup

    for index, entry in enumerate(members):
        if not isinstance(entry, Mapping):
            continue
        display_name = str(
            entry.get("display_name")
            or entry.get("name")
            or entry.get("id")
            or f"Member {index + 1}"
        )
        member_id = str(entry.get("id") or _slugify(display_name))
        lookup[member_id] = display_name
    return lookup


def _format_metrics(outcome: CouncilOutcome) -> str:
    metrics = {}
    if isinstance(outcome.metadata, Mapping):
        metrics = outcome.metadata.get("metrics", {}) or {}
    if not metrics:
        return "No metrics reported."

    lines = ["Key metrics:"]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _format_answers(outcome: CouncilOutcome, members: Mapping[str, str], show_all: bool) -> str:
    answers = outcome.answers
    if not show_all and outcome.winning_member_ids:
        winner_ids = set(outcome.winning_member_ids)
        answers = [answer for answer in answers if answer.member_id in winner_ids]

    if not answers:
        return "No answers were returned by the council."

    lines: list[str] = []
    for answer in answers:
        name = members.get(answer.member_id, answer.member_id)
        lines.append(f"- {name} ({answer.member_id})")
        if answer.confidence is not None:
            lines.append(f"  Confidence: {answer.confidence}")
        if answer.reasoning:
            lines.append(f"  Reasoning: {answer.reasoning}")
        lines.append(f"  Answer: {answer.answer}")
    return "\n".join(lines)


def _print_outcome(outcome: CouncilOutcome, members: Mapping[str, str], show_all: bool) -> None:
    if outcome.winning_member_ids:
        winner_id = outcome.winning_member_ids[0]
        winner_name = members.get(winner_id, winner_id)
        print(f"Winner: {winner_name} ({winner_id})")
    else:
        print("No consensus winner was selected.")

    if outcome.summary:
        print(f"\nCouncil summary:\n{outcome.summary}")

    print(f"\nAnswers:\n{_format_answers(outcome, members, show_all)}")
    print(f"\n{_format_metrics(outcome)}")


def main() -> None:
    args = parse_args()

    infra_config.load_config(validate_required=False)
    if not bool(get_config("USE_COUNCIL_MODE")):
        print("Council Mode is disabled. Set USE_COUNCIL_MODE=true to run this command.")
        raise SystemExit(1)

    council_config = load_council_config()
    if not bool(council_config.get("enabled", True)):
        print("Council Mode configuration is disabled. Enable it in the council config file.")
        raise SystemExit(1)

    members = _build_member_lookup(council_config)
    question = CouncilQuestion(question_id=str(uuid.uuid4()), prompt=args.question)
    outcome = run_council(question)

    _print_outcome(outcome, members, args.show_all)


if __name__ == "__main__":
    main()
