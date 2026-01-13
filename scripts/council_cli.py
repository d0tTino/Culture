#!/usr/bin/env python3
"""Command-line interface for running council deliberations."""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from typing import Any

import typer

from src.agents.council.orchestrator import run_council
from src.agents.council.types import CouncilOutcome, CouncilQuestion
from src.infra import config as infra_config
from src.infra.config import get_config, load_council_config

app = typer.Typer(add_completion=False)


def _slugify(value: str) -> str:
    """Convert display names into predictable member identifiers."""

    lowered = value.strip().lower()
    sanitized = re.sub(r"[^a-z0-9]+", "-", lowered)
    return sanitized.strip("-") or "member"


def _build_member_lookup(raw_config: Mapping[str, Any] | None) -> dict[str, str]:
    """Return a mapping of member_id -> display name using the shared loader rules."""

    lookup: dict[str, str] = {}
    members = raw_config.get("members") if isinstance(raw_config, Mapping) else None
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


def _format_metrics(outcome_metrics: Mapping[str, Any]) -> str:
    if not outcome_metrics:
        return "No metrics reported."

    lines = ["Key metrics:"]
    for key, value in outcome_metrics.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _format_answers(
    outcome: CouncilOutcome, members: Mapping[str, str], show_all: bool
) -> str:
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


def _load_council_settings() -> tuple[Mapping[str, Any], dict[str, str]]:
    infra_config.load_config(validate_required=False)
    council_config = load_council_config()
    members = _build_member_lookup(council_config)
    return council_config, members


def _ensure_council_enabled(
    council_config: Mapping[str, Any], bypass_env_guard: bool
) -> None:
    if bypass_env_guard:
        return

    if not bool(get_config("USE_COUNCIL_MODE")):
        typer.echo("Council Mode is disabled. Set USE_COUNCIL_MODE=true to run this command.")
        raise typer.Exit(code=1)

    if not bool(council_config.get("enabled", True)):
        typer.echo("Council Mode configuration is disabled. Enable it in the council config file.")
        raise typer.Exit(code=1)


def _collect_metrics(outcome: CouncilOutcome) -> Mapping[str, Any]:
    if outcome.metrics:
        return outcome.metrics
    if isinstance(outcome.metadata, Mapping):
        metrics = outcome.metadata.get("metrics")
        if isinstance(metrics, Mapping):
            return metrics
    return {}


def _print_outcome(
    outcome: CouncilOutcome,
    members: Mapping[str, str],
    *,
    show_all: bool,
    show_votes: bool,
    show_metrics: bool,
) -> None:
    typer.echo(f"Question: {outcome.question.prompt}")
    if outcome.question.context:
        typer.echo(f"Context: {outcome.question.context}")
    if outcome.question.rag_documents:
        typer.echo("RAG Docs:")
        for doc in outcome.question.rag_documents:
            typer.echo(f"- {doc}")

    typer.echo(f"Resolution: {outcome.resolution}")

    if outcome.summary:
        typer.echo(f"Summary: {outcome.summary}")

    if outcome.winning_member_ids:
        winners = ", ".join(
            f"{members.get(member_id, member_id)} ({member_id})"
            for member_id in outcome.winning_member_ids
        )
        typer.echo(f"Winners: {winners}")
    else:
        typer.echo("No consensus winner was selected.")

    typer.echo("Answers:")
    typer.echo(_format_answers(outcome, members, show_all))

    if show_votes:
        if outcome.votes:
            typer.echo("Judge Scores:")
            for member_id, score in outcome.votes.items():
                name = members.get(member_id, member_id)
                typer.echo(f"- {name} ({member_id}): {score}")

        votes_to_display = [
            (answer.member_id, answer.votes)
            for answer in outcome.answers
            if answer.votes
        ]
        if votes_to_display:
            typer.echo("Votes:")
            for member_id, votes in votes_to_display:
                typer.echo(f"- {members.get(member_id, member_id)} ({member_id}):")
                for target_member, score in votes.items():
                    name = members.get(target_member, target_member)
                    typer.echo(f"  - {name} ({target_member}): {score}")

    if show_metrics:
        metrics = _collect_metrics(outcome)
        typer.echo(_format_metrics(metrics))


@app.command()
def main(
    prompt: str | None = typer.Argument(
        None, help="Question to pose to the council"
    ),
    question: str | None = typer.Option(
        None,
        "--question",
        "-q",
        help="Question to pose to the council (alias for the prompt argument)",
    ),
    context: str | None = typer.Option(None, "--context", "-c", help="Optional context"),
    rag_doc: list[str] | None = typer.Option(
        None,
        "--rag-doc",
        "--rag-docs",
        help="Additional RAG documents to supply to the council",
    ),
    show_all: bool = typer.Option(
        False,
        "--show-all",
        "-a",
        help="Print answers from all members instead of only winners",
    ),
    show_votes: bool = typer.Option(
        False,
        "--show-votes",
        help="Display per-member votes and judge score summaries",
    ),
    show_metrics: bool = typer.Option(
        False,
        "--show-metrics",
        help="Display fitness or collusion metrics from the council outcome",
    ),
    bypass_env_guard: bool = typer.Option(
        True,
        "--bypass-env-guard/--enforce-env-guard",
        help=(
            "Allow the council to run even when USE_COUNCIL_MODE is false. Disable to "
            "respect the environment guard."
        ),
    ),
    question_id: str | None = typer.Option(
        None,
        "--question-id",
        help="Identifier for the question posed to the council",
    ),
    agent_id: str | None = typer.Option(
        None,
        "--agent-id",
        help="Identifier for the agent making the request (used for RAG lookups)",
    ),
) -> None:
    """Run the council with the provided ``prompt`` and display the outcome."""

    if prompt and question and prompt != question:
        raise typer.BadParameter(
            "Provide the question either as the prompt argument or via --question, not both."
        )

    resolved_prompt = question or prompt
    if not resolved_prompt:
        raise typer.BadParameter(
            "A question is required. Provide the prompt argument or --question."
        )

    council_config, members = _load_council_settings()
    _ensure_council_enabled(council_config, bypass_env_guard)

    rag_docs = list(rag_doc or [])
    resolved_question_id = question_id or str(uuid.uuid4())
    resolved_agent_id = agent_id or question_id or "council-cli"
    question = CouncilQuestion(
        question_id=resolved_question_id,
        question=resolved_prompt,
        context=context,
        rag_documents=rag_docs,
        metadata={"agent_id": resolved_agent_id},
    )

    outcome = run_council(
        question,
        extra_context=question.extra_context,
        rag_docs=rag_docs,
        allow_disabled_mode=bypass_env_guard,
    )

    _print_outcome(
        outcome,
        members,
        show_all=show_all,
        show_votes=show_votes,
        show_metrics=show_metrics,
    )


if __name__ == "__main__":  # pragma: no cover - manual tool
    app()
