#!/usr/bin/env python3
"""Command-line interface for running council deliberations."""

from __future__ import annotations

import json

import typer

from src.agents.council.orchestrator import run_council
from src.agents.council.types import CouncilQuestion

app = typer.Typer(add_completion=False)


@app.command()
def main(
    prompt: str | None = typer.Argument(
        None, help="Question to pose to the council"
    ),
    question: str | None = typer.Option(
        None,
        "--question",
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
    question_id: str = typer.Option(
        "cli-question",
        "--question-id",
        "-q",
        help="Identifier for the question posed to the council",
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

    rag_docs = list(rag_doc or [])
    question = CouncilQuestion(
        question_id=question_id,
        prompt=resolved_prompt,
        context=context,
        rag_documents=rag_docs,
    )

    outcome = run_council(
        question,
        extra_context=context,
        rag_docs=rag_docs,
        allow_disabled_mode=bypass_env_guard,
    )

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
        winners = ", ".join(outcome.winning_member_ids)
        typer.echo(f"Winners: {winners}")

    if outcome.answers:
        answers = outcome.answers
        if not show_all and outcome.winning_member_ids:
            winner_ids = set(outcome.winning_member_ids)
            answers = [
                answer for answer in outcome.answers if answer.member_id in winner_ids
            ]

        if answers:
            typer.echo("Answers:")
            for answer in answers:
                typer.echo(f"- {answer.member_id}: {answer.answer}")

    if show_votes:
        if outcome.votes:
            typer.echo("Judge Scores:")
            for member_id, score in outcome.votes.items():
                typer.echo(f"- {member_id}: {score}")

        votes_to_display = [
            (answer.member_id, answer.votes) for answer in outcome.answers if answer.votes
        ]
        if votes_to_display:
            typer.echo("Votes:")
            for member_id, votes in votes_to_display:
                typer.echo(f"- {member_id}:")
                for target_member, score in votes.items():
                    typer.echo(f"  - {target_member}: {score}")

    if show_metrics and outcome.metrics:
        typer.echo("Metrics:")
        typer.echo(json.dumps(outcome.metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover - manual tool
    app()
