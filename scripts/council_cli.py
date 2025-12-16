#!/usr/bin/env python3
"""Command-line interface for running council deliberations."""

from __future__ import annotations

import typer

from src.agents.council.orchestrator import run_council
from src.agents.council.types import CouncilQuestion

app = typer.Typer(add_completion=False)


@app.command()
def main(
    prompt: str = typer.Argument(..., help="Question to pose to the council"),
    context: str | None = typer.Option(None, "--context", "-c", help="Optional context"),
    rag_doc: list[str] | None = typer.Option(
        None,
        "--rag-doc",
        "--rag-docs",
        help="Additional RAG documents to supply to the council",
    ),
    question_id: str = typer.Option(
        "cli-question",
        "--question-id",
        "-q",
        help="Identifier for the question posed to the council",
    ),
) -> None:
    """Run the council with the provided ``prompt`` and display the outcome."""

    rag_docs = list(rag_doc or [])
    question = CouncilQuestion(
        question_id=question_id,
        prompt=prompt,
        context=context,
        rag_documents=rag_docs,
    )

    outcome = run_council(question, extra_context=context, rag_docs=rag_docs)

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
        typer.echo("Answers:")
        for answer in outcome.answers:
            typer.echo(f"- {answer.member_id}: {answer.answer}")


if __name__ == "__main__":  # pragma: no cover - manual tool
    app()
