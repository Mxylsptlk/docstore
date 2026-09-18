"""Thin Typer CLI over the docstore Python API.

The Python API is the primary interface; this wrapper just exposes `ingest` and `ask`
for quick terminal use. It adds no logic beyond arg parsing and printing a short summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from docstore.ask import ask as api_ask
from docstore.config import Config, load_config
from docstore.ingest import ingest as api_ingest

app = typer.Typer(help="Local PDF document-store: vision-based stats extraction + retrieval.")


def _load_cfg(config_path: Optional[str]) -> Config:
    if config_path:
        return load_config(config_path)
    default = Path("config.yaml")
    return load_config(default) if default.exists() else Config()


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to the PDF to ingest."),
    instruction: Optional[str] = typer.Option(
        None, "--instruction", "-i", help="What to extract (e.g. 'pull all statistics and labels')."
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Extraction backend: 'ollama' (default, local) or 'claude'."
    ),
    graph: bool = typer.Option(
        True, "--graph/--no-graph",
        help="Include this document in the knowledge graph (default on). Use --no-graph to exclude it.",
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml."),
):
    """Ingest a PDF into the local store."""
    cfg = _load_cfg(config)
    summary = api_ingest(path, instruction=instruction, backend=backend, cfg=cfg, graph=graph)
    typer.echo(f"Ingested {summary['doc_id']}: {summary['page_count']} pages "
               f"({summary['pages_vision']} via vision, {summary['pages_textonly']} text-only, "
               f"{summary['pages_skipped']} skipped)")
    typer.echo(f"  chunks: {summary['chunk_count']}  stats: {summary['stat_count']}  "
               f"unverified: {summary['unverified_count']}")
    if summary["graph_built"]:
        typer.echo(f"  graph: {summary['entity_count']} entities linked")
    else:
        typer.echo("  graph: excluded (--no-graph)")
    if summary["unverified_count"]:
        typer.echo(f"  ⚠ {summary['unverified_count']} statistic(s) unverified — confirm against source.")


@app.command()
def ask(
    query: str = typer.Argument(..., help="Your question."),
    k: int = typer.Option(5, "--k", help="Number of passages to retrieve."),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml."),
):
    """Retrieve relevant passages and write ANSWER.md for the assistant to read."""
    cfg = _load_cfg(config)
    out = api_ask(query, k=k, cfg=cfg)
    typer.echo(f"Wrote retrieval context to {out}")


if __name__ == "__main__":
    app()
