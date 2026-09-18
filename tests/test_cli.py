"""Tests for docstore.cli (thin Typer wrapper over the Python API)."""

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from docstore.cli import app

runner = CliRunner()


def _summary(**overrides):
    base = {
        "doc_id": "f-1234", "page_count": 2, "pages_vision": 1, "pages_textonly": 1,
        "pages_skipped": 0, "chunk_count": 3, "stat_count": 2, "unverified_count": 1,
        "graph_built": True, "entity_count": 4,
    }
    base.update(overrides)
    return base


def test_cli_ingest_calls_api(tmp_path: Path):
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    with patch("docstore.cli.api_ingest", return_value=_summary()) as m:
        result = runner.invoke(app, ["ingest", str(pdf), "--instruction", "pull stats"])
    assert result.exit_code == 0, result.output
    m.assert_called_once()
    # graph defaults on -> API called with graph=True
    _, kwargs = m.call_args
    assert kwargs.get("graph") is True
    # unverified count + entity count surfaced to the user
    assert "unverified" in result.output.lower()
    assert "entities" in result.output.lower()


def test_cli_ask_calls_api(tmp_path: Path):
    answer = tmp_path / "ANSWER.md"
    answer.write_text("stub")
    with patch("docstore.cli.api_ask", return_value=str(answer)) as m:
        result = runner.invoke(app, ["ask", "what is the on-time rate?"])
    assert result.exit_code == 0, result.output
    m.assert_called_once()
    assert str(answer) in result.output


def test_cli_ingest_passes_backend(tmp_path: Path):
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    with patch("docstore.cli.api_ingest",
               return_value=_summary(doc_id="x", page_count=1, pages_vision=1,
                                     pages_textonly=0, chunk_count=1, stat_count=0,
                                     unverified_count=0)) as m:
        result = runner.invoke(app, ["ingest", str(pdf), "--backend", "ollama"])
    assert result.exit_code == 0, result.output
    _, kwargs = m.call_args
    assert kwargs.get("backend") == "ollama"


def test_cli_no_graph_flag(tmp_path: Path):
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    with patch("docstore.cli.api_ingest",
               return_value=_summary(graph_built=False, entity_count=0,
                                     unverified_count=0)) as m:
        result = runner.invoke(app, ["ingest", str(pdf), "--no-graph"])
    assert result.exit_code == 0, result.output
    _, kwargs = m.call_args
    assert kwargs.get("graph") is False
    assert "excluded" in result.output.lower()
