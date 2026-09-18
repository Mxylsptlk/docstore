"""Tests for docstore.ask (writes ANSWER.md)."""

from pathlib import Path
from unittest.mock import patch

from docstore.config import Config
from docstore.ask import ask
from docstore.models import RetrievalResult


def _results():
    return [
        RetrievalResult(
            chunk_id="id0", text="On-time completion rate 42%",
            source_path="/tmp/reports/q3.pdf", page=12, bbox=(1, 2, 3, 4),
            is_stat=True, stat_verified=True, score=0.1,
        ),
        RetrievalResult(
            chunk_id="id1", text="Change-order rate 8%",
            source_path="/tmp/reports/q3.pdf", page=5, bbox=(1, 2, 3, 4),
            is_stat=True, stat_verified=False, score=0.2,
        ),
    ]


def test_ask_writes_answer_md(tmp_path: Path):
    cfg = Config(answer_path=str(tmp_path / "ANSWER.md"))
    with patch("docstore.ask.retrieve", return_value=_results()) as r:
        out = ask("what is the on-time rate?", k=2, cfg=cfg)
    r.assert_called_once()
    p = Path(out)
    assert p.exists()
    body = p.read_text()
    assert "what is the on-time rate?" in body
    assert "q3.pdf:p12" in body
    assert "On-time completion rate 42%" in body


def test_ask_flags_unverified_stat(tmp_path: Path):
    cfg = Config(answer_path=str(tmp_path / "ANSWER.md"))
    with patch("docstore.ask.retrieve", return_value=_results()):
        out = ask("q", k=2, cfg=cfg)
    body = Path(out).read_text()
    # the failed-verification stat must carry a visible warning
    assert "unverified" in body.lower() or "⚠" in body
    # instructions to the assistant present
    assert "cite" in body.lower()


def test_ask_handles_no_results(tmp_path: Path):
    cfg = Config(answer_path=str(tmp_path / "ANSWER.md"))
    with patch("docstore.ask.retrieve", return_value=[]):
        out = ask("q", k=2, cfg=cfg)
    body = Path(out).read_text()
    assert "no " in body.lower()  # "no relevant passages" / "no results"
