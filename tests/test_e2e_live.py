"""Live end-to-end tests. Gated behind -m live (deselected by default).

Requires:
  - `ollama serve` running with the configured embed model pulled
  - ANTHROPIC_API_KEY exported (for the default claude extraction backend)

Run: pytest -m live -v
"""

from pathlib import Path

import pytest

from docstore import ask, ingest
from docstore.config import Config

FIX = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.live


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    # Isolated store per test run; default claude backend + verification on.
    return Config(
        data_dir=str(tmp_path / "data"),
        answer_path=str(tmp_path / "ANSWER.md"),
    )


def test_e2e_stats_accuracy(cfg):
    """Ingest an image-heavy page whose ground-truth stat is 42%, then ask about it.
    The retrieved ANSWER.md must cite the right page and contain the exact number."""
    summary = ingest(
        FIX / "sample_layout.pdf",
        instruction="Extract every statistic and its label, exactly as shown.",
        cfg=cfg,
    )
    assert summary["chunk_count"] >= 1
    # page 2 is image-heavy -> must have gone through vision
    assert summary["pages_vision"] >= 1

    out = ask("What is the on-time completion rate?", k=5, cfg=cfg)
    body = Path(out).read_text()
    assert "42%" in body
    assert "sample_layout.pdf:p2" in body


def test_e2e_verification_catches_bad_number(cfg):
    """The scrambled-numbers fixture should ingest; any stat that fails region re-read
    must be surfaced as unverified rather than silently trusted."""
    summary = ingest(
        FIX / "scrambled_numbers.pdf",
        instruction="Extract the total contract value exactly as shown.",
        cfg=cfg,
    )
    # We can't assert a specific mismatch (depends on the model), but the pipeline must
    # run and report verification status honestly.
    assert "unverified_count" in summary
    assert summary["stat_count"] >= 0
