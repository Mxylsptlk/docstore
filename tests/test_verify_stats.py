"""Tests for docstore.verify_stats."""

from pathlib import Path
from unittest.mock import patch

from docstore.config import Config
from docstore.models import Stat
from docstore.verify_stats import normalize_number, verify_stats

FIX = Path(__file__).parent / "fixtures"


def test_normalize_number():
    assert normalize_number("$4.2M") == normalize_number("4.2M")
    assert normalize_number(" 42% ") == normalize_number("42%")
    assert normalize_number("1,204") == normalize_number("1204")
    assert normalize_number("42%") != normalize_number("43%")


def test_verify_confirms_match():
    cfg = Config(verify_stats=True)
    stats = [Stat(value_text="42%", label="on-time", page=2, bbox=(72, 100, 200, 160))]
    # region re-read returns the same number
    with patch("docstore.verify_stats._reread_number", return_value="42%"):
        out = verify_stats(FIX / "sample_layout.pdf", stats, cfg)
    assert out[0].verified is True
    assert out[0].reread_value == "42%"


def test_verify_flags_mismatch():
    cfg = Config(verify_stats=True)
    stats = [Stat(value_text="42%", label="on-time", page=2, bbox=(72, 100, 200, 160))]
    with patch("docstore.verify_stats._reread_number", return_value="24%"):
        out = verify_stats(FIX / "sample_layout.pdf", stats, cfg)
    assert out[0].verified is False
    assert out[0].reread_value == "24%"


def test_verify_unverifiable_without_bbox():
    cfg = Config(verify_stats=True)
    stats = [Stat(value_text="17", label="crews", page=1, bbox=None)]
    with patch("docstore.verify_stats._reread_number") as reread:
        out = verify_stats(FIX / "sample_layout.pdf", stats, cfg)
    reread.assert_not_called()  # nothing to crop
    assert out[0].verified is None


def test_verify_auto_off_for_ollama_backend():
    # AUTO (verify_stats unset) + ollama backend -> no re-read, stat stays unverifiable.
    cfg = Config(extraction_backend="ollama")  # verify_stats defaults to None (auto)
    stats = [Stat(value_text="42%", label="on-time", page=2, bbox=(72, 100, 200, 160))]
    with patch("docstore.verify_stats._reread_number") as reread:
        out = verify_stats(FIX / "sample_layout.pdf", stats, cfg)
    reread.assert_not_called()
    assert out[0].verified is None


def test_verify_disabled_by_config():
    cfg = Config(verify_stats=False)
    stats = [Stat(value_text="42%", label="on-time", page=2, bbox=(72, 100, 200, 160))]
    with patch("docstore.verify_stats._reread_number") as reread:
        out = verify_stats(FIX / "sample_layout.pdf", stats, cfg)
    reread.assert_not_called()
    assert out[0].verified is None


def test_verify_degenerate_bbox_is_unverifiable_not_crash():
    # Regression: a zero-area bbox from the extractor must NOT crash the ingest; the stat
    # stays unverifiable (verified=None). Uses the real render path (no mock) to reproduce.
    cfg = Config(verify_stats=True)
    stats = [Stat(value_text="42%", label="on-time", page=2, bbox=(72, 100, 72, 100))]
    out = verify_stats(FIX / "sample_layout.pdf", stats, cfg)
    assert out[0].verified is None
    assert out[0].reread_value is None
