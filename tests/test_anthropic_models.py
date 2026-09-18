"""Tests for docstore.anthropic_models (model 'latest' resolution)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import docstore.anthropic_models as am


def setup_function(_):
    am.clear_cache()


def _model(id_, created_at, display_name=""):
    return SimpleNamespace(id=id_, created_at=created_at, display_name=display_name or id_)


def test_explicit_id_passes_through():
    # A concrete id must be returned unchanged and must NOT hit the API.
    with patch("anthropic.Anthropic") as ctor:
        assert am.resolve_model("claude-sonnet-5") == "claude-sonnet-5"
        assert am.resolve_model("claude-sonnet-5-20250101") == "claude-sonnet-5-20250101"
    ctor.assert_not_called()


def test_latest_resolves_newest_sonnet(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake = MagicMock()
    fake.models.list.return_value = [
        _model("claude-sonnet-4-20240101", "2024-01-01T00:00:00Z"),
        _model("claude-sonnet-5-20250601", "2025-06-01T00:00:00Z"),
        _model("claude-opus-4-20250101", "2025-01-01T00:00:00Z"),
        _model("claude-haiku-4-20250101", "2025-01-01T00:00:00Z"),
    ]
    with patch("anthropic.Anthropic", return_value=fake):
        assert am.resolve_model("latest") == "claude-sonnet-5-20250601"


def test_latest_family_selector(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake = MagicMock()
    fake.models.list.return_value = [
        _model("claude-sonnet-5-20250601", "2025-06-01T00:00:00Z"),
        _model("claude-opus-4-20250101", "2025-01-01T00:00:00Z"),
        _model("claude-opus-5-20250701", "2025-07-01T00:00:00Z"),
    ]
    with patch("anthropic.Anthropic", return_value=fake):
        assert am.resolve_model("latest:opus") == "claude-opus-5-20250701"


def test_result_is_cached(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake = MagicMock()
    fake.models.list.return_value = [_model("claude-sonnet-5-x", "2025-06-01T00:00:00Z")]
    with patch("anthropic.Anthropic", return_value=fake):
        am.resolve_model("latest")
        am.resolve_model("latest")
    fake.models.list.assert_called_once()  # second call served from cache


def test_falls_back_when_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # No key -> can't query; must fall back to the pinned Sonnet id, not crash.
    assert am.resolve_model("latest") == am._FALLBACK["sonnet"]


def test_falls_back_on_api_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake = MagicMock()
    fake.models.list.side_effect = RuntimeError("network down")
    with patch("anthropic.Anthropic", return_value=fake):
        assert am.resolve_model("latest") == am._FALLBACK["sonnet"]


def test_falls_back_when_family_absent(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake = MagicMock()
    fake.models.list.return_value = [_model("claude-opus-4-x", "2025-01-01T00:00:00Z")]
    with patch("anthropic.Anthropic", return_value=fake):
        # No sonnet in the list -> pinned sonnet fallback.
        assert am.resolve_model("latest:sonnet") == am._FALLBACK["sonnet"]
