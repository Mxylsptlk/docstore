"""Tests for docstore.vision_extract (pluggable vision extraction)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from docstore.vision_extract import extract_from_image
from docstore.models import Stat

PNG = b"\x89PNG\r\n\x1a\nFAKE"

RESP = {
    "curated_text": "On-time completion rate: 42%",
    "stats": [{"value_text": "42%", "label": "on-time completion rate", "bbox": [72, 100, 200, 140]}],
}


def _claude_message(payload: dict) -> MagicMock:
    msg = MagicMock()
    block = MagicMock()
    block.text = json.dumps(payload)
    msg.content = [block]
    return msg


def test_vision_dispatch_claude(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _claude_message(RESP)
    with patch("docstore.vision_extract.anthropic.Anthropic", return_value=fake_client) as ctor:
        text, stats = extract_from_image(PNG, "pull all stats", backend="claude", model="claude-x")
    ctor.assert_called_once()
    fake_client.messages.create.assert_called_once()
    assert "42%" in text
    assert len(stats) == 1 and isinstance(stats[0], Stat)
    assert stats[0].value_text == "42%"


def test_claude_requires_env_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        extract_from_image(PNG, "pull all stats", backend="claude", model="claude-x")


def test_vision_dispatch_ollama():
    fake_resp = {"message": {"content": json.dumps(RESP)}}
    with patch("docstore.vision_extract.ollama.chat", return_value=fake_resp) as chat:
        text, stats = extract_from_image(PNG, "pull all stats", backend="ollama", model="qwen3-vl:2b")
    chat.assert_called_once()
    _, kwargs = chat.call_args
    # image must ride INSIDE the user message (ollama.chat has no top-level images kwarg)
    assert "images" not in kwargs
    user_msg = kwargs["messages"][-1]
    assert user_msg["images"], "image not attached to the user message"
    assert kwargs["model"] == "qwen3-vl:2b"
    assert "42%" in text
    assert stats[0].label == "on-time completion rate"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        extract_from_image(PNG, "x", backend="foo", model="m")


def test_tolerates_non_json_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake_client = MagicMock()
    # Model returns prose, not JSON — we should still get curated_text back, no crash.
    msg = MagicMock()
    block = MagicMock()
    block.text = "The on-time rate is 42%."
    msg.content = [block]
    fake_client.messages.create.return_value = msg
    with patch("docstore.vision_extract.anthropic.Anthropic", return_value=fake_client):
        text, stats = extract_from_image(PNG, "pull", backend="claude", model="claude-x")
    assert "42%" in text
    assert stats == []
