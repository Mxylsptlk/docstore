"""Tests for docstore.embed (Ollama embeddings)."""

from unittest.mock import patch

import pytest

from docstore.embed import embed_query, embed_texts


def _fake_embeddings(model, prompt):
    # deterministic pseudo-embedding based on text length
    n = len(prompt) % 5 + 1
    return {"embedding": [float(n)] * 8}


def test_embed_batch_mocked():
    with patch("docstore.embed.ollama.embeddings", side_effect=_fake_embeddings):
        vecs = embed_texts(["hello", "a longer sentence here"], model="nomic-embed-text")
    assert len(vecs) == 2
    assert all(len(v) == 8 for v in vecs)


def test_embed_query_mocked():
    with patch("docstore.embed.ollama.embeddings", side_effect=_fake_embeddings):
        v = embed_query("what is the on-time rate?", model="nomic-embed-text")
    assert len(v) == 8


def test_embed_empty_list():
    with patch("docstore.embed.ollama.embeddings", side_effect=_fake_embeddings) as m:
        vecs = embed_texts([], model="nomic-embed-text")
    assert vecs == []
    m.assert_not_called()


def test_embed_surfaces_connection_error():
    with patch("docstore.embed.ollama.embeddings", side_effect=ConnectionError("no server")):
        with pytest.raises(RuntimeError, match="Ollama"):
            embed_texts(["x"], model="nomic-embed-text")
