"""Tests for docstore.retrieve."""

from unittest.mock import patch

from docstore.config import Config
from docstore.retrieve import retrieve
from docstore.models import RetrievalResult


FAKE_ROWS = [
    {
        "chunk_id": "id0",
        "text": "On-time completion rate 42%",
        "source_path": "/tmp/reports/q3.pdf",
        "page": 12,
        "bbox": (1.0, 2.0, 3.0, 4.0),
        "is_stat": True,
        "stat_verified": False,
        "score": 0.12,
    },
    {
        "chunk_id": "id1",
        "text": "Budget summary",
        "source_path": "/tmp/reports/q3.pdf",
        "page": 3,
        "bbox": (5.0, 6.0, 7.0, 8.0),
        "is_stat": False,
        "stat_verified": None,
        "score": 0.34,
    },
]


def test_retrieve_returns_citations():
    cfg = Config()
    with patch("docstore.retrieve.embed_query", return_value=[0.1, 0.2, 0.3]), \
         patch("docstore.retrieve.vector.open_store", return_value="TABLE"), \
         patch("docstore.retrieve.vector.search", return_value=FAKE_ROWS) as search, \
         patch("docstore.retrieve.vector.stored_embed_model", return_value="nomic-embed-text"):
        results = retrieve("what is the on-time rate?", k=2, cfg=cfg)
    search.assert_called_once()
    assert len(results) == 2
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert results[0].citation == "q3.pdf:p12"
    # unverified stat is surfaced
    assert results[0].is_stat is True
    assert results[0].stat_verified is False


def test_retrieve_embeds_the_query():
    cfg = Config(embed_model="nomic-embed-text")
    with patch("docstore.retrieve.embed_query", return_value=[0.1, 0.2, 0.3]) as eq, \
         patch("docstore.retrieve.vector.open_store", return_value="TABLE"), \
         patch("docstore.retrieve.vector.search", return_value=[]), \
         patch("docstore.retrieve.vector.stored_embed_model", return_value="nomic-embed-text"):
        retrieve("q", k=5, cfg=cfg)
    eq.assert_called_once_with("q", model="nomic-embed-text")
