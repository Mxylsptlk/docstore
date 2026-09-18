"""Tests for docstore.stores.vector (LanceDB)."""

from pathlib import Path

import pytest

from docstore.config import Config
from docstore.models import Chunk
from docstore.stores import vector


def _chunk(i: int, text: str, page: int = 1, is_stat: bool = False, verified=None) -> Chunk:
    return Chunk(
        chunk_id=f"id{i}",
        doc_id="doc1",
        source_path="/tmp/f.pdf",
        page=page,
        bbox=(0.0, float(i), 10.0, float(i) + 5),
        section="S",
        char_start=i * 10,
        char_end=i * 10 + len(text),
        text=text,
        is_stat=is_stat,
        stat_verified=verified,
    )


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(data_dir=str(tmp_path / "data"), table_name="chunks")


def test_upsert_and_search(cfg):
    store = vector.open_store(cfg, dim=3)
    chunks = [
        _chunk(0, "on-time completion rate 42%", is_stat=True, verified=True),
        _chunk(1, "budget summary total value"),
        _chunk(2, "crew scheduling notes"),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    vector.upsert_chunks(store, chunks, vectors)

    hits = vector.search(store, [0.9, 0.1, 0.0], k=2)
    assert len(hits) == 2
    # nearest should be the first chunk
    top = hits[0]
    assert top["chunk_id"] == "id0"
    assert top["is_stat"] is True
    assert top["stat_verified"] is True
    assert top["page"] == 1
    assert "bbox" in top


def test_dedupe_on_chunk_id(cfg):
    store = vector.open_store(cfg, dim=3)
    c = _chunk(0, "first version")
    vector.upsert_chunks(store, [c], [[1.0, 0.0, 0.0]])
    c2 = _chunk(0, "second version")  # same chunk_id
    vector.upsert_chunks(store, [c2], [[1.0, 0.0, 0.0]])
    hits = vector.search(store, [1.0, 0.0, 0.0], k=5)
    matching = [h for h in hits if h["chunk_id"] == "id0"]
    assert len(matching) == 1
    assert matching[0]["text"] == "second version"


def test_store_persists_embed_model_metadata(cfg):
    vector.open_store(cfg, dim=3, embed_model="nomic-embed-text")
    assert vector.stored_embed_model(cfg) == "nomic-embed-text"
