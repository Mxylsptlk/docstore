"""Tests for docstore.models."""

from docstore.models import Block, Chunk, DocMeta, RetrievalResult, Stat, make_chunk_id


def test_chunk_has_bbox_provenance():
    c = Chunk(
        chunk_id="abc",
        doc_id="doc1",
        source_path="/tmp/f.pdf",
        page=3,
        bbox=(10.0, 20.0, 100.0, 60.0),
        section="Results",
        char_start=0,
        char_end=42,
        text="Revenue rose to $4.2M",
        is_stat=True,
    )
    assert c.page == 3
    assert c.bbox == (10.0, 20.0, 100.0, 60.0)
    assert c.is_stat is True


def test_stat_model():
    s = Stat(
        value_text="42%",
        label="on-time completion",
        page=2,
        bbox=(1.0, 2.0, 3.0, 4.0),
        verified=True,
    )
    assert s.value_text == "42%"
    assert s.verified is True
    # unverifiable stats allowed (no reliable bbox)
    s2 = Stat(value_text="17", label="crews", page=1, bbox=None, verified=None)
    assert s2.verified is None


def test_block_model():
    b = Block(text="hello", bbox=(0.0, 0.0, 5.0, 5.0), block_type="text")
    assert b.block_type == "text"


def test_docmeta_model():
    d = DocMeta(doc_id="doc1", source_path="/tmp/f.pdf", page_count=5, content_hash="deadbeef")
    assert d.page_count == 5


def test_chunk_id_deterministic_and_stable():
    a = make_chunk_id("doc1", 3, 0, 42)
    b = make_chunk_id("doc1", 3, 0, 42)
    c = make_chunk_id("doc1", 3, 0, 43)
    assert a == b
    assert a != c
    assert isinstance(a, str) and len(a) > 0


def test_retrieval_result_citation():
    r = RetrievalResult(
        chunk_id="abc",
        text="Revenue rose to $4.2M",
        source_path="/tmp/reports/f.pdf",
        page=12,
        bbox=(1.0, 2.0, 3.0, 4.0),
        is_stat=True,
        stat_verified=False,
        score=0.87,
    )
    assert r.citation == "f.pdf:p12"
    assert r.stat_verified is False
