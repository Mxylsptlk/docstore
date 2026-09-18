"""Tests for docstore.chunk (layout-aware chunking that keeps stats with labels)."""

from docstore.chunk import chunk_page
from docstore.models import Block, Stat


def _blocks():
    return [
        Block(text="Key Statistics", bbox=(72, 60, 300, 80), block_type="text"),
        Block(text="On-time completion rate", bbox=(72, 100, 300, 118), block_type="text"),
        Block(text="42%", bbox=(72, 120, 140, 150), block_type="text"),
        Block(text="Some closing remarks about the quarter.", bbox=(72, 200, 400, 220), block_type="text"),
    ]


def test_chunks_carry_bbox_and_page():
    chunks = chunk_page(
        doc_id="doc1", source_path="/tmp/f.pdf", page=2,
        blocks=_blocks(), stats=[], size=1000, overlap=100,
    )
    assert chunks
    for c in chunks:
        assert c.page == 2
        x0, y0, x1, y1 = c.bbox
        assert x1 >= x0 and y1 >= y0
        assert c.doc_id == "doc1"


def test_stat_kept_with_label():
    stat = Stat(value_text="42%", label="On-time completion rate", page=2,
                bbox=(72, 120, 140, 150), verified=True)
    chunks = chunk_page(
        doc_id="doc1", source_path="/tmp/f.pdf", page=2,
        blocks=_blocks(), stats=[stat], size=1000, overlap=100,
    )
    # find the chunk containing the stat value
    stat_chunks = [c for c in chunks if "42%" in c.text]
    assert len(stat_chunks) == 1
    c = stat_chunks[0]
    # its label must be in the SAME chunk
    assert "On-time completion rate" in c.text
    assert c.is_stat is True
    assert c.stat_verified is True


def test_respects_size_limit():
    # Realistic paragraph-sized blocks (~100 chars each); chunker packs whole blocks
    # up to `size` without splitting any single block.
    big = [Block(text="word " * 20, bbox=(0, i * 20, 500, i * 20 + 15), block_type="text")
           for i in range(10)]
    chunks = chunk_page(
        doc_id="d", source_path="/tmp/f.pdf", page=1,
        blocks=big, stats=[], size=500, overlap=50,
    )
    assert len(chunks) > 1
    for c in chunks:
        # each chunk stays within one block's-worth of slack over the target size
        assert len(c.text) <= 500 + 100


def test_char_spans_are_monotonic_and_unique_ids():
    chunks = chunk_page(
        doc_id="doc1", source_path="/tmp/f.pdf", page=2,
        blocks=_blocks(), stats=[], size=1000, overlap=100,
    )
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert c.char_end >= c.char_start


def test_empty_page_yields_no_chunks():
    assert chunk_page(doc_id="d", source_path="/tmp/f.pdf", page=1,
                      blocks=[], stats=[], size=500, overlap=50) == []
