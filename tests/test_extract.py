"""Tests for docstore.extract (layout-aware block extraction)."""

from pathlib import Path

from docstore.extract import extract_layout
from docstore.models import Block

FIX = Path(__file__).parent / "fixtures"


def test_extract_returns_pages_in_order():
    pages = extract_layout(FIX / "sample_layout.pdf")
    assert len(pages) == 2
    # 1-indexed page numbers
    assert [pg.page for pg in pages] == [1, 2]


def test_extract_blocks_have_bbox():
    pages = extract_layout(FIX / "sample_layout.pdf")
    for pg in pages:
        assert len(pg.blocks) > 0
        for b in pg.blocks:
            assert isinstance(b, Block)
            x0, y0, x1, y1 = b.bbox
            assert x1 >= x0 and y1 >= y0


def test_image_block_detected_on_image_page():
    pages = extract_layout(FIX / "sample_layout.pdf")
    page2 = pages[1]
    types = {b.block_type for b in page2.blocks}
    assert "image" in types


def test_plain_page_is_text_only():
    pages = extract_layout(FIX / "sample_layout.pdf")
    page1 = pages[0]
    assert all(b.block_type == "text" for b in page1.blocks)
    joined = " ".join(b.text for b in page1.blocks)
    assert "Project Overview" in joined


def test_reading_order_top_to_bottom():
    pages = extract_layout(FIX / "sample_layout.pdf")
    page1 = pages[0]
    text_blocks = [b for b in page1.blocks if b.block_type == "text"]
    ys = [b.bbox[1] for b in text_blocks]
    assert ys == sorted(ys)
