"""Tests for docstore.gate (per-page vision gating)."""

from docstore.extract import Page
from docstore.gate import needs_vision
from docstore.models import Block


def _page(blocks, num=1, w=612.0, h=792.0):
    return Page(page=num, blocks=blocks, width=w, height=h)


def test_gate_flags_image_page():
    pg = _page(
        [
            Block(text="Some caption", bbox=(0, 0, 100, 10), block_type="text"),
            Block(text="", bbox=(0, 20, 200, 200), block_type="image"),
        ]
    )
    assert needs_vision(pg) is True


def test_gate_flags_table_page():
    pg = _page([Block(text="1 2 3\n4 5 6\n7 8 9", bbox=(0, 0, 100, 60), block_type="table")])
    assert needs_vision(pg) is True


def test_gate_passes_plain_text_page():
    pg = _page(
        [
            Block(
                text="This report summarizes field operations for the quarter. "
                "All crews reported to their assigned sites on schedule and within budget.",
                bbox=(72, 72, 520, 120),
                block_type="text",
            ),
            Block(
                text="Every milestone tracked green throughout the reporting period.",
                bbox=(72, 130, 520, 160),
                block_type="text",
            ),
        ]
    )
    assert needs_vision(pg) is False


def test_gate_flags_numeral_heavy_low_text():
    # Numbers present but almost no words -> looks like a stats infographic w/ weak text layer.
    pg = _page([Block(text="42  87  1,204  $4.2M", bbox=(0, 0, 200, 20), block_type="text")])
    assert needs_vision(pg) is True


def test_gate_force_override():
    pg = _page(
        [Block(text="plain prose with plenty of ordinary words here", bbox=(0, 0, 300, 20), block_type="text")]
    )
    assert needs_vision(pg, force_vision=True) is True


def test_gate_empty_page_needs_vision():
    # No extractable text at all (e.g. scanned image) -> let vision look.
    pg = _page([])
    assert needs_vision(pg) is True
