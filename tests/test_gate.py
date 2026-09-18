"""Tests for docstore.gate (per-page routing: skip / vision / text-layer)."""

from docstore.extract import Page
from docstore.gate import needs_vision, page_should_skip
from docstore.models import Block


def _page(blocks, num=1, w=612.0, h=792.0):
    return Page(page=num, blocks=blocks, width=w, height=h)


# ---------- needs_vision ----------

def test_vision_for_stat_callouts():
    # Scattered formatted stats in short standalone blocks -> vision.
    pg = _page(
        [
            Block(text="On-time rate", bbox=(72, 100, 200, 118), block_type="text"),
            Block(text="42%", bbox=(72, 120, 140, 150), block_type="text"),
            Block(text="$4.2M", bbox=(300, 120, 400, 150), block_type="text"),
        ]
    )
    assert needs_vision(pg) is True


def test_vision_for_table_page():
    pg = _page([Block(text="1 2 3\n4 5 6\n7 8 9", bbox=(0, 0, 100, 60), block_type="table")])
    assert needs_vision(pg) is True


def test_vision_for_bare_number_grid():
    # Several standalone bare numbers (infographic number grid) -> vision.
    pg = _page(
        [
            Block(text="128", bbox=(72, 100, 120, 130), block_type="text"),
            Block(text="94", bbox=(200, 100, 250, 130), block_type="text"),
            Block(text="17", bbox=(330, 100, 380, 130), block_type="text"),
        ]
    )
    assert needs_vision(pg) is True


def test_no_vision_for_plain_prose():
    pg = _page(
        [
            Block(
                text="This report summarizes field operations for the quarter. All crews "
                "reported to their assigned sites on schedule and within budget.",
                bbox=(72, 72, 520, 120),
                block_type="text",
            )
        ]
    )
    assert needs_vision(pg) is False


def test_no_vision_for_number_in_prose_sentence():
    # A number embedded in a running sentence reads fine in order -> text layer, not vision.
    pg = _page(
        [
            Block(
                text="Total revenue grew to $4.2M in the third quarter, up from the prior "
                "period, driven by higher field utilization across all active crews.",
                bbox=(72, 72, 520, 120),
                block_type="text",
            )
        ]
    )
    assert needs_vision(pg) is False


def test_no_vision_for_decorative_image_page():
    # Image + a couple of words, no stats -> NOT vision (it will be skipped instead).
    pg = _page(
        [
            Block(text="Section Two", bbox=(72, 60, 300, 90), block_type="text"),
            Block(text="", bbox=(0, 100, 400, 400), block_type="image"),
        ]
    )
    assert needs_vision(pg) is False


def test_force_vision_override():
    pg = _page([Block(text="plain prose with plenty of ordinary words here and there",
                      bbox=(0, 0, 300, 20), block_type="text")])
    assert needs_vision(pg, force_vision=True) is True


# ---------- page_should_skip ----------

def test_skip_sparse_cover_page():
    pg = _page(
        [
            Block(text="Quarterly Report", bbox=(72, 60, 300, 90), block_type="text"),
            Block(text="2024", bbox=(72, 100, 140, 120), block_type="text"),  # a year, not a stat
        ]
    )
    assert page_should_skip(pg) is True


def test_skip_image_only_section_divider():
    pg = _page(
        [
            Block(text="Section Two", bbox=(72, 60, 300, 90), block_type="text"),
            Block(text="", bbox=(0, 100, 400, 400), block_type="image"),
        ]
    )
    assert page_should_skip(pg) is True


def test_skip_empty_page():
    assert page_should_skip(_page([])) is True


def test_do_not_skip_page_with_stats():
    # Sparse page BUT it has a real statistic -> must be kept (and routed to vision).
    pg = _page(
        [
            Block(text="On-time rate", bbox=(72, 100, 200, 118), block_type="text"),
            Block(text="42%", bbox=(72, 120, 140, 150), block_type="text"),
        ]
    )
    assert page_should_skip(pg) is False
    assert needs_vision(pg) is True  # kept AND sent to vision


def test_do_not_skip_substantive_prose():
    pg = _page(
        [
            Block(
                text="This report summarizes field operations for the quarter across every "
                "active site, covering scheduling, safety, and subcontractor coordination.",
                bbox=(72, 72, 520, 120),
                block_type="text",
            )
        ]
    )
    assert page_should_skip(pg) is False


def test_do_not_skip_bare_number_grid():
    pg = _page(
        [
            Block(text="128", bbox=(72, 100, 120, 130), block_type="text"),
            Block(text="94", bbox=(200, 100, 250, 130), block_type="text"),
        ]
    )
    assert page_should_skip(pg) is False
