"""Tests for docstore.render."""

from pathlib import Path

from docstore.render import RegionError, render_page, render_region

FIX = Path(__file__).parent / "fixtures"


def _png_size(data: bytes) -> tuple[int, int]:
    # PNG header: width/height are big-endian uint32 at offsets 16 and 20.
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w = int.from_bytes(data[16:20], "big")
    h = int.from_bytes(data[20:24], "big")
    return w, h


def test_render_page_png():
    data = render_page(FIX / "sample_layout.pdf", page=1, dpi=150)
    assert isinstance(data, bytes) and len(data) > 0
    w, h = _png_size(data)
    assert w > 0 and h > 0


def test_render_region_crop_is_smaller():
    full = render_page(FIX / "sample_layout.pdf", page=2, dpi=150)
    fw, fh = _png_size(full)
    # crop a small region near the top-left
    crop = render_region(FIX / "sample_layout.pdf", page=2, bbox=(72, 100, 200, 160), dpi=150)
    cw, ch = _png_size(crop)
    assert cw < fw and ch < fh


def test_render_dpi_scales_resolution():
    low = render_page(FIX / "sample_layout.pdf", page=1, dpi=72)
    high = render_page(FIX / "sample_layout.pdf", page=1, dpi=200)
    lw, _ = _png_size(low)
    hw, _ = _png_size(high)
    assert hw > lw


def test_render_region_degenerate_bbox_raises():
    # Zero-area / zero-width boxes must raise RegionError, not crash PyMuPDF's PNG encoder
    # (regression: vision models sometimes emit a point bbox, which crashed ingest).
    import pytest

    for bad in [(72, 100, 72, 100), (72, 100, 72, 160), (72, 100, 200, 100)]:
        with pytest.raises(RegionError):
            render_region(FIX / "sample_layout.pdf", page=2, bbox=bad, dpi=150)


def test_render_region_offpage_bbox_raises():
    # A box entirely off the page (e.g. image-pixel coords mistaken for points).
    import pytest

    with pytest.raises(RegionError):
        render_region(FIX / "sample_layout.pdf", page=2, bbox=(5000, 5000, 5100, 5100), dpi=150)


def test_render_region_reversed_corners_ok():
    # Swapped corners should still render the region, not fail.
    data = render_region(FIX / "sample_layout.pdf", page=2, bbox=(200, 160, 72, 100), dpi=150)
    w, h = _png_size(data)
    assert w > 0 and h > 0


def test_render_region_partial_offpage_clips():
    # A box straddling the page edge renders only the on-page part.
    data = render_region(FIX / "sample_layout.pdf", page=2, bbox=(500, 700, 900, 1100), dpi=150)
    w, h = _png_size(data)
    assert w > 0 and h > 0
