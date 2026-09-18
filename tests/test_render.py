"""Tests for docstore.render."""

from pathlib import Path

from docstore.render import render_page, render_region

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
