"""Render PDF pages (and bbox regions) to PNG bytes for vision extraction and for
re-reading individual statistics during verification.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from docstore.models import BBox

# A clipped region must be at least this many points on each side to be renderable
# (below ~1pt PyMuPDF produces an empty pixmap that cannot be PNG-encoded).
_MIN_REGION_PTS = 1.0


class RegionError(ValueError):
    """A bbox region could not be rendered (empty, degenerate, or off-page)."""


def _matrix(dpi: int) -> pymupdf.Matrix:
    # 72 is the PDF's native points-per-inch.
    scale = dpi / 72.0
    return pymupdf.Matrix(scale, scale)


def render_page(path: str | Path, page: int, dpi: int = 220) -> bytes:
    """Render a full page (1-indexed) to PNG bytes at the given DPI."""
    doc = pymupdf.open(path)
    try:
        pg = doc[page - 1]
        pix = pg.get_pixmap(matrix=_matrix(dpi))
        return pix.tobytes("png")
    finally:
        doc.close()


def render_region(path: str | Path, page: int, bbox: BBox, dpi: int = 220) -> bytes:
    """Render just the given bbox region (PDF points) of a page to PNG bytes.

    Used to crop a single statistic so a vision model can re-read it in isolation.

    The bbox is normalized (min/max ordered) and intersected with the page bounds so a
    slightly-off or reversed box still renders the overlapping part. A box that is empty,
    degenerate, or entirely off-page raises RegionError rather than producing an
    unencodable pixmap (which would otherwise crash the whole ingest).
    """
    doc = pymupdf.open(path)
    try:
        pg = doc[page - 1]
        x0, y0, x1, y1 = bbox
        # Order corners defensively (models sometimes emit reversed coords).
        clip = pymupdf.Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        # Keep only the part that actually lies on the page.
        clip = clip & pg.rect
        if clip.is_empty or clip.width < _MIN_REGION_PTS or clip.height < _MIN_REGION_PTS:
            raise RegionError(
                f"bbox {bbox} yields no renderable region on page {page} "
                f"(page rect {tuple(pg.rect)})"
            )
        pix = pg.get_pixmap(matrix=_matrix(dpi), clip=clip)
        if pix.width == 0 or pix.height == 0:
            raise RegionError(f"bbox {bbox} produced an empty pixmap on page {page}")
        return pix.tobytes("png")
    finally:
        doc.close()
