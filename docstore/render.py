"""Render PDF pages (and bbox regions) to PNG bytes for vision extraction and for
re-reading individual statistics during verification.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from docstore.models import BBox


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
    """
    doc = pymupdf.open(path)
    try:
        pg = doc[page - 1]
        clip = pymupdf.Rect(*bbox)
        pix = pg.get_pixmap(matrix=_matrix(dpi), clip=clip)
        return pix.tobytes("png")
    finally:
        doc.close()
