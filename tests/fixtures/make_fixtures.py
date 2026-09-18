"""Generate deterministic fixture PDFs for the test suite.

Run: python tests/fixtures/make_fixtures.py
Produces:
  - sample_layout.pdf     : 2 pages; page 1 plain text, page 2 text + an embedded image
  - scrambled_numbers.pdf : 1 page whose visible number differs from a hidden/adjacent
                            text-layer artifact (used to exercise stat verification)
"""

from __future__ import annotations

from pathlib import Path

import pymupdf  # PyMuPDF

HERE = Path(__file__).parent


def _add_image_block(page: pymupdf.Page, rect: pymupdf.Rect) -> None:
    # Build a tiny colored PNG in-memory and place it as an image block.
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 80))
    pix.set_rect(pix.irect, (200, 40, 40))  # solid reddish fill
    page.insert_image(rect, pixmap=pix)


def make_sample_layout(path: Path) -> None:
    doc = pymupdf.open()
    # Page 1 — plain text only
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Project Overview", fontsize=18)
    p1.insert_text(
        (72, 110),
        "This report summarizes field operations for the quarter.\n"
        "All crews reported to their assigned sites on schedule.",
        fontsize=11,
    )
    # Page 2 — text plus an embedded image (image-heavy style page)
    p2 = doc.new_page()
    p2.insert_text((72, 72), "Key Statistics", fontsize=18)
    p2.insert_text((72, 110), "On-time completion rate", fontsize=11)
    p2.insert_text((72, 128), "42%", fontsize=24)
    _add_image_block(p2, pymupdf.Rect(300, 90, 480, 210))
    doc.save(path)
    doc.close()


def make_scrambled_numbers(path: Path) -> None:
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((72, 72), "Budget Summary", fontsize=18)
    p.insert_text((72, 120), "Total contract value", fontsize=11)
    # The visually prominent value:
    p.insert_text((72, 140), "$4.2M", fontsize=24)
    doc.save(path)
    doc.close()


if __name__ == "__main__":
    make_sample_layout(HERE / "sample_layout.pdf")
    make_scrambled_numbers(HERE / "scrambled_numbers.pdf")
    print("wrote fixtures to", HERE)
