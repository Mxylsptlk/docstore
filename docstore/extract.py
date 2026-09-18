"""Layout-aware extraction from PDFs using PyMuPDF.

Returns, per page, a reading-ordered list of Blocks each carrying a bounding box and
a coarse type (text / image / table). This spatial information is what lets downstream
chunking keep a statistic glued to its label, and lets the vision gate decide which
pages actually need a vision pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pymupdf
from pydantic import BaseModel

from docstore.models import BBox, Block

# Heuristic: a text block whose lines look tabular (many short numeric cells aligned)
# could be flagged as a table. MVP keeps this best-effort and conservative.


class Page(BaseModel):
    page: int  # 1-indexed
    blocks: list[Block]
    width: float
    height: float


def _looks_tabular(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    numeric_lines = sum(1 for ln in lines if sum(c.isdigit() for c in ln) >= 3)
    return numeric_lines >= max(2, len(lines) // 2)


def extract_layout(path: str | Path) -> list[Page]:
    """Extract per-page blocks with bounding boxes in reading order."""
    doc = pymupdf.open(path)
    pages: list[Page] = []
    try:
        for i in range(doc.page_count):
            page = doc[i]
            raw = page.get_text("dict")
            blocks: list[Block] = []
            for blk in raw.get("blocks", []):
                bbox: BBox = tuple(float(v) for v in blk["bbox"])  # type: ignore[assignment]
                if blk.get("type") == 1:
                    # image block
                    blocks.append(Block(text="", bbox=bbox, block_type="image"))
                    continue
                # text block: join its spans
                text = _block_text(blk)
                if not text.strip():
                    continue
                btype = "table" if _looks_tabular(text) else "text"
                blocks.append(Block(text=text, bbox=bbox, block_type=btype))
            # reading order: top-to-bottom, then left-to-right
            blocks.sort(key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
            pages.append(
                Page(
                    page=i + 1,
                    blocks=blocks,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                )
            )
    finally:
        doc.close()
    return pages


def _block_text(blk: dict) -> str:
    parts: list[str] = []
    for line in blk.get("lines", []):
        span_text = "".join(span.get("text", "") for span in line.get("spans", []))
        if span_text:
            parts.append(span_text)
    return "\n".join(parts)
