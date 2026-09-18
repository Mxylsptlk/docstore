"""Per-page vision gate — the cost-control heart of ingestion.

Extraction defaults to Claude vision, but sending every page to a vision model is
expensive. This gate decides, per page, whether vision is actually warranted:

  - Pages with image or table blocks     -> vision (charts/infographics/tables).
  - Pages with no extractable text        -> vision (likely scanned).
  - Pages where numerals appear but the   -> vision (stats infographic whose text
    text layer is sparse/garbled             layer can't be trusted — matches the
                                             "scrambled-PDF numbers" failure mode).
  - Clean, word-rich text pages           -> skip vision, use the free text layer.

force_vision=True short-circuits everything to True (maximum fidelity).
"""

from __future__ import annotations

import re

from docstore.extract import Page

_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_NUMERAL_RE = re.compile(r"\d")

# A page needs at least this many real words in its text layer to be trusted as
# "text-only" when numerals are present.
_MIN_WORDS_WITH_NUMERALS = 15


def needs_vision(page: Page, *, force_vision: bool = False) -> bool:
    if force_vision:
        return True

    # Any non-text visual content -> vision.
    if any(b.block_type in ("image", "table") for b in page.blocks):
        return True

    text_blocks = [b for b in page.blocks if b.block_type == "text"]
    all_text = " ".join(b.text for b in text_blocks)

    words = _WORD_RE.findall(all_text)
    # Completely empty text layer -> likely scanned; let vision look.
    if not all_text.strip():
        return True

    has_numerals = bool(_NUMERAL_RE.search(all_text))
    if has_numerals and len(words) < _MIN_WORDS_WITH_NUMERALS:
        # Numbers with almost no surrounding prose: a stats graphic with a weak
        # text layer. Vision reads these far more reliably.
        return True

    return False
