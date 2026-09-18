"""Per-page routing — the cost-control + accuracy heart of ingestion.

Stats in these PDFs are selectable TEXT; the failure mode is scattered numeric callouts
that read in random order when a page is flattened to a text stream (and whose number
loses its label). We use PyMuPDF's spatial layout to fix gross reading order, and route
the genuinely ambiguous pages to a vision pass that reads them the way a human sees them.

Each page gets one of three routes:

  SKIP        page_should_skip() — sparse decorative pages (cover, section divider) that
              carry no statistic. Dropped on ingest so they don't pollute retrieval.
              A page is only skipped when there is NOTHING statistical to lose.

  VISION      needs_vision() — pages whose numbers would scramble as text:
                - table blocks (tabular numbers), or
                - stat-like numbers sitting in short standalone "callout" blocks, or
                - a grid of several bare-number blocks (infographic number grid).

  TEXT LAYER  everything else — clean prose (including numbers embedded in sentences,
              which read fine in order) uses the free, accurate text layer.

force_vision bypasses SKIP and routes every page to VISION (maximum fidelity).
"""

from __future__ import annotations

import re

from docstore.extract import Page
from docstore.models import Block

_WORD_RE = re.compile(r"[A-Za-z]{2,}")

# A "stat-like" number carries a format/unit marker that decorative numbers (page
# numbers, section numbers, plain years) almost never do: currency, percent, decimal,
# thousands separator, or a magnitude/multiplier suffix (K/M/B/T/x).
_STAT_RE = re.compile(
    r"""
      \$\s?\d                                 # currency: $4, $4.2M
    | \d[\d,]*\.?\d*\s*%                       # percent: 42%, 3.5%
    | \d[\d,]*\.\d+                            # decimal: 3.5, 4.20
    | \d{1,3}(?:,\d{3})+                       # thousands separated: 1,204
    | \d+(?:\.\d+)?\s*[KMBT](?![A-Za-z])       # magnitude: 500K, 4.2M
    | \d+(?:\.\d+)?\s*[x×](?![A-Za-z])         # multiplier: 3x, 2.5x
    """,
    re.VERBOSE | re.IGNORECASE,
)

# A block that is essentially just a number (an infographic callout like "42" or "1,204").
_BARE_NUMBER_RE = re.compile(r"^\W*\d[\d,]*(?:\.\d+)?\W*$")

# Short blocks are "callouts", not running prose.
_CALLOUT_MAX_WORDS = 8
# Pages with fewer real words than this are candidates for skipping (if no stats).
_SKIP_MAX_WORDS = 12


def _text_blocks(page: Page) -> list[Block]:
    return [b for b in page.blocks if b.block_type == "text"]


def _is_stat_callout(block: Block) -> bool:
    """A short standalone block containing a clearly-formatted statistic — the thing that
    scrambles when flattened to text and loses its label."""
    text = block.text.strip()
    if not text:
        return False
    if len(_WORD_RE.findall(text)) > _CALLOUT_MAX_WORDS:
        return False  # a sentence, not a callout
    return bool(_STAT_RE.search(text))


def _bare_number_blocks(blocks: list[Block]) -> list[Block]:
    return [b for b in blocks if _BARE_NUMBER_RE.match(b.text.strip())]


def _has_stats(page: Page) -> bool:
    """True if the page carries something statistical worth keeping."""
    text_blocks = _text_blocks(page)
    all_text = " ".join(b.text for b in text_blocks)
    if _STAT_RE.search(all_text):
        return True
    # A number grid (several standalone numbers) counts as stats even if unformatted.
    return len(_bare_number_blocks(text_blocks)) >= 2


def page_should_skip(page: Page) -> bool:
    """Skip sparse decorative pages (cover, section divider) — but only when there is no
    statistic to lose. Any stat-like content keeps the page."""
    words = _WORD_RE.findall(" ".join(b.text for b in _text_blocks(page)))
    if len(words) >= _SKIP_MAX_WORDS:
        return False  # substantive text -> keep
    return not _has_stats(page)


def needs_vision(page: Page, *, force_vision: bool = False) -> bool:
    """Route a (non-skipped) page to vision when its numbers would scramble as text."""
    if force_vision:
        return True
    # Tabular numbers scramble when flattened.
    if any(b.block_type == "table" for b in page.blocks):
        return True
    text_blocks = _text_blocks(page)
    # Formatted stats sitting in short callout blocks (scattered / gridded).
    if any(_is_stat_callout(b) for b in text_blocks):
        return True
    # A grid of several bare-number callouts (infographic-style number grid).
    if len(_bare_number_blocks(text_blocks)) >= 2:
        return True
    return False
