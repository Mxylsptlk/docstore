"""Statistics verification — the accuracy backstop.

For each extracted Stat that has a bounding box, crop that region from the page, re-read
the number in isolation with the vision model, and compare it (normalized) to the value
the extraction pass reported. This catches:
  - text-layer scrambling (the "scrambled-PDF numbers" failure mode), and
  - extraction transcription errors.

Verification outcomes:
  verified = True   -> re-read agrees
  verified = False  -> re-read disagrees (flagged loudly downstream)
  verified = None   -> unverifiable (no bbox to crop, region unrenderable, or disabled)

We never mark a stat verified without actually re-reading it.
"""

from __future__ import annotations

import re
from pathlib import Path

from docstore.config import Config
from docstore.models import Stat
from docstore.render import RegionError, render_region
from docstore.vision_extract import extract_from_image

_REREAD_INSTRUCTION = (
    "This image is a small crop containing a single statistic. Respond with ONLY that "
    "number, exactly as shown (keep %, $, decimals, and units). Do not add words."
)


def normalize_number(s: str) -> str:
    """Normalize a numeric string for comparison: drop currency, thousands separators,
    surrounding whitespace, and trailing punctuation; keep digits, decimal point, sign,
    percent, and magnitude suffix (K/M/B), case-insensitively."""
    s = s.strip().lower()
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    s = re.sub(r"[^0-9.%kmb+-]", "", s)
    return s


def _reread_number(path: str | Path, page: int, bbox, cfg: Config) -> str:
    """Crop the stat's region and ask the vision model to read just that number."""
    png = render_region(path, page, bbox, dpi=cfg.render_dpi)
    text, _stats = extract_from_image(
        png,
        _REREAD_INSTRUCTION,
        backend=cfg.extraction_backend,
        model=cfg.vision_model,
    )
    return text.strip()


def verify_stats(path: str | Path, stats: list[Stat], cfg: Config) -> list[Stat]:
    if not cfg.verify_stats:
        return stats  # verification off: leave verified=None (honest, not falsely trusted)

    for stat in stats:
        if not stat.bbox:
            continue  # unverifiable -> stays verified=None
        try:
            reread = _reread_number(path, stat.page, stat.bbox, cfg)
        except RegionError:
            # Degenerate / off-page bbox from the extractor: can't crop it to re-read,
            # so it stays unverifiable (verified=None) rather than crashing the ingest.
            continue
        stat.reread_value = reread
        stat.verified = normalize_number(reread) == normalize_number(stat.value_text)
    return stats
