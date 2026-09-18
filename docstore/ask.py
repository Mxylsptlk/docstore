"""ask(): retrieve relevant passages and write them to ANSWER.md for the assistant.

The MVP's "answering model" is Claude (me), reading this file — the vector store only
retrieves passages, it does not compose answers. ANSWER.md therefore contains the
question, the retrieved context (each passage under its citation heading), explicit
warnings on any unverified/failed statistic, and an instruction block telling the
assistant to cite passages, never invent or re-round numbers, and say when the context
is insufficient.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from docstore.config import Config
from docstore.models import RetrievalResult
from docstore.retrieve import retrieve

_INSTRUCTIONS = (
    "## Instructions for the assistant\n\n"
    "- Answer the question using ONLY the passages above.\n"
    "- Cite the passage heading (e.g. `q3.pdf:p12`) for every claim.\n"
    "- Transcribe statistics EXACTLY as written; never compute, sum, or re-round them.\n"
    "- If a passage is marked `⚠ unverified` or `⚠ verification FAILED`, warn the user "
    "and tell them to confirm that number against the source before relying on it.\n"
    "- If the passages do not contain the answer, say so plainly instead of guessing.\n"
)


def _verif_badge(r: RetrievalResult) -> str:
    if not r.is_stat:
        return ""
    if r.stat_verified is True:
        return "  ✓ verified"
    if r.stat_verified is False:
        return "  ⚠ verification FAILED — confirm against source"
    return "  ⚠ unverified — confirm against source"


def _render(query: str, results: list[RetrievalResult]) -> str:
    lines: list[str] = []
    lines.append("# docstore retrieval context\n")
    lines.append(f"**Question:** {query}\n")
    lines.append(f"_Retrieved {len(results)} passage(s) at {datetime.now().isoformat(timespec='seconds')}._\n")
    lines.append("---\n")

    if not results:
        lines.append("_No relevant passages were found in the store for this question._\n")
        lines.append("---\n")
        lines.append(_INSTRUCTIONS)
        return "\n".join(lines)

    lines.append("## Context passages\n")
    for r in results:
        lines.append(f"### {r.citation}{_verif_badge(r)}\n")
        lines.append(r.text.strip() + "\n")
    lines.append("---\n")
    lines.append(_INSTRUCTIONS)
    return "\n".join(lines)


def ask(query: str, k: int = 5, cfg: Optional[Config] = None) -> str:
    """Retrieve top-k passages for `query` and write ANSWER.md. Returns its path."""
    cfg = cfg or Config()
    results = retrieve(query, k=k, cfg=cfg)
    out_path = Path(cfg.answer_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(query, results))
    return str(out_path)
