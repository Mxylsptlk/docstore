"""Pluggable vision-based extraction.

Given a rendered page image plus a per-ingest instruction, return:
  (curated_text, [Stat, ...])

The system prompt demands VERBATIM numeric transcription — the model must copy each
number exactly as shown and must never compute, round, or reformat it. Accurate
statistics are the whole point of this tool.

Backends:
  - "claude" (default): Anthropic vision. Key from ANTHROPIC_API_KEY (never stored).
  - "ollama": a local vision model (e.g. llama3.2-vision). Weaker at exact numbers;
    offered for offline/bulk use, not recommended for stat-critical work.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Optional

import anthropic
import ollama

from docstore.models import BBox, Stat

SYSTEM_PROMPT = (
    "You extract information from a single rendered page of a PDF. "
    "Follow the user's instruction about WHAT to extract. "
    "CRITICAL RULES FOR NUMBERS: transcribe every statistic EXACTLY as shown on the "
    "page — same digits, decimals, sign, currency symbol, and unit (%, $, K, M, B). "
    "Never compute, sum, round, or reformat a number. If a digit is unclear, transcribe "
    "your best reading and do not guess additional precision.\n\n"
    "Respond ONLY with a JSON object of the form:\n"
    '{"curated_text": "<the extracted information as readable text>", '
    '"stats": [{"value_text": "<number exactly as shown>", "label": "<what it measures>", '
    '"bbox": [x0,y0,x1,y1] or null}]}\n'
    "The bbox (in PDF points, origin top-left) is optional; use null if you cannot place it. "
    "If there are no statistics, use an empty stats list."
)


def extract_from_image(
    png: bytes,
    instruction: str,
    *,
    backend: str = "claude",
    model: str = "claude-sonnet-5",
) -> tuple[str, list[Stat]]:
    if backend == "claude":
        raw = _extract_claude(png, instruction, model)
    elif backend == "ollama":
        raw = _extract_ollama(png, instruction, model)
    else:
        raise ValueError(f"Unknown extraction backend: {backend!r} (use 'claude' or 'ollama')")
    return _parse_response(raw)


def _user_instruction(instruction: str) -> str:
    return (
        f"Instruction: {instruction}\n\n"
        "Extract the requested information from the page image and respond with the "
        "JSON object described in the system prompt."
    )


def _extract_claude(png: bytes, instruction: str, model: str) -> str:
    import os

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before using the Claude backend."
        )
    client = anthropic.Anthropic(api_key=key)
    b64 = base64.standard_b64encode(png).decode()
    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64},
                    },
                    {"type": "text", "text": _user_instruction(instruction)},
                ],
            }
        ],
    )
    return "".join(getattr(block, "text", "") for block in msg.content)


def _extract_ollama(png: bytes, instruction: str, model: str) -> str:
    b64 = base64.standard_b64encode(png).decode()
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_instruction(instruction), "images": [b64]},
        ],
        images=[b64],
    )
    return resp["message"]["content"]


def _parse_response(raw: str) -> tuple[str, list[Stat]]:
    """Parse the model's JSON. Tolerant: if it isn't valid JSON, fall back to using the
    raw text as curated_text with no stats (never crash the pipeline)."""
    obj = _loads_lenient(raw)
    if obj is None:
        return raw.strip(), []
    curated = str(obj.get("curated_text", "")).strip() or raw.strip()
    stats: list[Stat] = []
    for s in obj.get("stats", []) or []:
        bbox: Optional[BBox] = None
        b = s.get("bbox")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                bbox = tuple(float(v) for v in b)  # type: ignore[assignment]
            except (TypeError, ValueError):
                bbox = None
        stats.append(
            Stat(
                value_text=str(s.get("value_text", "")).strip(),
                label=str(s.get("label", "")).strip(),
                page=int(s.get("page", 0)) or 0,
                bbox=bbox,
                verified=None,
            )
        )
    return curated, stats


def _loads_lenient(raw: str) -> Optional[dict]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip ```json fences if present, then try the first {...} span.
    fenced = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", fenced, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None
