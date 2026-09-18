"""Pluggable entity extraction for the knowledge graph.

Given a Chunk, return the named entities it mentions as `Entity` objects (deduped on
canonical key within the chunk). Backends:

  - "regex" (default): zero-dependency patterns for MONEY / DATE / ORG. Deterministic,
    fast, no network. Good enough to seed a useful graph today.
  - "ollama": a local instruct model (e.g. llama3.1) for richer entities.
  - "claude": Anthropic for the best entities (costs tokens per chunk).

The richer backends fall back to regex on any failure, so ingestion never breaks just
because a model is unavailable.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from docstore.config import Config
from docstore.models import Chunk, Entity, EntityType, normalize_entity_name

# ---- regex patterns -----------------------------------------------------------

# $4.2M, $1,204, $3.5 billion, USD 500K
_MONEY_RE = re.compile(
    r"(?:\$|USD\s?)\s?\d[\d,]*(?:\.\d+)?\s*(?:K|M|B|T|thousand|million|billion|trillion)?",
    re.IGNORECASE,
)
# 03/15/2024, 2024-03-15, March 15, 2024, Q3 2024
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|Q[1-4]\s+\d{4})\b",
    re.IGNORECASE,
)
# Proper-noun runs, optionally ending in a company suffix: "Acme Corporation",
# "Oakdale", "Smith & Co". Two+ capitalized words, or one capitalized word + suffix.
_ORG_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z&.]+(?:\s+[A-Z][A-Za-z&.]+)+"
    r"|[A-Z][A-Za-z&.]+\s+(?:Inc|LLC|Ltd|Corp|Corporation|Company|Co|Group|Associates|Partners)\.?)"
)

# Very common capitalized non-entities to drop from the ORG pass.
_ORG_STOPWORDS = {"the", "a", "an", "this", "that"}


def _regex_entities(text: str) -> list[Entity]:
    found: list[Entity] = []
    for m in _MONEY_RE.finditer(text):
        found.append(_ent(m.group(0), "MONEY"))
    for m in _DATE_RE.finditer(text):
        found.append(_ent(m.group(0), "DATE"))
    for m in _ORG_RE.finditer(text):
        surface = m.group(0).strip()
        if surface.lower() in _ORG_STOPWORDS:
            continue
        found.append(_ent(surface, "ORG"))
    return found


def _ent(surface: str, etype: EntityType) -> Entity:
    surface = surface.strip()
    return Entity(name=surface, key=normalize_entity_name(surface), etype=etype)


def _dedupe(entities: list[Entity]) -> list[Entity]:
    """One node per (etype, key) within a chunk; keep the first surface form."""
    seen: set[tuple[str, str]] = set()
    out: list[Entity] = []
    for e in entities:
        sig = (e.etype, e.key)
        if e.key and sig not in seen:
            seen.add(sig)
            out.append(e)
    return out


# ---- LLM backends -------------------------------------------------------------

_LLM_PROMPT = (
    "Extract named entities from the text. Return ONLY a JSON list of objects "
    '{"name": "...", "type": "ORG|PERSON|MONEY|DATE|PROJECT|TERM|MISC"}. '
    "Use the exact surface form for name. No commentary.\n\nText:\n"
)

_VALID_TYPES = {"ORG", "PERSON", "MONEY", "DATE", "PROJECT", "TERM", "MISC"}


def _parse_llm_entities(raw: str) -> list[Entity]:
    raw = raw.strip()
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: list[Entity] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name", "")).strip()
        etype = str(it.get("type", "MISC")).upper()
        if not name:
            continue
        if etype not in _VALID_TYPES:
            etype = "MISC"
        out.append(_ent(name, etype))  # type: ignore[arg-type]
    return out


def _extract_ollama(text: str, cfg: Config) -> list[Entity]:
    import ollama

    resp = ollama.chat(
        model=cfg.entity_model,
        messages=[{"role": "user", "content": _LLM_PROMPT + text}],
    )
    return _parse_llm_entities(resp["message"]["content"])


def _extract_claude(text: str, cfg: Config) -> list[Entity]:
    import anthropic

    from docstore.anthropic_models import resolve_model

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key())
    msg = client.messages.create(
        model=resolve_model(cfg.vision_model),
        max_tokens=1024,
        messages=[{"role": "user", "content": _LLM_PROMPT + text}],
    )
    raw = "".join(getattr(b, "text", "") for b in msg.content)
    return _parse_llm_entities(raw)


# ---- public dispatch ----------------------------------------------------------

def extract_entities(
    chunk: Chunk,
    backend: Optional[str] = None,
    cfg: Optional[Config] = None,
) -> list[Entity]:
    cfg = cfg or Config()
    backend = backend or cfg.entity_backend

    if backend == "regex":
        return _dedupe(_regex_entities(chunk.text))
    if backend == "ollama":
        return _dedupe(_llm_with_fallback(_extract_ollama, chunk.text, cfg))
    if backend == "claude":
        return _dedupe(_llm_with_fallback(_extract_claude, chunk.text, cfg))
    raise ValueError(f"Unknown entity backend: {backend!r} (use regex|ollama|claude)")


def _llm_with_fallback(fn, text: str, cfg: Config) -> list[Entity]:
    """Run an LLM backend; on any failure, fall back to regex so ingest never breaks."""
    try:
        ents = fn(text, cfg)
        # If the model returned nothing useful, still salvage regex hits.
        return ents or _regex_entities(text)
    except Exception:  # noqa: BLE001
        return _regex_entities(text)
