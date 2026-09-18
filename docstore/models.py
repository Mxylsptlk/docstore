"""Core data models for docstore.

Provenance is non-negotiable: every Chunk carries page + bounding-box location so
retrieved passages cite back to an exact spot in the source PDF. A stable chunk_id
(deterministic from doc_id + page + char span) lets a future graph layer attach edges
to existing chunks without re-ingesting.
"""

from __future__ import annotations

import hashlib
from pathlib import PurePath
from typing import Literal, Optional

from pydantic import BaseModel

# Bounding box in PDF points: (x0, y0, x1, y1)
BBox = tuple[float, float, float, float]

BlockType = Literal["text", "image", "table"]

# Entity categories the graph recognizes. Kept small and stable so nodes unify well.
EntityType = Literal["ORG", "PERSON", "MONEY", "DATE", "PROJECT", "TERM", "MISC"]


def normalize_entity_name(name: str) -> str:
    """Canonical key for an entity node so variants unify onto one node:
    'Acme Corp.' / 'acme corp' -> 'acme corp', and 'The Oakdale Corporation' /
    'Oakdale Corporation' -> 'oakdale corporation'.

    Lowercased, trimmed, internal whitespace collapsed, surrounding punctuation dropped,
    and a leading article (the/a/an) removed so a captured 'The ' prefix does not split
    an entity into two nodes."""
    import re

    n = re.sub(r"\s+", " ", name.strip().lower())
    n = n.strip(" .,;:")
    n = re.sub(r"^(?:the|a|an)\s+", "", n)
    return n


def make_chunk_id(doc_id: str, page: int, char_start: int, char_end: int) -> str:
    """Deterministic, stable chunk id (short hash). Stable across re-ingests so a
    future graph layer can join to it."""
    raw = f"{doc_id}:{page}:{char_start}:{char_end}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class DocMeta(BaseModel):
    """Metadata for one ingested document."""

    doc_id: str
    source_path: str
    page_count: int
    content_hash: str
    title: Optional[str] = None


class Block(BaseModel):
    """A layout block extracted from a page, with its bounding box."""

    text: str
    bbox: BBox
    block_type: BlockType = "text"


class Stat(BaseModel):
    """A single extracted statistic with provenance and verification status.

    verified: True (re-read confirms), False (re-read disagrees), None (unverifiable —
    no reliable bbox to crop and re-read). None is honest: we never pretend.
    """

    value_text: str
    label: str
    page: int
    bbox: Optional[BBox] = None
    verified: Optional[bool] = None
    reread_value: Optional[str] = None


class Chunk(BaseModel):
    """A retrievable unit of text with full positional provenance."""

    chunk_id: str
    doc_id: str
    source_path: str
    page: int
    bbox: BBox
    section: str = ""
    char_start: int
    char_end: int
    text: str
    is_stat: bool = False
    stat_verified: Optional[bool] = None


class Entity(BaseModel):
    """A named entity found in a chunk.

    name  — surface form as it appeared in the text (for display)
    key   — normalized canonical id; entity nodes unify on this
    etype — coarse category
    """

    name: str
    key: str
    etype: EntityType = "MISC"


class EntityMention(BaseModel):
    """A provenance-carrying link between a chunk and an entity it mentions."""

    chunk_id: str
    doc_id: str
    entity_key: str
    entity_name: str
    etype: EntityType = "MISC"
    page: int = 0


class RetrievalResult(BaseModel):
    """One search hit returned to the caller, with a resolvable citation."""

    chunk_id: str
    text: str
    source_path: str
    page: int
    bbox: Optional[BBox] = None
    is_stat: bool = False
    stat_verified: Optional[bool] = None
    score: float = 0.0

    @property
    def citation(self) -> str:
        """Human-readable citation like 'report.pdf:p12'."""
        return f"{PurePath(self.source_path).name}:p{self.page}"
