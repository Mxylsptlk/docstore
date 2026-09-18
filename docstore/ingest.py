"""Full ingest pipeline orchestration.

Per document:
  1. extract_layout            -> per-page blocks with bboxes
  2. per page: needs_vision?   -> gate (cost control)
       - gated IN:  render page -> vision extract (curated text + stats) -> verify stats
                    -> synthesize blocks for chunking (stats kept with labels)
       - gated OUT: use the free text-layer blocks directly (no vision, no API cost)
  3. chunk_page                -> provenance-carrying chunks
  4. embed_texts (local Ollama)
  5. vector.upsert_chunks      -> LanceDB

Returns a summary dict including pages_vision / pages_textonly so cost is visible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from docstore.chunk import chunk_page
from docstore.config import Config
from docstore.embed import embed_texts
from docstore.entities import extract_entities
from docstore.extract import Page, extract_layout
from docstore.gate import needs_vision, page_should_skip
from docstore.models import Block, Chunk, Entity, EntityMention, Stat
from docstore.render import render_page
from docstore.stores import graph as graph_store
from docstore.stores import vector
from docstore.verify_stats import verify_stats
from docstore.vision_extract import extract_from_image


def _doc_id(path: Path, content_hash: str) -> str:
    return f"{path.stem}-{content_hash[:8]}"


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vision_blocks(curated_text: str, stats: list[Stat], page: Page) -> list[Block]:
    """Turn a vision pass into blocks for the chunker: one block for the curated text,
    plus explicit 'label: value' blocks for each stat so the chunker keeps them together.
    Bboxes come from the stat when available, else the page bounds."""
    page_bbox = (0.0, 0.0, page.width, page.height)
    blocks: list[Block] = []
    if curated_text.strip():
        blocks.append(Block(text=curated_text.strip(), bbox=page_bbox, block_type="text"))
    for s in stats:
        bbox = s.bbox or page_bbox
        label = f"{s.label}: " if s.label else ""
        blocks.append(Block(text=f"{label}{s.value_text}", bbox=bbox, block_type="text"))
    return blocks


def ingest(
    path,
    instruction: Optional[str] = None,
    backend: Optional[str] = None,
    cfg: Optional[Config] = None,
    graph: Optional[bool] = None,
) -> dict:
    cfg = cfg or Config()
    if backend:
        cfg = cfg.model_copy(update={"extraction_backend": backend})
    # Per-ingest graph override: graph=False excludes THIS document from the knowledge
    # graph even when cfg.build_graph is True. None -> fall back to the config default.
    build_graph = cfg.build_graph if graph is None else graph
    path = Path(path)
    instruction = instruction or "Extract all key information and every statistic with its label."

    content_hash = _content_hash(path)
    pages = extract_layout(path)
    doc_id = _doc_id(path, content_hash)

    all_chunks: list[Chunk] = []
    all_stats: list[Stat] = []
    pages_vision = 0
    pages_textonly = 0
    pages_skipped = 0

    for page in pages:
        # Skip sparse decorative pages (cover, section divider) — but never a page that
        # carries a statistic. force_vision keeps everything.
        if not cfg.force_vision and page_should_skip(page):
            pages_skipped += 1
            continue

        if needs_vision(page, force_vision=cfg.force_vision):
            pages_vision += 1
            png = render_page(path, page.page, dpi=cfg.render_dpi_for())
            curated, stats = extract_from_image(
                png, instruction, backend=cfg.extraction_backend, model=cfg.vision_model_for()
            )
            # stamp the real page number onto each stat
            for s in stats:
                s.page = page.page
            stats = verify_stats(path, stats, cfg)
            all_stats.extend(stats)
            blocks = _vision_blocks(curated, stats, page)
        else:
            pages_textonly += 1
            blocks = [b for b in page.blocks if b.block_type != "image"]
            stats = []

        chunks = chunk_page(
            doc_id=doc_id,
            source_path=str(path),
            page=page.page,
            blocks=blocks,
            stats=stats,
            size=cfg.chunk_size,
            overlap=cfg.chunk_overlap,
        )
        all_chunks.extend(chunks)

    # Embed + store
    if all_chunks:
        vectors = embed_texts([c.text for c in all_chunks], model=cfg.embed_model)
        table = vector.open_store(cfg, dim=len(vectors[0]), embed_model=cfg.embed_model)
        vector.upsert_chunks(table, all_chunks, vectors)

    # Knowledge graph (optional, per-ingest gated). Extract entities from each chunk and
    # link them into the graph so this document unifies with others by shared entities.
    entity_count = 0
    graph_built = False
    if build_graph and all_chunks:
        all_entities: dict[str, Entity] = {}
        mentions: list[EntityMention] = []
        for ch in all_chunks:
            for e in extract_entities(ch, backend=cfg.entity_backend, cfg=cfg):
                all_entities.setdefault(e.key, e)
                mentions.append(
                    EntityMention(
                        chunk_id=ch.chunk_id, doc_id=doc_id, entity_key=e.key,
                        entity_name=e.name, etype=e.etype, page=ch.page,
                    )
                )
        g = graph_store.open_graph(cfg)
        graph_store.add_document(
            g, doc_id=doc_id, source_path=str(path),
            chunks=all_chunks, entities=list(all_entities.values()), mentions=mentions,
        )
        entity_count = len(all_entities)
        graph_built = True

    unverified = sum(1 for s in all_stats if s.verified is not True)

    return {
        "doc_id": doc_id,
        "content_hash": content_hash,
        "page_count": len(pages),
        "pages_vision": pages_vision,
        "pages_textonly": pages_textonly,
        "pages_skipped": pages_skipped,
        "chunk_count": len(all_chunks),
        "stat_count": len(all_stats),
        "unverified_count": unverified,
        "graph_built": graph_built,
        "entity_count": entity_count,
    }
