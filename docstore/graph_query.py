"""Public knowledge-graph query helpers.

Thin, cfg-driven wrappers over docstore.stores.graph so callers can ask connectivity
questions without touching the store directly. These are the graph counterpart to
`retrieve` (which does semantic search): they answer "what is connected to what".

    entities_in_document(doc_id)      -> entity keys a document references
    documents_mentioning(entity_key)  -> documents that reference an entity (cross-doc)
    related_documents(doc_id)         -> documents sharing >=1 entity (never itself)
    chunks_mentioning(entity_key)     -> chunk ids mentioning an entity (join to vectors)
"""

from __future__ import annotations

from typing import Optional

from docstore.config import Config
from docstore.models import normalize_entity_name
from docstore.stores import graph as _graph


def entities_in_document(doc_id: str, cfg: Optional[Config] = None) -> list[str]:
    g = _graph.open_graph(cfg or Config())
    return _graph.entities_for_document(g, doc_id)


def documents_mentioning(entity: str, cfg: Optional[Config] = None) -> list[str]:
    """Documents referencing `entity` (matched on its normalized key, so 'Acme Corp.'
    and 'acme corp' resolve the same)."""
    g = _graph.open_graph(cfg or Config())
    return _graph.documents_for_entity(g, normalize_entity_name(entity))


def related_documents(doc_id: str, cfg: Optional[Config] = None) -> list[str]:
    g = _graph.open_graph(cfg or Config())
    return _graph.related_documents(g, doc_id)


def chunks_mentioning(entity: str, cfg: Optional[Config] = None) -> list[str]:
    g = _graph.open_graph(cfg or Config())
    return _graph.chunks_for_entity(g, normalize_entity_name(entity))
