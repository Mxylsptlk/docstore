"""Kùzu knowledge graph store.

Embedded, file-based graph DB (no server) living under `graph_dir`. It complements the
vector store: vectors answer "what text is similar?", the graph answers "what is
connected?" — cross-document entity links, multi-hop questions, and explainable paths.

Schema:
    (Document {doc_id, source_path})
    (Chunk    {chunk_id, doc_id, page})
    (Entity   {key, name, etype})
    (Document)-[:HAS_CHUNK]->(Chunk)
    (Chunk)-[:MENTIONS]->(Entity)
    (Document)-[:REFERENCES {count}]->(Entity)   # doc-level rollup for fast joins

Entities unify on their normalized `key`, so the same org/date/amount across many PDFs
becomes ONE node — that unification is the whole point of the graph. Chunk ids are the
stable ids from the vector store, so the two stores join on chunk_id without re-ingest.

All writes use MERGE and are idempotent: re-ingesting a document does not duplicate nodes
or edges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import kuzu

from docstore.config import Config
from docstore.models import Chunk, Entity, EntityMention

# Process-wide cache of one open graph handle per resolved path (see open_graph).
_GRAPH_CACHE: dict[str, "Graph"] = {}

_SCHEMA = [
    "CREATE NODE TABLE IF NOT EXISTS Document(doc_id STRING, source_path STRING, PRIMARY KEY(doc_id))",
    "CREATE NODE TABLE IF NOT EXISTS Chunk(chunk_id STRING, doc_id STRING, page INT64, PRIMARY KEY(chunk_id))",
    "CREATE NODE TABLE IF NOT EXISTS Entity(key STRING, name STRING, etype STRING, PRIMARY KEY(key))",
    "CREATE REL TABLE IF NOT EXISTS HAS_CHUNK(FROM Document TO Chunk)",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Chunk TO Entity)",
    "CREATE REL TABLE IF NOT EXISTS REFERENCES(FROM Document TO Entity, count INT64)",
]


class Graph:
    """Thin handle around a Kùzu database + connection."""

    def __init__(self, db: kuzu.Database, conn: kuzu.Connection):
        self.db = db
        self.conn = conn

    def execute(self, query: str, params: dict[str, Any] | None = None):
        return self.conn.execute(query, params or {})


def open_graph(cfg: Config) -> Graph:
    """Open (creating if needed) the on-disk graph and ensure the schema exists.

    Kùzu allows only ONE `Database` instance per path within a process (a second
    concurrent instance silently loses writes). So we cache one Database+Connection per
    resolved path and hand back the same handle — callers can open freely across many
    ingest() calls without corrupting the graph.
    """
    path = Path(cfg.graph_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path.resolve())

    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached

    db = kuzu.Database(key)
    conn = kuzu.Connection(db)
    for ddl in _SCHEMA:
        conn.execute(ddl)
    g = Graph(db, conn)
    _GRAPH_CACHE[key] = g
    return g


def close_all() -> None:
    """Drop all cached graph handles (releases the on-disk locks). Mainly for tests."""
    _GRAPH_CACHE.clear()


def add_document(
    g: Graph,
    *,
    doc_id: str,
    source_path: str,
    chunks: list[Chunk],
    entities: list[Entity],
    mentions: list[EntityMention],
) -> None:
    """Idempotently add a document, its chunks, its entities, and their links."""
    g.execute(
        "MERGE (d:Document {doc_id: $doc_id}) SET d.source_path = $source_path",
        {"doc_id": doc_id, "source_path": source_path},
    )
    for ch in chunks:
        g.execute(
            "MERGE (c:Chunk {chunk_id: $cid}) SET c.doc_id = $doc_id, c.page = $page",
            {"cid": ch.chunk_id, "doc_id": doc_id, "page": ch.page},
        )
        g.execute(
            "MATCH (d:Document {doc_id: $doc_id}), (c:Chunk {chunk_id: $cid}) "
            "MERGE (d)-[:HAS_CHUNK]->(c)",
            {"doc_id": doc_id, "cid": ch.chunk_id},
        )
    for e in entities:
        g.execute(
            "MERGE (e:Entity {key: $key}) SET e.name = $name, e.etype = $etype",
            {"key": e.key, "name": e.name, "etype": e.etype},
        )
    for m in mentions:
        g.execute(
            "MATCH (c:Chunk {chunk_id: $cid}), (e:Entity {key: $key}) "
            "MERGE (c)-[:MENTIONS]->(e)",
            {"cid": m.chunk_id, "key": m.entity_key},
        )
    # Document-level rollup: count of distinct chunks mentioning each entity.
    counts: dict[str, int] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for m in mentions:
        pair = (m.chunk_id, m.entity_key)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        counts[m.entity_key] = counts.get(m.entity_key, 0) + 1
    for key, count in counts.items():
        g.execute(
            "MATCH (d:Document {doc_id: $doc_id}), (e:Entity {key: $key}) "
            "MERGE (d)-[r:REFERENCES]->(e) SET r.count = $count",
            {"doc_id": doc_id, "key": key, "count": count},
        )


def _collect(result, idx: int = 0) -> list[Any]:
    out: list[Any] = []
    while result.has_next():
        out.append(result.get_next()[idx])
    return out


def entities_for_document(g: Graph, doc_id: str) -> list[str]:
    """Entity keys referenced by a document."""
    res = g.execute(
        "MATCH (d:Document {doc_id: $doc_id})-[:REFERENCES]->(e:Entity) RETURN e.key",
        {"doc_id": doc_id},
    )
    return _collect(res)


def documents_for_entity(g: Graph, entity_key: str) -> list[str]:
    """Document ids that reference an entity (cross-document unification)."""
    res = g.execute(
        "MATCH (d:Document)-[:REFERENCES]->(e:Entity {key: $key}) RETURN d.doc_id ORDER BY d.doc_id",
        {"key": entity_key},
    )
    return _collect(res)


def related_documents(g: Graph, doc_id: str) -> list[str]:
    """Documents sharing at least one entity with the given document (never itself)."""
    res = g.execute(
        "MATCH (d:Document {doc_id: $doc_id})-[:REFERENCES]->(e:Entity)"
        "<-[:REFERENCES]-(other:Document) "
        "WHERE other.doc_id <> $doc_id "
        "RETURN DISTINCT other.doc_id ORDER BY other.doc_id",
        {"doc_id": doc_id},
    )
    return _collect(res)


def chunks_for_entity(g: Graph, entity_key: str) -> list[str]:
    """Chunk ids that mention an entity (join key back to the vector store)."""
    res = g.execute(
        "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {key: $key}) RETURN c.chunk_id",
        {"key": entity_key},
    )
    return _collect(res)
