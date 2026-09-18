"""Tests for docstore.stores.graph (Kùzu knowledge graph)."""

from pathlib import Path

import pytest

from docstore.config import Config
from docstore.models import Chunk, Entity, EntityMention
from docstore.stores import graph


def _chunk(cid: str, doc: str, page: int, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc, source_path=f"/tmp/{doc}.pdf", page=page,
                 bbox=(0, 0, 1, 1), section="", char_start=0, char_end=len(text), text=text)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(graph_dir=str(tmp_path / "graph"))


def test_add_document_and_query_entities(cfg):
    g = graph.open_graph(cfg)
    chunk = _chunk("c1", "docA", 3, "Acme Corp won $4.2M for Oakdale.")
    ents = [
        Entity(name="Acme Corp", key="acme corp", etype="ORG"),
        Entity(name="$4.2M", key="$4.2m", etype="MONEY"),
    ]
    mentions = [
        EntityMention(chunk_id="c1", doc_id="docA", entity_key=e.key,
                      entity_name=e.name, etype=e.etype, page=3)
        for e in ents
    ]
    graph.add_document(g, doc_id="docA", source_path="/tmp/docA.pdf",
                       chunks=[chunk], entities=ents, mentions=mentions)

    names = graph.entities_for_document(g, "docA")
    assert "acme corp" in names
    assert "$4.2m" in names


def test_cross_document_shared_entity(cfg):
    g = graph.open_graph(cfg)
    for doc in ("docA", "docB"):
        ch = _chunk(f"{doc}_c1", doc, 1, "Acme Corp appears here.")
        ent = Entity(name="Acme Corp", key="acme corp", etype="ORG")
        men = EntityMention(chunk_id=ch.chunk_id, doc_id=doc, entity_key=ent.key,
                            entity_name=ent.name, etype="ORG", page=1)
        graph.add_document(g, doc_id=doc, source_path=f"/tmp/{doc}.pdf",
                           chunks=[ch], entities=[ent], mentions=[men])

    docs = graph.documents_for_entity(g, "acme corp")
    assert set(docs) == {"docA", "docB"}


def test_idempotent_reingest(cfg):
    g = graph.open_graph(cfg)
    ch = _chunk("c1", "docA", 1, "Acme Corp here.")
    ent = Entity(name="Acme Corp", key="acme corp", etype="ORG")
    men = EntityMention(chunk_id="c1", doc_id="docA", entity_key="acme corp",
                        entity_name="Acme Corp", etype="ORG", page=1)
    for _ in range(2):  # ingest the same doc twice
        graph.add_document(g, doc_id="docA", source_path="/tmp/docA.pdf",
                           chunks=[ch], entities=[ent], mentions=[men])
    # still exactly one document node linked to the entity
    docs = graph.documents_for_entity(g, "acme corp")
    assert docs == ["docA"]


def test_related_documents(cfg):
    g = graph.open_graph(cfg)
    # docA and docB share Acme; docC is unrelated
    data = {
        "docA": [Entity(name="Acme", key="acme", etype="ORG"),
                 Entity(name="Oak", key="oak", etype="PROJECT")],
        "docB": [Entity(name="Acme", key="acme", etype="ORG")],
        "docC": [Entity(name="Zeta", key="zeta", etype="ORG")],
    }
    for doc, ents in data.items():
        mens = [EntityMention(chunk_id=f"{doc}_c1", doc_id=doc, entity_key=e.key,
                              entity_name=e.name, etype=e.etype, page=1) for e in ents]
        ch = _chunk(f"{doc}_c1", doc, 1, "text")
        graph.add_document(g, doc_id=doc, source_path=f"/tmp/{doc}.pdf",
                           chunks=[ch], entities=ents, mentions=mens)

    related = graph.related_documents(g, "docA")
    assert "docB" in related       # shares Acme
    assert "docC" not in related   # shares nothing
    assert "docA" not in related   # never itself


def test_persists_across_reopen(cfg):
    g = graph.open_graph(cfg)
    ch = _chunk("c1", "docA", 1, "Acme Corp here.")
    ent = Entity(name="Acme Corp", key="acme corp", etype="ORG")
    men = EntityMention(chunk_id="c1", doc_id="docA", entity_key="acme corp",
                        entity_name="Acme Corp", etype="ORG", page=1)
    graph.add_document(g, doc_id="docA", source_path="/tmp/docA.pdf",
                       chunks=[ch], entities=[ent], mentions=[men])
    graph.close_all()  # force a genuine reopen from disk (drop cached handle)
    g2 = graph.open_graph(cfg)  # reopen the on-disk graph
    assert "acme corp" in graph.entities_for_document(g2, "docA")
