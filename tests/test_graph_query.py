"""Tests for docstore.graph_query (public graph query helpers)."""

from pathlib import Path

import pytest

from docstore.config import Config
from docstore.graph_query import (
    documents_mentioning,
    entities_in_document,
    related_documents,
)
from docstore.models import Chunk, Entity, EntityMention
from docstore.stores import graph as gstore


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    gstore.close_all()
    c = Config(graph_dir=str(tmp_path / "graph"))
    # seed: docA & docB share Acme; docC alone
    g = gstore.open_graph(c)
    seed = {
        "docA": [Entity(name="Acme", key="acme", etype="ORG"),
                 Entity(name="Oak", key="oak", etype="PROJECT")],
        "docB": [Entity(name="Acme", key="acme", etype="ORG")],
        "docC": [Entity(name="Zeta", key="zeta", etype="ORG")],
    }
    for doc, ents in seed.items():
        ch = Chunk(chunk_id=f"{doc}_c1", doc_id=doc, source_path=f"/tmp/{doc}.pdf", page=1,
                   bbox=(0, 0, 1, 1), section="", char_start=0, char_end=4, text="t")
        mens = [EntityMention(chunk_id=ch.chunk_id, doc_id=doc, entity_key=e.key,
                              entity_name=e.name, etype=e.etype, page=1) for e in ents]
        gstore.add_document(g, doc_id=doc, source_path=f"/tmp/{doc}.pdf",
                            chunks=[ch], entities=ents, mentions=mens)
    return c


def test_entities_in_document(cfg):
    ents = entities_in_document("docA", cfg=cfg)
    assert set(ents) == {"acme", "oak"}


def test_documents_mentioning(cfg):
    docs = documents_mentioning("acme", cfg=cfg)
    assert set(docs) == {"docA", "docB"}


def test_related_documents(cfg):
    rel = related_documents("docA", cfg=cfg)
    assert rel == ["docB"]
