"""Tests for docstore.entities (pluggable entity extraction)."""

from unittest.mock import patch

import pytest

from docstore.config import Config
from docstore.entities import extract_entities
from docstore.models import Chunk, Entity

CHUNK = Chunk(
    chunk_id="c1", doc_id="d", source_path="/tmp/f.pdf", page=3, bbox=(0, 0, 1, 1),
    section="", char_start=0, char_end=80,
    text="Acme Corporation was awarded $4.2M on 03/15/2024 for the Oakdale project.",
)


def test_regex_extracts_money_date_org():
    ents = extract_entities(CHUNK, backend="regex", cfg=Config())
    keys = {(e.etype, e.key) for e in ents}
    assert ("MONEY", "$4.2m") in keys
    assert any(e.etype == "DATE" for e in ents)
    assert any(e.etype == "ORG" and "acme" in e.key for e in ents)
    assert all(isinstance(e, Entity) for e in ents)


def test_regex_dedupes_within_chunk():
    c = Chunk(chunk_id="c2", doc_id="d", source_path="x", page=1, bbox=(0, 0, 1, 1),
              section="", char_start=0, char_end=60,
              text="Acme Corp won the bid. Acme Corp then delivered. Acme Corp again.")
    ents = extract_entities(c, backend="regex", cfg=Config())
    acme = [e for e in ents if "acme corp" in e.key]
    assert len(acme) == 1  # one node per canonical key


def test_regex_needs_no_network():
    # regex backend must never touch ollama/claude
    with patch("docstore.entities._extract_ollama") as mo, \
         patch("docstore.entities._extract_claude") as mc:
        extract_entities(CHUNK, backend="regex", cfg=Config())
    mo.assert_not_called()
    mc.assert_not_called()


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="entity backend"):
        extract_entities(CHUNK, backend="nope", cfg=Config())


def test_ollama_backend_dispatch():
    fake = [Entity(name="Acme", key="acme", etype="ORG")]
    with patch("docstore.entities._extract_ollama", return_value=fake) as mo:
        ents = extract_entities(CHUNK, backend="ollama", cfg=Config())
    mo.assert_called_once()
    assert ents == fake


def test_claude_backend_dispatch():
    fake = [Entity(name="Oakdale", key="oakdale", etype="PROJECT")]
    with patch("docstore.entities._extract_claude", return_value=fake) as mc:
        ents = extract_entities(CHUNK, backend="claude", cfg=Config())
    mc.assert_called_once()
    assert ents == fake


def test_default_backend_from_config():
    cfg = Config(entity_backend="regex")
    ents = extract_entities(CHUNK, backend=None, cfg=cfg)
    assert any(e.etype == "MONEY" for e in ents)


def test_leading_article_unifies_entity_key():
    # Regression: 'The Oakdale Corporation' and 'Oakdale Corporation' must produce the
    # SAME canonical key, or the same org splits into two graph nodes and never unifies
    # across documents. (Found via live cross-document check.)
    from docstore.models import normalize_entity_name

    assert normalize_entity_name("The Oakdale Corporation") == "oakdale corporation"
    assert normalize_entity_name("Oakdale Corporation") == "oakdale corporation"
    assert normalize_entity_name("An Acme Corp.") == "acme corp"

    c_the = Chunk(chunk_id="ca", doc_id="d", source_path="x", page=1, bbox=(0, 0, 1, 1),
                  section="", char_start=0, char_end=60,
                  text="The Oakdale Corporation completed the work on schedule.")
    c_bare = Chunk(chunk_id="cb", doc_id="d", source_path="x", page=1, bbox=(0, 0, 1, 1),
                   section="", char_start=0, char_end=60,
                   text="Oakdale Corporation reported strong margins this quarter.")
    keys_the = {e.key for e in extract_entities(c_the, backend="regex", cfg=Config()) if e.etype == "ORG"}
    keys_bare = {e.key for e in extract_entities(c_bare, backend="regex", cfg=Config()) if e.etype == "ORG"}
    assert "oakdale corporation" in keys_the
    assert "oakdale corporation" in keys_bare
