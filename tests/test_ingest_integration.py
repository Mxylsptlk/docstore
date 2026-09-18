"""Tests for docstore.ingest (full pipeline orchestration)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from docstore.config import Config
from docstore.extract import Page
from docstore.models import Block, Chunk, Stat

FIX = Path(__file__).parent / "fixtures"


def _pages():
    text_page = Page(
        page=1,
        blocks=[Block(text="Lots of ordinary prose words here describing the quarter in detail "
                           "across many sentences and clauses.", bbox=(0, 0, 500, 40), block_type="text")],
        width=612, height=792,
    )
    # Image page carrying a real stat callout -> kept and routed to vision.
    stat_page = Page(
        page=2,
        blocks=[
            Block(text="On-time rate", bbox=(0, 0, 100, 20), block_type="text"),
            Block(text="42%", bbox=(0, 30, 60, 60), block_type="text"),
            Block(text="", bbox=(0, 70, 200, 240), block_type="image"),
        ],
        width=612, height=792,
    )
    # Sparse decorative divider (no stats) -> should be skipped.
    divider_page = Page(
        page=3,
        blocks=[Block(text="Section Two", bbox=(72, 60, 300, 90), block_type="text")],
        width=612, height=792,
    )
    return [text_page, stat_page, divider_page]


def test_ingest_pipeline_mocked(tmp_path: Path):
    cfg = Config(data_dir=str(tmp_path / "data"))

    stat = Stat(value_text="42%", label="On-time rate", page=2, bbox=(0, 0, 100, 20), verified=True)
    chunk = Chunk(chunk_id="c1", doc_id="d", source_path="x", page=2, bbox=(0, 0, 1, 1),
                  section="", char_start=0, char_end=3, text="42%", is_stat=True, stat_verified=True)

    with patch("docstore.ingest.extract_layout", return_value=_pages()) as m_extract, \
         patch("docstore.ingest.needs_vision", side_effect=[False, True]) as m_gate, \
         patch("docstore.ingest.render_page", return_value=b"PNG") as m_render, \
         patch("docstore.ingest.extract_from_image", return_value=("On-time rate 42%", [stat])) as m_vision, \
         patch("docstore.ingest.verify_stats", side_effect=lambda p, s, c: s) as m_verify, \
         patch("docstore.ingest.chunk_page", return_value=[chunk]) as m_chunk, \
         patch("docstore.ingest.embed_texts", return_value=[[0.1, 0.2, 0.3]]) as m_embed, \
         patch("docstore.ingest.vector.open_store", return_value="TABLE") as m_open, \
         patch("docstore.ingest.vector.upsert_chunks") as m_upsert:
        from docstore.ingest import ingest
        summary = ingest(FIX / "sample_layout.pdf", instruction="pull all stats", cfg=cfg)

    # gate consulted per non-skipped page (pages 1 and 2; page 3 skipped before the gate)
    assert m_gate.call_count == 2
    # vision only on the gated-in page (page 2)
    assert m_vision.call_count == 1
    m_render.assert_called_once()  # only rendered the vision page
    m_verify.assert_called_once()
    m_embed.assert_called_once()
    m_upsert.assert_called_once()

    assert summary["pages_vision"] == 1
    assert summary["pages_textonly"] == 1
    assert summary["pages_skipped"] == 1
    assert summary["chunk_count"] >= 1
    assert summary["stat_count"] == 1
    assert summary["unverified_count"] == 0
    assert "doc_id" in summary


def test_public_api_exports():
    import docstore
    assert callable(docstore.ingest)
    assert callable(docstore.retrieve)
    assert callable(docstore.ask)


def test_force_vision_sends_all_pages(tmp_path: Path):
    cfg = Config(data_dir=str(tmp_path / "data"), force_vision=True)
    stat = Stat(value_text="42%", label="x", page=1, bbox=None, verified=None)
    chunk = Chunk(chunk_id="c1", doc_id="d", source_path="x", page=1, bbox=(0, 0, 1, 1),
                  section="", char_start=0, char_end=3, text="42%", is_stat=True)
    with patch("docstore.ingest.extract_layout", return_value=_pages()), \
         patch("docstore.ingest.render_page", return_value=b"PNG"), \
         patch("docstore.ingest.extract_from_image", return_value=("t", [stat])) as m_vision, \
         patch("docstore.ingest.verify_stats", side_effect=lambda p, s, c: s), \
         patch("docstore.ingest.chunk_page", return_value=[chunk]), \
         patch("docstore.ingest.embed_texts", return_value=[[0.1, 0.2, 0.3]]), \
         patch("docstore.ingest.vector.open_store", return_value="TABLE"), \
         patch("docstore.ingest.vector.upsert_chunks"):
        from docstore.ingest import ingest
        summary = ingest(FIX / "sample_layout.pdf", instruction="x", cfg=cfg)
    assert m_vision.call_count == 3
    assert summary["pages_vision"] == 3
    assert summary["pages_skipped"] == 0
