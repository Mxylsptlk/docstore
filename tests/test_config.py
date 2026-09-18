"""Tests for docstore.config."""

import textwrap
from pathlib import Path

import pytest

from docstore.config import Config, load_config


@pytest.fixture
def sample_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        textwrap.dedent(
            """
            embed_model: nomic-embed-text
            extraction_backend: claude
            vision_model: latest
            render_dpi: 220
            chunk_size: 1200
            chunk_overlap: 150
            data_dir: data
            table_name: chunks
            answer_path: ANSWER.md
            verify_stats: true
            force_vision: false
            graph_dir: graph
            build_graph: true
            entity_backend: regex
            entity_model: llama3.1
            """
        )
    )
    return p


def test_config_loads_defaults(sample_yaml: Path):
    cfg = load_config(sample_yaml)
    assert cfg.embed_model == "nomic-embed-text"
    assert cfg.extraction_backend == "claude"
    assert cfg.render_dpi >= 200
    assert cfg.chunk_size > 0
    assert cfg.data_dir  # set
    assert cfg.verify_stats is True
    assert cfg.force_vision is False
    assert cfg.build_graph is True
    assert cfg.entity_backend == "regex"


def test_config_has_no_secret_fields():
    # Secrets must never live in Config; they come from the environment.
    field_names = set(Config.model_fields)
    for forbidden in ("api_key", "anthropic_api_key", "key", "token", "secret"):
        assert forbidden not in field_names


def test_load_config_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_config_defaults_without_yaml():
    # A bare Config() should be usable with sane defaults.
    cfg = Config()
    assert cfg.extraction_backend == "claude"
    assert cfg.verify_stats is True
    assert cfg.force_vision is False


def test_extraction_backend_validated():
    with pytest.raises(Exception):
        Config(extraction_backend="openai")
