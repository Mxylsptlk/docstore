"""Configuration for docstore.

Secrets (e.g. ANTHROPIC_API_KEY) are NEVER stored here — they are read from the
environment at call time. This module only holds non-secret runtime settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

ExtractionBackend = Literal["claude", "ollama"]
EntityBackend = Literal["regex", "ollama", "claude"]


class Config(BaseModel):
    """Non-secret runtime configuration for docstore."""

    model_config = {"extra": "forbid"}

    # Embeddings (local, via Ollama)
    embed_model: str = "nomic-embed-text"

    # Vision extraction backend
    extraction_backend: ExtractionBackend = "claude"
    vision_model: str = "latest"

    # Rendering
    render_dpi: int = Field(default=220, ge=72)

    # Chunking
    chunk_size: int = Field(default=1200, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    # Storage / paths
    data_dir: str = "data"
    table_name: str = "chunks"
    answer_path: str = "ANSWER.md"
    graph_dir: str = "graph"

    # Behavior
    verify_stats: bool = True
    force_vision: bool = False

    # Knowledge graph
    build_graph: bool = True
    entity_backend: EntityBackend = "regex"
    entity_model: str = "llama3.1"  # used when entity_backend == "ollama"

    def anthropic_api_key(self) -> str:
        """Read the Anthropic key from the environment at call time.

        Never stored on the Config object. Raises a clear error if unset.
        """
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it in your environment "
                "before using the Claude extraction backend."
            )
        return key


def load_config(path: str | Path) -> Config:
    """Load a Config from a YAML file. Raises FileNotFoundError if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    return Config(**data)
