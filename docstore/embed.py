"""Local embeddings via Ollama.

Embeddings are the one model kept fully local (per the design): a lightweight Ollama
model turns text into vectors at ingest time (per chunk) and at query time (the
question). Nothing here writes prose — it only measures meaning.
"""

from __future__ import annotations

import ollama


def _embed_one(text: str, model: str) -> list[float]:
    try:
        resp = ollama.embeddings(model=model, prompt=text)
    except Exception as e:  # noqa: BLE001 — surface a clear, actionable message
        raise RuntimeError(
            f"Ollama embedding call failed for model {model!r}: {e}. "
            "Is `ollama serve` running and the model pulled "
            f"(`ollama pull {model}`)?"
        ) from e
    return list(resp["embedding"])


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    """Embed a list of texts, returning one vector each. Empty input -> empty output."""
    return [_embed_one(t, model) for t in texts]


def embed_query(text: str, model: str) -> list[float]:
    """Embed a single query string."""
    return _embed_one(text, model)
