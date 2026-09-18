"""Retrieval: embed the query, search the vector store, return cited results.

Each result carries a resolvable citation (source.pdf:pN), its bbox, and — crucially —
the statistic verification status, so the answering step can warn when a retrieved
number was never verified or failed verification.
"""

from __future__ import annotations

import warnings
from typing import Optional

from docstore.config import Config
from docstore.embed import embed_query
from docstore.models import RetrievalResult
from docstore.stores import vector


def retrieve(query: str, k: int = 5, cfg: Optional[Config] = None) -> list[RetrievalResult]:
    cfg = cfg or Config()

    # Warn (don't crash) on an embedding-model mismatch vs. what the store was built with.
    built_with = vector.stored_embed_model(cfg)
    if built_with is not None and built_with != cfg.embed_model:
        warnings.warn(
            f"Store was built with embed_model={built_with!r} but config uses "
            f"{cfg.embed_model!r}; results may be meaningless. Re-ingest to fix.",
            stacklevel=2,
        )

    qvec = embed_query(query, model=cfg.embed_model)
    table = vector.open_store(cfg, dim=len(qvec))
    rows = vector.search(table, qvec, k=k)

    results: list[RetrievalResult] = []
    for r in rows:
        results.append(
            RetrievalResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                source_path=r["source_path"],
                page=int(r["page"]),
                bbox=r.get("bbox"),
                is_stat=bool(r.get("is_stat", False)),
                stat_verified=r.get("stat_verified"),
                score=float(r.get("score", 0.0)),
            )
        )
    return results
