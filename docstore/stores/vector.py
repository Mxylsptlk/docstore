"""LanceDB vector store.

Embedded, file-backed, zero-server — the whole store lives under `data_dir`. Holds each
chunk's embedding alongside its full provenance (page, bbox, is_stat, stat_verified) so a
search hit can be cited and its statistic's trust level surfaced. Dedupes on chunk_id.

The embedding model name is recorded next to the table so a later run can detect a model
mismatch (which would require re-embedding).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import lancedb
import pyarrow as pa

from docstore.config import Config
from docstore.models import Chunk


def _db_path(cfg: Config) -> Path:
    p = Path(cfg.data_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _meta_path(cfg: Config) -> Path:
    return _db_path(cfg) / f"{cfg.table_name}.meta.json"


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("bbox", pa.list_(pa.float32(), 4)),
            pa.field("section", pa.string()),
            pa.field("char_start", pa.int32()),
            pa.field("char_end", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("is_stat", pa.bool_()),
            # stat_verified is tri-state (True/False/None) -> store as int8: 1/0/-1
            pa.field("stat_verified", pa.int8()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def open_store(cfg: Config, *, dim: int, embed_model: Optional[str] = None):
    """Open (creating if needed) the LanceDB table. Returns the table handle.

    Robust to a fresh connection not yet seeing an existing table: try to open first,
    and only create on a genuine miss.
    """
    db = lancedb.connect(str(_db_path(cfg)))
    try:
        table = db.open_table(cfg.table_name)
    except Exception:  # noqa: BLE001 — table doesn't exist yet
        try:
            table = db.create_table(cfg.table_name, schema=_schema(dim))
        except Exception:  # noqa: BLE001 — lost a create race; open the winner
            table = db.open_table(cfg.table_name)
    if embed_model is not None:
        _meta_path(cfg).write_text(json.dumps({"embed_model": embed_model, "dim": dim}))
    return table


def stored_embed_model(cfg: Config) -> Optional[str]:
    mp = _meta_path(cfg)
    if not mp.exists():
        return None
    return json.loads(mp.read_text()).get("embed_model")


def _tri_to_int(v: Optional[bool]) -> int:
    return 1 if v is True else (0 if v is False else -1)


def _int_to_tri(v: int) -> Optional[bool]:
    return True if v == 1 else (False if v == 0 else None)


def _row(chunk: Chunk, vec: list[float]) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source_path": chunk.source_path,
        "page": int(chunk.page),
        "bbox": [float(x) for x in chunk.bbox],
        "section": chunk.section,
        "char_start": int(chunk.char_start),
        "char_end": int(chunk.char_end),
        "text": chunk.text,
        "is_stat": bool(chunk.is_stat),
        "stat_verified": _tri_to_int(chunk.stat_verified),
        "vector": [float(x) for x in vec],
    }


def upsert_chunks(table, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """Insert chunks, replacing any existing rows with the same chunk_id (dedupe)."""
    if not chunks:
        return
    ids = [c.chunk_id for c in chunks]
    id_list = ", ".join(f"'{i}'" for i in ids)
    table.delete(f"chunk_id IN ({id_list})")
    table.add([_row(c, v) for c, v in zip(chunks, vectors)])


def search(table, query_vector: list[float], k: int = 5) -> list[dict[str, Any]]:
    """Return up to k nearest rows as plain dicts, with tri-state verification restored."""
    rows = table.search(query_vector).limit(k).to_list()
    out: list[dict[str, Any]] = []
    for r in rows:
        r = dict(r)
        r["stat_verified"] = _int_to_tri(int(r.get("stat_verified", -1)))
        r["is_stat"] = bool(r.get("is_stat", False))
        bbox = r.get("bbox")
        r["bbox"] = tuple(float(x) for x in bbox) if bbox is not None else None
        # LanceDB returns the distance as _distance; expose it as score.
        if "_distance" in r:
            r["score"] = float(r["_distance"])
        out.append(r)
    return out
