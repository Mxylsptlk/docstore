"""docstore — local PDF document-store with vision-based statistics extraction.

Public API:
    ingest(path, instruction=None, backend=None, cfg=None, graph=None) -> dict
    retrieve(query, k=5, cfg=None) -> list[RetrievalResult]
    ask(query, k=5, cfg=None) -> str  (path to ANSWER.md)

Knowledge-graph queries (cfg-driven):
    documents_mentioning(entity, cfg=None) -> list[str]
    related_documents(doc_id, cfg=None) -> list[str]
    entities_in_document(doc_id, cfg=None) -> list[str]
    chunks_mentioning(entity, cfg=None) -> list[str]
"""

from docstore.ingest import ingest
from docstore.retrieve import retrieve
from docstore.ask import ask
from docstore.graph_query import (
    chunks_mentioning,
    documents_mentioning,
    entities_in_document,
    related_documents,
)

__all__ = [
    "ingest",
    "retrieve",
    "ask",
    "documents_mentioning",
    "related_documents",
    "entities_in_document",
    "chunks_mentioning",
]
__version__ = "0.2.0"
