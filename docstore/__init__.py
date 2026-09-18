"""docstore — local PDF document-store with vision-based statistics extraction.

Public API:
    ingest(path, instruction=None, backend=None, cfg=None) -> dict
    retrieve(query, k=5, cfg=None) -> list[RetrievalResult]
    ask(query, k=5, cfg=None) -> str  (path to ANSWER.md)
"""

from docstore.ingest import ingest
from docstore.retrieve import retrieve
from docstore.ask import ask

__all__ = ["ingest", "retrieve", "ask"]
__version__ = "0.1.0"
