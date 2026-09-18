"""docstore — local PDF document-store with vision-based statistics extraction.

Public API (lazily exposed so submodules can be built/tested independently):
    ingest(path, instruction=None, backend=None, cfg=None) -> dict
    retrieve(query, k=5, cfg=None) -> list[RetrievalResult]
    ask(query, k=5, cfg=None) -> str  (path to ANSWER.md)
"""

import importlib
from typing import Any

__all__ = ["ingest", "retrieve", "ask"]
__version__ = "0.1.0"

_LAZY = {"ingest": "docstore.ingest", "retrieve": "docstore.retrieve", "ask": "docstore.ask"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module 'docstore' has no attribute {name!r}")
