"""Resolve a model-name sentinel to a concrete Anthropic model id.

Anthropic does not publish a `-latest` alias for every model family (notably Sonnet),
so `vision_model: latest` in config is a docstore sentinel we resolve ourselves by
querying the live model list and picking the newest model in the requested family.

Sentinels (case-insensitive):
    "latest"            -> newest Sonnet
    "latest:sonnet"     -> newest Sonnet
    "latest:opus"       -> newest Opus
    "latest:haiku"      -> newest Haiku

Anything else (e.g. "claude-sonnet-5", a dated snapshot) is returned unchanged.

Resolution hits the API once per (family) per process and is cached. If the API is
unreachable or returns nothing usable, we fall back to a pinned known-good id so
ingestion never hard-fails on model discovery.
"""

from __future__ import annotations

from typing import Optional

# Pinned fallbacks used only when live resolution fails (no key / offline / empty list).
# Update these when the family's newest stable id changes.
_FALLBACK = {
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-4",
    "haiku": "claude-haiku-4",
}

_DEFAULT_FAMILY = "sonnet"

# Process-wide cache: family -> resolved id.
_RESOLVED: dict[str, str] = {}


def _parse_sentinel(model: str) -> Optional[str]:
    """Return the family to resolve if `model` is a 'latest' sentinel, else None."""
    m = model.strip().lower()
    if m == "latest":
        return _DEFAULT_FAMILY
    if m.startswith("latest:"):
        family = m.split(":", 1)[1].strip()
        return family or _DEFAULT_FAMILY
    return None


def _newest_in_family(family: str) -> Optional[str]:
    """Query the live model list and return the newest id whose id/display_name matches
    the family. Returns None on any failure (caller falls back to the pin)."""
    try:
        import os

        import anthropic

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        client = anthropic.Anthropic(api_key=key)
        matches = []
        # models.list() is paginated; the SDK iterator walks all pages.
        for info in client.models.list():
            hay = f"{getattr(info, 'id', '')} {getattr(info, 'display_name', '')}".lower()
            if family in hay:
                matches.append(info)
        if not matches:
            return None
        # Newest first by created_at; fall back to id sort if timestamps are missing.
        matches.sort(key=lambda i: (getattr(i, "created_at", None) or "", i.id), reverse=True)
        return matches[0].id
    except Exception:  # noqa: BLE001 — discovery must never crash ingestion
        return None


def resolve_model(model: str) -> str:
    """Resolve a possibly-sentinel model name to a concrete Anthropic model id.

    Explicit ids pass through unchanged. 'latest'[:family] resolves via the live model
    list (cached per process), falling back to a pinned id if resolution fails.
    """
    family = _parse_sentinel(model)
    if family is None:
        return model  # explicit id — use as-is

    if family in _RESOLVED:
        return _RESOLVED[family]

    resolved = _newest_in_family(family) or _FALLBACK.get(family) or _FALLBACK[_DEFAULT_FAMILY]
    _RESOLVED[family] = resolved
    return resolved


def clear_cache() -> None:
    """Drop the resolution cache (mainly for tests)."""
    _RESOLVED.clear()
