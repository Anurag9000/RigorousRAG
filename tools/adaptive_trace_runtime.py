"""Path-keyed runtime construction for optional adaptive retrieval diagnostics."""

from __future__ import annotations

import os
import threading

from tools.adaptive_trace_store import AdaptiveTraceStore

_LOCK = threading.RLock()
_STORES: dict[str, AdaptiveTraceStore] = {}


def _configured_path() -> str | None:
    raw = os.getenv("ADAPTIVE_TRACE_DB_PATH", "")
    if not raw:
        return None
    if raw != raw.strip():
        raise ValueError("ADAPTIVE_TRACE_DB_PATH may not contain surrounding whitespace.")
    return raw


def get_adaptive_trace_store(
    path: str | os.PathLike[str] | None = None,
) -> AdaptiveTraceStore | None:
    """Return a path-keyed singleton, or ``None`` when tracing is not configured."""

    selected = path if path is not None else _configured_path()
    if selected is None:
        return None
    rendered = os.fspath(selected)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Adaptive trace path is invalid.")
    absolute = os.path.abspath(rendered)
    with _LOCK:
        existing = _STORES.get(absolute)
        if existing is not None:
            return existing
        store = AdaptiveTraceStore(absolute)
        _STORES[absolute] = store
        return store


def clear_adaptive_trace_store_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = ["clear_adaptive_trace_store_cache", "get_adaptive_trace_store"]
