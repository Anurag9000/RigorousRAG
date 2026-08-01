"""Path-keyed runtime construction for optional multi-hop diagnostics."""

from __future__ import annotations

import os
import threading

from tools.multihop_trace_store import MultiHopTraceStore

_LOCK = threading.RLock()
_STORES: dict[str, MultiHopTraceStore] = {}


def _configured_path() -> str | None:
    raw = os.getenv("MULTIHOP_TRACE_DB_PATH", "")
    if not raw:
        return None
    if raw != raw.strip():
        raise ValueError(
            "MULTIHOP_TRACE_DB_PATH may not contain surrounding whitespace."
        )
    return raw


def get_multihop_trace_store(
    path: str | os.PathLike[str] | None = None,
) -> MultiHopTraceStore | None:
    selected = path if path is not None else _configured_path()
    if selected is None:
        return None
    rendered = os.fspath(selected)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Multi-hop trace path is invalid.")
    absolute = os.path.abspath(rendered)
    with _LOCK:
        existing = _STORES.get(absolute)
        if existing is not None:
            return existing
        store = MultiHopTraceStore(absolute)
        _STORES[absolute] = store
        return store


def clear_multihop_trace_store_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = [
    "clear_multihop_trace_store_cache",
    "get_multihop_trace_store",
]
