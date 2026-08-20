"""Production admission limits for canonical advanced-RAG artifacts.

Historical low-level authorities may preserve wider bounds for receipt compatibility. Installed
production operators use this module so creation, bundle admission and recipe admission agree on
one bounded surface without rewriting old receipt schemas.
"""
from __future__ import annotations

from typing import Any

PRODUCTION_MAX_CANONICAL_SPLITS = 100


def assert_production_split_count(value: Any, *, label: str = "canonical split count") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if not 1 <= value <= PRODUCTION_MAX_CANONICAL_SPLITS:
        raise ValueError(
            f"{label} must lie in [1,{PRODUCTION_MAX_CANONICAL_SPLITS}] for production admission"
        )
    return value


def assert_production_split_sequence(values: Any, *, label: str = "canonical splits") -> int:
    try:
        count = len(values)
    except Exception as exc:
        raise ValueError(f"{label} must be a sized sequence") from exc
    return assert_production_split_count(count, label=f"{label} count")


__all__ = [
    "PRODUCTION_MAX_CANONICAL_SPLITS",
    "assert_production_split_count",
    "assert_production_split_sequence",
]
