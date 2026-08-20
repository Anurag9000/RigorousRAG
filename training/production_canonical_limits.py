"""Production admission limits for canonical advanced-RAG artifacts.

Historical low-level authorities may preserve wider bounds for receipt compatibility. Installed
production operators use this module so creation, bundle admission and recipe admission agree on
one bounded surface without rewriting old receipt schemas.  Grounded creation uses a lightweight
receipt/manifest envelope preflight here; the canonical materializer then performs the full
restart verification exactly once.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from training.advanced_path_authority import safe_advanced_path

PRODUCTION_MAX_CANONICAL_SPLITS = 100
_MAX_JSON_BYTES = 64 * 1024 * 1024


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


def _strict_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.stat().st_size <= 0 or path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds byte safety bound")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def grounded_source_split_count_from_receipt(receipt_path: str | Path) -> int:
    """Read only the bounded receipt/manifest envelopes to preflight production split count.

    This helper intentionally does *not* claim to verify the dataset.  Its caller must still run
    the normal grounded import verifier/materializer, which hashes/parses every authoritative
    split.  The purpose here is solely to reject a grossly over-wide split universe before that
    expensive pass without performing the same pass twice.
    """
    receipt_file = safe_advanced_path(
        receipt_path,
        label="grounded source receipt preflight",
        must_exist=True,
        require_file=True,
    )
    receipt = _strict_json(receipt_file, "grounded source receipt preflight")
    manifest_value = receipt.get("manifest_path")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ValueError("grounded source receipt preflight lacks manifest_path")
    manifest_file = safe_advanced_path(
        manifest_value,
        label="grounded source manifest preflight",
        must_exist=True,
        require_file=True,
    )
    envelope = _strict_json(manifest_file, "grounded source manifest preflight")
    manifest = envelope.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("grounded source manifest preflight lacks manifest object")
    splits = manifest.get("splits")
    if not isinstance(splits, list):
        raise ValueError("grounded source manifest preflight splits must be an array")
    return len(splits)


__all__ = [
    "PRODUCTION_MAX_CANONICAL_SPLITS",
    "assert_production_split_count",
    "assert_production_split_sequence",
    "grounded_source_split_count_from_receipt",
]
