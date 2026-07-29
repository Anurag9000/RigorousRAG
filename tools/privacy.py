"""Best-effort masking for public document and telemetry metadata.

This module remains dependency-light so models, persistence layers, and telemetry can
apply one bounded, no-throw privacy boundary without import cycles.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import Any, Dict, Iterator, MutableSet, Tuple

_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s().-]*)?(?:\d[\s().-]*){7,14}\d(?!\w)")
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)
_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.' -]{2,80}\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|"
    r"Way|Court|Ct|Place|Pl|Highway|Hwy)\b\.?",
    flags=re.IGNORECASE,
)
_URI_CREDENTIAL_RE = re.compile(
    r"(?P<prefix>[A-Za-z][A-Za-z0-9+.-]*://)"
    r"[^/@\s:]+(?::[^/@\s]*)?@",
    flags=re.IGNORECASE,
)
_SECRET_QUERY_RE = re.compile(
    r"(?P<prefix>[?&](?:api[_-]?key|access[_-]?token|token|password|passwd|secret)=)"
    r"[^&#\s]+",
    flags=re.IGNORECASE,
)
_FILE_URI_RE = re.compile(r"\bfile://(?:localhost)?/[^\s'\"<>]+", flags=re.IGNORECASE)
_WINDOWS_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:\\(?:[^\\\s:*?\"<>|]+\\)*"
    r"[^\\\s:*?\"<>|]*"
)
_POSIX_PATH_RE = re.compile(
    r"(?<![:/A-Za-z0-9_])/(?:[^/\s'\"<>]+/)+[^/\s'\"<>,;:)}\]]*"
)
_HOME_PATH_RE = re.compile(r"(?<!\w)~/(?:[^/\s'\"<>]+/)*[^/\s'\"<>]*")
_MAX_METADATA_KEY_CHARS = 500
_MAX_METADATA_STRING_CHARS = 100_000
_MAX_METADATA_ITEMS = 1000
_MAX_METADATA_DEPTH = 8
_MAX_FALLBACK_CHARS = 4000
_MAX_INTEGER_BITS = 4096
_TRUNCATED_DEPTH = "[TRUNCATED_DEPTH]"
_CIRCULAR_REFERENCE = "[CIRCULAR_REFERENCE]"
_UNREADABLE_CONTAINER = "[UNREADABLE_CONTAINER]"
_INTEGER_OUT_OF_RANGE = "[INTEGER_OUT_OF_RANGE]"


def _type_name(value: Any) -> str:
    try:
        return type(value).__name__[:100]
    except Exception:
        return "OBJECT"


def _stringify(value: Any, *, fallback_prefix: str = "UNPRINTABLE") -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return f"[{fallback_prefix}_{_type_name(value)}]"


def mask_metadata_text(value: Any) -> str:
    """Mask common PII, credentials, and paths in one bounded public string."""

    text = _stringify(value)[:_MAX_METADATA_STRING_CHARS]
    text = _URI_CREDENTIAL_RE.sub(r"\g<prefix>[REDACTED_CREDENTIALS]@", text)
    text = _SECRET_QUERY_RE.sub(r"\g<prefix>[REDACTED_SECRET]", text)
    text = _FILE_URI_RE.sub("[REDACTED_PATH]", text)
    text = _WINDOWS_PATH_RE.sub("[REDACTED_PATH]", text)
    text = _HOME_PATH_RE.sub("[REDACTED_PATH]", text)
    text = _POSIX_PATH_RE.sub("[REDACTED_PATH]", text)
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _ADDRESS_RE.sub("[REDACTED_ADDRESS]", text)
    text = _IPV4_RE.sub("[REDACTED_IP]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def _unique_sanitized_key(raw_key: Any, existing: Dict[str, Any]) -> str:
    base = mask_metadata_text(raw_key)[:_MAX_METADATA_KEY_CHARS]
    if not base:
        base = "[REDACTED_KEY]"
    if base not in existing:
        return base
    for suffix_index in range(2, _MAX_METADATA_ITEMS + 2):
        suffix = f"#{suffix_index}"
        candidate = f"{base[:_MAX_METADATA_KEY_CHARS - len(suffix)]}{suffix}"
        if candidate not in existing:
            return candidate
    return "[TRUNCATED_KEY_COLLISIONS]"


def _fallback_text(value: Any) -> str:
    return mask_metadata_text(
        _stringify(value, fallback_prefix="UNPRINTABLE")
    )[:_MAX_FALLBACK_CHARS]


def _bounded_mapping_items(value: dict[Any, Any]) -> Tuple[list[tuple[Any, Any]], bool, bool]:
    try:
        iterator: Iterator[tuple[Any, Any]] = iter(value.items())
        items = list(itertools.islice(iterator, _MAX_METADATA_ITEMS + 1))
    except Exception:
        return [], False, False
    truncated = len(items) > _MAX_METADATA_ITEMS
    return items[:_MAX_METADATA_ITEMS], truncated, True


def _bounded_sequence_items(value: list[Any] | tuple[Any, ...]) -> Tuple[list[Any], bool, bool]:
    try:
        items = list(itertools.islice(iter(value), _MAX_METADATA_ITEMS + 1))
    except Exception:
        return [], False, False
    truncated = len(items) > _MAX_METADATA_ITEMS
    return items[:_MAX_METADATA_ITEMS], truncated, True


def sanitize_metadata(
    value: Any,
    *,
    _depth: int = 0,
    _active: MutableSet[int] | None = None,
) -> Any:
    """Recursively mask and bound arbitrary metadata into JSON-safe values."""

    if _depth > _MAX_METADATA_DEPTH:
        return _TRUNCATED_DEPTH
    if isinstance(value, str):
        return mask_metadata_text(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        try:
            return value if value.bit_length() <= _MAX_INTEGER_BITS else _INTEGER_OUT_OF_RANGE
        except Exception:
            return _INTEGER_OUT_OF_RANGE
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    active = _active if _active is not None else set()
    if isinstance(value, dict):
        identity = id(value)
        if identity in active:
            return _CIRCULAR_REFERENCE
        active.add(identity)
        try:
            items, truncated, readable = _bounded_mapping_items(value)
            if not readable:
                return _UNREADABLE_CONTAINER
            sanitized: Dict[str, Any] = {}
            for key, item in items:
                safe_key = _unique_sanitized_key(key, sanitized)
                sanitized[safe_key] = sanitize_metadata(
                    item,
                    _depth=_depth + 1,
                    _active=active,
                )
            if truncated:
                marker = _unique_sanitized_key("__truncated_items__", sanitized)
                sanitized[marker] = True
            return sanitized
        finally:
            active.discard(identity)

    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            return _CIRCULAR_REFERENCE
        active.add(identity)
        try:
            items, truncated, readable = _bounded_sequence_items(value)
            if not readable:
                return [_UNREADABLE_CONTAINER]
            sanitized_items = [
                sanitize_metadata(
                    item,
                    _depth=_depth + 1,
                    _active=active,
                )
                for item in items
            ]
            if truncated:
                sanitized_items.append({"__truncated_items__": True})
            return sanitized_items
        finally:
            active.discard(identity)

    return _fallback_text(value)


def sanitize_metadata_dict(value: Any) -> Dict[str, Any]:
    sanitized = sanitize_metadata(value)
    return sanitized if isinstance(sanitized, dict) else {}
