"""Best-effort masking for user-visible document metadata.

The full document-text masker lives in the ingestion pipeline. This module remains
dependency-light so models and persistence layers can apply one bounded privacy
boundary without import cycles.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, MutableSet

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
_TRUNCATED_DEPTH = "[TRUNCATED_DEPTH]"
_CIRCULAR_REFERENCE = "[CIRCULAR_REFERENCE]"


def mask_metadata_text(value: str) -> str:
    """Mask common PII, credentials, and paths in one bounded public string."""

    text = str(value or "")[:_MAX_METADATA_STRING_CHARS]
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
    try:
        raw_text = str(raw_key)
    except Exception:
        raw_text = "[UNPRINTABLE_KEY]"
    base = mask_metadata_text(raw_text)[:_MAX_METADATA_KEY_CHARS]
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
    try:
        rendered = str(value)
    except Exception:
        rendered = f"[UNPRINTABLE_{type(value).__name__}]"
    return mask_metadata_text(rendered)[:_MAX_FALLBACK_CHARS]


def sanitize_metadata(
    value: Any,
    *,
    _depth: int = 0,
    _active: MutableSet[int] | None = None,
) -> Any:
    """Recursively mask and bound JSON-like public metadata.

    Cyclic containers, excessive depth/items, non-finite numbers, and hostile custom
    ``__str__`` methods fail closed into explicit JSON-safe sentinel values.
    """

    if _depth > _MAX_METADATA_DEPTH:
        return _TRUNCATED_DEPTH
    if isinstance(value, str):
        return mask_metadata_text(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    active = _active if _active is not None else set()
    if isinstance(value, (dict, list, tuple)):
        identity = id(value)
        if identity in active:
            return _CIRCULAR_REFERENCE
        active.add(identity)
        try:
            if isinstance(value, dict):
                sanitized: Dict[str, Any] = {}
                for index, (key, item) in enumerate(value.items()):
                    if index >= _MAX_METADATA_ITEMS:
                        marker = _unique_sanitized_key("__truncated_items__", sanitized)
                        sanitized[marker] = True
                        break
                    safe_key = _unique_sanitized_key(key, sanitized)
                    sanitized[safe_key] = sanitize_metadata(
                        item,
                        _depth=_depth + 1,
                        _active=active,
                    )
                return sanitized
            items = list(value[:_MAX_METADATA_ITEMS])
            sanitized_items = [
                sanitize_metadata(item, _depth=_depth + 1, _active=active)
                for item in items
            ]
            if len(value) > _MAX_METADATA_ITEMS:
                sanitized_items.append({"__truncated_items__": True})
            return sanitized_items
        finally:
            active.discard(identity)
    return _fallback_text(value)


def sanitize_metadata_dict(value: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = sanitize_metadata(value)
    return sanitized if isinstance(sanitized, dict) else {}
