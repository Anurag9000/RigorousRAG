"""Best-effort masking for user-visible document metadata.

The full document-text masker lives in the ingestion pipeline. This module is
kept dependency-light so Pydantic models and persistence layers can apply the
same privacy boundary without creating import cycles.
"""

from __future__ import annotations

import re
from typing import Any, Dict

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


def mask_metadata_text(value: str) -> str:
    """Mask common PII, credentials, and local filesystem paths in public strings."""

    text = value or ""
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
    base = mask_metadata_text(str(raw_key))[:_MAX_METADATA_KEY_CHARS]
    if not base:
        base = "[REDACTED_KEY]"
    if base not in existing:
        return base
    suffix_index = 2
    while True:
        suffix = f"#{suffix_index}"
        candidate = f"{base[:_MAX_METADATA_KEY_CHARS - len(suffix)]}{suffix}"
        if candidate not in existing:
            return candidate
        suffix_index += 1


def sanitize_metadata(value: Any) -> Any:
    """Recursively mask strings and mapping keys while preserving JSON structure."""

    if isinstance(value, str):
        return mask_metadata_text(value)
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for key, item in value.items():
            safe_key = _unique_sanitized_key(key, sanitized)
            sanitized[safe_key] = sanitize_metadata(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_metadata(item) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return mask_metadata_text(str(value))


def sanitize_metadata_dict(value: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = sanitize_metadata(value)
    return sanitized if isinstance(sanitized, dict) else {}
