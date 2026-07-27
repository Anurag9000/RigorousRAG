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


def mask_metadata_text(value: str) -> str:
    text = value or ""
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _ADDRESS_RE.sub("[REDACTED_ADDRESS]", text)
    text = _IPV4_RE.sub("[REDACTED_IP]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def sanitize_metadata(value: Any) -> Any:
    """Recursively mask strings while preserving JSON-compatible structure."""

    if isinstance(value, str):
        return mask_metadata_text(value)
    if isinstance(value, dict):
        return {str(key): sanitize_metadata(item) for key, item in value.items()}
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
