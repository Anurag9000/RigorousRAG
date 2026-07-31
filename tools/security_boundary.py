"""Strict configuration and direct-call boundary for :mod:`tools.security`.

The underlying transport implementation remains in ``tools.security``.  This
module installs fail-closed validators before callers import public security
helpers, following the repository's existing compatibility-boundary pattern.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import operator
import os
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from tools import security as _implementation


_original_safe_upload_suffix = _implementation.safe_upload_suffix
_original_validate_public_url = _implementation.validate_public_url


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key.")
        result[key] = value
    return result


def _api_key(value: Any) -> str:
    if not isinstance(value, str):
        raise RuntimeError("Every configured API key must be a string.")
    if (
        not value
        or value != value.strip()
        or len(value) > _implementation._MAX_API_KEY_CHARS
        or _contains_ascii_control(value)
    ):
        raise RuntimeError(
            "Every configured API key must already be canonical and contain "
            f"1-{_implementation._MAX_API_KEY_CHARS} valid characters."
        )
    return value


def parse_api_key_owners() -> Dict[str, str]:
    """Load one duplicate-free, canonical API-key-to-owner mapping."""

    raw_mapping = os.getenv("API_KEY_OWNERS_JSON", "")
    if (
        len(raw_mapping.encode("utf-8", errors="ignore"))
        > _implementation._MAX_API_KEY_CONFIG_BYTES
    ):
        raise RuntimeError("API_KEY_OWNERS_JSON exceeds the configuration byte limit.")
    if raw_mapping.strip():
        if raw_mapping != raw_mapping.strip():
            raise RuntimeError("API_KEY_OWNERS_JSON must already be canonical JSON text.")
        try:
            parsed = json.loads(
                raw_mapping,
                object_pairs_hook=_strict_json_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"Non-standard JSON constant {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise RuntimeError("API_KEY_OWNERS_JSON must contain valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("API_KEY_OWNERS_JSON must be a JSON object.")
        if len(parsed) > _implementation._MAX_API_KEYS:
            raise RuntimeError(
                "API_KEY_OWNERS_JSON may contain at most "
                f"{_implementation._MAX_API_KEYS} keys."
            )
        result: Dict[str, str] = {}
        for api_key, owner_id in parsed.items():
            key = _api_key(api_key)
            if not isinstance(owner_id, str):
                raise RuntimeError("Every configured owner ID must be a string.")
            if owner_id != owner_id.strip():
                raise RuntimeError("Every configured owner ID must already be canonical.")
            owner = _implementation.normalize_owner_id(owner_id)
            if owner != owner_id:
                raise RuntimeError("Every configured owner ID must already be canonical.")
            result[key] = owner
        return result

    result: Dict[str, str] = {}
    raw_legacy = os.getenv("ALLOWED_API_KEYS", "")
    if (
        len(raw_legacy.encode("utf-8", errors="ignore"))
        > _implementation._MAX_API_KEY_CONFIG_BYTES
    ):
        raise RuntimeError("ALLOWED_API_KEYS exceeds the configuration byte limit.")
    for raw_key in raw_legacy.split(","):
        if raw_key == "":
            continue
        if raw_key != raw_key.strip():
            raise RuntimeError("Legacy API keys must already be canonical.")
        key = _api_key(raw_key)
        if key in result:
            raise RuntimeError("Legacy API keys must be unique.")
        if len(result) >= _implementation._MAX_API_KEYS:
            raise RuntimeError(
                "ALLOWED_API_KEYS may contain at most "
                f"{_implementation._MAX_API_KEYS} keys."
            )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        result[key] = f"api-{digest}"
    return result


def safe_upload_suffix(filename: Optional[str]) -> str:
    if (
        not isinstance(filename, str)
        or len(filename) > 500
        or _contains_ascii_control(filename)
    ):
        raise _implementation.SecurityError(
            "Upload filenames must contain at most 500 valid characters."
        )
    return _original_safe_upload_suffix(filename)


def _canonical_hostname(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 253
        or any(
            character.isspace()
            or ord(character) < 33
            or ord(character) == 127
            for character in value
        )
    ):
        return ""
    return _implementation._legacy_canonical_hostname(value)


def validate_public_url(url: str) -> str:
    if not isinstance(url, str):
        raise _implementation.SecurityError("URLs must be strings.")
    if (
        not url
        or url != url.strip()
        or len(url) > _implementation._MAX_URL_CHARS
    ):
        raise _implementation.SecurityError(
            "URLs must contain canonical bounded text without surrounding whitespace."
        )
    if _contains_ascii_control(url) or "\\" in url:
        raise _implementation.SecurityError(
            "URLs may not contain control characters or backslashes."
        )
    return _original_validate_public_url(url)


def _allowed_domain(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 253
        or _contains_ascii_control(value)
        or "\\" in value
    ):
        return ""
    try:
        parsed = urlparse(value if "://" in value else f"https://{value}")
        port = parsed.port
    except (ValueError, UnicodeError):
        return ""
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        return ""
    return _canonical_hostname(parsed.hostname or "")


def hostname_matches(hostname: str, allowed_domains: Iterable[str]) -> bool:
    host = _canonical_hostname(hostname)
    if not host or isinstance(allowed_domains, (str, bytes, bytearray)):
        return False
    try:
        candidates = itertools.islice(
            iter(allowed_domains),
            _implementation._MAX_ALLOWED_DOMAINS,
        )
    except Exception:
        return False
    for raw_domain in candidates:
        domain = _allowed_domain(raw_domain)
        if domain and (host == domain or host.endswith(f".{domain}")):
            return True
    return False


def _sanitize_request_headers(
    headers: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    if headers is None:
        return {}
    if not isinstance(headers, Mapping):
        raise _implementation.SecurityError(
            "Remote request headers must be a mapping."
        )
    if len(headers) > _implementation._MAX_REQUEST_HEADERS:
        raise _implementation.SecurityError(
            f"At most {_implementation._MAX_REQUEST_HEADERS} request headers are allowed."
        )
    sanitized: Dict[str, str] = {}
    for raw_name, raw_value in headers.items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, str):
            raise _implementation.SecurityError(
                "Remote request header names and values must be strings."
            )
        if raw_name != raw_name.strip() or raw_value != raw_value.strip():
            raise _implementation.SecurityError(
                "Remote request headers must already be canonical."
            )
        lowered = raw_name.lower()
        if not _implementation._HEADER_NAME_RE.fullmatch(raw_name):
            raise _implementation.SecurityError(
                "Remote request header names contain invalid characters."
            )
        if lowered in _implementation._FORBIDDEN_CALLER_HEADERS:
            raise _implementation.SecurityError(
                f"Caller-controlled header '{raw_name}' is not allowed."
            )
        if len(raw_value) > _implementation._MAX_HEADER_VALUE_CHARS:
            raise _implementation.SecurityError(
                "Remote request header values exceed the size limit."
            )
        if _contains_ascii_control(raw_value):
            raise _implementation.SecurityError(
                "Remote request headers may not contain control characters."
            )
        sanitized[raw_name] = raw_value
    return sanitized


def _bounded_response_headers(headers: Any) -> Dict[str, str]:
    if not isinstance(headers, Mapping):
        return {}
    bounded: Dict[str, str] = {}
    try:
        items = headers.items()
    except Exception:
        return {}
    try:
        candidates = itertools.islice(items, _implementation._MAX_RESPONSE_HEADERS)
        for raw_name, raw_value in candidates:
            if not isinstance(raw_name, str) or not isinstance(raw_value, str):
                continue
            name = raw_name[:200]
            value = raw_value[: _implementation._MAX_HEADER_VALUE_CHARS]
            if (
                not _implementation._HEADER_NAME_RE.fullmatch(name)
                or _contains_ascii_control(value)
                or name.lower() in _implementation._SENSITIVE_RESPONSE_HEADERS
            ):
                continue
            bounded[name] = value
    except Exception:
        return {}
    return bounded


def _positive_integer(value: Any, label: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        numeric = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not 1 <= numeric <= maximum:
        raise ValueError(f"{label} must be between 1 and {maximum}.")
    return numeric


def _positive_timeout(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("timeout must be numeric.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timeout must be numeric.") from exc
    if not math.isfinite(numeric) or not 0.1 <= numeric <= 300.0:
        raise ValueError(
            "timeout must be finite and between 0.1 and 300 seconds."
        )
    return numeric


# Preserve the implementation helper once, then install the strict boundary.
if not hasattr(_implementation, "_legacy_canonical_hostname"):
    _implementation._legacy_canonical_hostname = _implementation._canonical_hostname

_implementation._api_key = _api_key
_implementation.parse_api_key_owners = parse_api_key_owners
_implementation.safe_upload_suffix = safe_upload_suffix
_implementation._canonical_hostname = _canonical_hostname
_implementation.validate_public_url = validate_public_url
_implementation.hostname_matches = hostname_matches
_implementation._sanitize_request_headers = _sanitize_request_headers
_implementation._bounded_response_headers = _bounded_response_headers
_implementation._positive_integer = _positive_integer
_implementation._positive_timeout = _positive_timeout
