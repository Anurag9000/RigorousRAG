"""Validation, path identity, JSON and tokenization helpers for the sparse index."""

from __future__ import annotations

import itertools
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

try:
    from tools.security import normalize_owner_id as _normalize_owner_id
except ImportError:  # standalone focused-test fallback
    _OWNER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

    def _normalize_owner_id(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Owner identifiers must be strings.")
        cleaned = value.strip()
        if not _OWNER_RE.fullmatch(cleaned):
            raise ValueError("Owner identifier is invalid.")
        return cleaned

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,499}$")
_CUSTOM_FIELD_RE = re.compile(r"^custom:[a-z0-9][a-z0-9_.-]{0,63}$")
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_SCHEMA_VERSION = 1
_ALLOWED_FIELDS = frozenset(
    {"title", "abstract", "heading", "body", "caption", "table", "reference"}
)
DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "title": 3.0,
    "abstract": 2.0,
    "heading": 1.8,
    "body": 1.0,
    "caption": 1.4,
    "table": 1.5,
    "reference": 0.4,
}
_MAX_FIELDS = 10_000
_MAX_FIELD_CHARS = 5_000_000
_MAX_DOCUMENT_CHARS = 50_000_000
_MAX_TOKENS_PER_FIELD = 1_000_000
_MAX_DOCUMENT_TOKENS = 5_000_000
_MAX_UNIQUE_TERMS_PER_FIELD = 500_000
_MAX_QUERY_CHARS = 20_000
_MAX_QUERY_TERMS = 256
_MAX_RESULTS = 1_000
_MAX_METADATA_BYTES = 100_000
_MAX_MATCHES_PER_HIT = 100


def _is_redirecting(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        int(getattr(metadata, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT
    )


def _reject_redirecting_components(path: Path) -> None:
    for component in (path, *path.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("Sparse index path could not be validated.") from exc
        if _is_redirecting(metadata):
            raise ValueError("Sparse index paths may not contain links or reparse points.")


def _identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if _is_redirecting(metadata):
        raise RuntimeError("Sparse index path became redirecting.")
    return int(metadata.st_dev), int(metadata.st_ino)


def _identifier(value: Any, label: str, *, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if (
        not text
        or len(text) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
        or not _IDENTIFIER_RE.fullmatch(text)
    ):
        raise ValueError(f"{label} is invalid.")
    return text


def _field_type(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("field_type must be a string.")
    text = value.strip().lower()
    if text not in _ALLOWED_FIELDS and not _CUSTOM_FIELD_RE.fullmatch(text):
        raise ValueError(f"Unsupported sparse field type: {value!r}.")
    return text


def _exact_int(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _optional_int(value: Any, label: str, *, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    return _exact_int(value, label, minimum=minimum, maximum=maximum)


def _finite(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _json_text(value: Mapping[str, Any] | None, label: str) -> str:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    copied: dict[str, Any] = {}
    try:
        items = itertools.islice(value.items(), 1_002)
        for index, (key, item) in enumerate(items):
            if index >= 1_001:
                raise ValueError(f"{label} contains too many fields.")
            if not isinstance(key, str) or not key or len(key) > 200 or any(
                ord(ch) < 32 or ord(ch) == 127 for ch in key
            ):
                raise ValueError(f"{label} contains an invalid key.")
            if key in copied:
                raise ValueError(f"{label} contains a duplicate key.")
            copied[key] = item
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{label} must be safely iterable.") from exc
    try:
        encoded = json.dumps(
            copied,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} contains unsupported values.") from exc
    if len(encoded.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValueError(f"{label} exceeds the byte limit.")
    return encoded


def _strict_json(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            value,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"Non-standard JSON constant: {constant}")
            ),
        )
    except (json.JSONDecodeError, ValueError, TypeError, RecursionError) as exc:
        raise RuntimeError(f"Stored {label} is corrupt.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Stored {label} is not an object.")
    return parsed


def tokenize(value: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise ValueError("Sparse text must be a string.")
    if len(value) > _MAX_FIELD_CHARS:
        raise ValueError("Sparse field text exceeds the character limit.")
    if any(ord(character) < 32 and character not in "\t\r\n" for character in value):
        raise ValueError("Sparse text contains invalid control characters.")
    tokens = tuple(
        match.group(0).lower()
        for match in itertools.islice(_TOKEN_RE.finditer(value), _MAX_TOKENS_PER_FIELD + 1)
    )
    if len(tokens) > _MAX_TOKENS_PER_FIELD:
        raise ValueError("Sparse field exceeds the token limit.")
    return tokens


