"""Bounded privacy-safe serialization for public API payloads."""

from __future__ import annotations

import math
from typing import Any

_PRIVATE_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "file_path",
    "local_path",
    "password",
    "secret",
    "source_path",
    "storage_path",
    "token",
}
_MAX_DEPTH = 5
_MAX_ITEMS = 1_000
_MAX_TEXT_CHARS = 100_000


def _private_key(value: str) -> bool:
    lowered = value.casefold()
    return (
        lowered in _PRIVATE_KEYS
        or lowered.endswith("_path")
        or any(
            marker in lowered
            for marker in (
                "authorization",
                "cookie",
                "password",
                "secret",
                "token",
            )
        )
    )


def _key(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("payload keys must be strings.")
    rendered = value.strip()
    if (
        not rendered
        or len(rendered) > 200
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("payload key is invalid.")
    return rendered


def sanitize_public_payload(value: Any, *, _depth: int = 0) -> Any:
    """Return JSON-compatible public data without invoking arbitrary conversion hooks."""

    if _depth > _MAX_DEPTH:
        raise ValueError("payload nesting exceeds the limit.")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("payload contains a non-finite number.")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_TEXT_CHARS or any(
            (ord(character) < 32 and character not in "\t\r\n")
            or ord(character) == 127
            for character in value
        ):
            raise ValueError("payload contains invalid or oversized text.")
        return value
    if type(value) is dict:
        if len(value) > _MAX_ITEMS:
            raise ValueError("payload mapping exceeds the item limit.")
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _key(raw_key)
            if _private_key(key):
                continue
            result[key] = sanitize_public_payload(raw_value, _depth=_depth + 1)
        return result
    if type(value) in {list, tuple}:
        if len(value) > _MAX_ITEMS:
            raise ValueError("payload collection exceeds the item limit.")
        return [
            sanitize_public_payload(item, _depth=_depth + 1)
            for item in value
        ]
    raise ValueError("payload contains an unsupported value.")


def public_model_payload(value: Any) -> dict[str, Any] | None:
    """Safely serialize a model-like value or exact dictionary."""

    candidate: Any = None
    if type(value) is dict:
        candidate = value
    else:
        try:
            model_dump = getattr(value, "model_dump", None)
        except Exception:
            model_dump = None
        if callable(model_dump):
            try:
                candidate = model_dump(mode="json", exclude_none=True)
            except Exception:
                candidate = None
    if type(candidate) is not dict:
        return None
    try:
        rendered = sanitize_public_payload(candidate)
    except ValueError:
        return None
    return rendered if isinstance(rendered, dict) else None


__all__ = ["public_model_payload", "sanitize_public_payload"]
