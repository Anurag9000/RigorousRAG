"""Canonical collision-resistant identities for dynamic-RAG episode steps."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _identifier(value: Any, label: str, maximum: int = 2000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def dynamic_step_pair(episode_id: Any, step_id: Any) -> tuple[str, str]:
    """Return the exact validated pair for in-memory dictionaries/sets."""
    return _identifier(episode_id, "episode_id"), _identifier(step_id, "step_id")


def dynamic_step_identity_sha256(episode_id: Any, step_id: Any) -> str:
    """Hash an injective canonical JSON encoding of the exact episode/step pair."""
    episode, step = dynamic_step_pair(episode_id, step_id)
    payload = json.dumps([episode, step], ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dynamic_step_identity(episode_id: Any, step_id: Any) -> str:
    return "dynamic-step:" + dynamic_step_identity_sha256(episode_id, step_id)


def dynamic_hidden_cache_key(episode_id: Any, step_id: Any) -> str:
    return "dynamic-hidden:" + dynamic_step_identity_sha256(episode_id, step_id)


__all__ = ["dynamic_hidden_cache_key", "dynamic_step_identity", "dynamic_step_identity_sha256", "dynamic_step_pair"]
