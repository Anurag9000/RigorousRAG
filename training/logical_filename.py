"""Deterministic path-safe filenames for governed logical identifiers.

Governance identifiers are semantic strings, not filesystem components. They may legitimately
contain punctuation that is unsafe or ambiguous in a path. Writers therefore keep the exact
logical name inside manifests/receipts and derive the on-disk filename solely from a canonical
SHA-256 of that name plus a fixed caller-owned extension.
"""
from __future__ import annotations

import hashlib
from typing import Any


def _logical_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("logical filename name must be a string")
    selected = value.strip()
    if not selected or len(selected) > 10_000 or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError("logical filename name is invalid")
    return selected


def _extension(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("logical filename extension must be a string")
    selected = value.strip()
    if not selected.startswith(".") or len(selected) > 80:
        raise ValueError("logical filename extension must be a bounded dot-prefixed suffix")
    if any(ch not in ".abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in selected):
        raise ValueError("logical filename extension contains unsupported characters")
    return selected


def logical_filename(name: Any, extension: Any) -> str:
    """Return ``<sha256(logical-name)><fixed-extension>`` as one safe path component."""
    selected = _logical_name(name)
    suffix = _extension(extension)
    digest = hashlib.sha256(selected.encode("utf-8")).hexdigest()
    return digest + suffix


__all__ = ["logical_filename"]
