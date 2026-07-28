"""Failure-safe parsing for bounded environment configuration."""

from __future__ import annotations

import math
import os
from typing import Optional


def bounded_int_env(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
    write_back: bool = False,
) -> int:
    """Return a bounded integer, falling back on malformed configuration."""

    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        value = int(default)
    value = max(int(minimum), min(value, int(maximum)))
    if write_back:
        os.environ[name] = str(value)
    return value


def bounded_float_env(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
    write_back: bool = False,
) -> float:
    """Return a finite bounded float, falling back on malformed configuration."""

    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    value = max(float(minimum), min(value, float(maximum)))
    if write_back:
        os.environ[name] = str(value)
    return value


def bounded_optional_int_env(
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> Optional[int]:
    """Return a bounded integer when configured, otherwise ``None``."""

    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(int(minimum), min(value, int(maximum)))
