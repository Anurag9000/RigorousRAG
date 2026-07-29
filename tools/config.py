"""Failure-safe parsing for bounded environment configuration."""

from __future__ import annotations

import math
import os
from typing import Any, Optional

_MAX_ENV_NAME_CHARS = 200


def _environment_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Environment variable names must be strings.")
    if (
        not value
        or len(value) > _MAX_ENV_NAME_CHARS
        or "=" in value
        or "\x00" in value
    ):
        raise ValueError(
            "Environment variable names must contain 1-200 valid characters."
        )
    return value


def _integer_parameter(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer.")
    return parsed


def _float_parameter(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    return parsed


def _write_back_flag(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("write_back must be a boolean.")
    return value


def bounded_int_env(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
    write_back: bool = False,
) -> int:
    """Return a bounded integer, falling back on malformed configuration."""

    selected_name = _environment_name(name)
    lower = _integer_parameter(minimum, "minimum")
    upper = _integer_parameter(maximum, "maximum")
    if lower > upper:
        raise ValueError("minimum may not exceed maximum.")
    fallback = _integer_parameter(default, "default")
    write = _write_back_flag(write_back)
    try:
        value = int(os.getenv(selected_name, str(fallback)))
    except (TypeError, ValueError, OverflowError):
        value = fallback
    value = max(lower, min(value, upper))
    if write:
        os.environ[selected_name] = str(value)
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

    selected_name = _environment_name(name)
    lower = _float_parameter(minimum, "minimum")
    upper = _float_parameter(maximum, "maximum")
    if lower > upper:
        raise ValueError("minimum may not exceed maximum.")
    fallback = _float_parameter(default, "default")
    write = _write_back_flag(write_back)
    try:
        value = float(os.getenv(selected_name, str(fallback)))
    except (TypeError, ValueError, OverflowError):
        value = fallback
    if not math.isfinite(value):
        value = fallback
    value = max(lower, min(value, upper))
    if write:
        os.environ[selected_name] = str(value)
    return value


def bounded_optional_int_env(
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> Optional[int]:
    """Return a bounded integer when configured, otherwise ``None``."""

    selected_name = _environment_name(name)
    lower = _integer_parameter(minimum, "minimum")
    upper = _integer_parameter(maximum, "maximum")
    if lower > upper:
        raise ValueError("minimum may not exceed maximum.")
    raw = os.getenv(selected_name)
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(lower, min(value, upper))
