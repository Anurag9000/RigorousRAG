"""Failure-safe parsing for bounded environment configuration."""

from __future__ import annotations

import math
import operator
import os
import re
from typing import Any, Optional

_MAX_ENV_NAME_CHARS = 200
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,199}$")


def _environment_name(value: Any) -> str:
    if not isinstance(value, str) or not _ENVIRONMENT_NAME_RE.fullmatch(value):
        raise ValueError(
            "Environment variable names must contain 1-200 ASCII letters, digits, or "
            "underscores and may not begin with a digit."
        )
    return value


def _integer_parameter(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    return int(parsed)


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
