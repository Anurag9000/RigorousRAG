"""Strict JSON boundary over the scientific-integrity compatibility layer."""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict

from tools import integrity_boundary as _implementation

_MAX_SCIENTIFIC_JSON_CHARS = 100_000


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant '{value}' is not allowed.")


def _parse_json_object(raw: str) -> Dict[str, Any]:
    cleaned = str(raw or "").strip()
    if len(cleaned) > _MAX_SCIENTIFIC_JSON_CHARS:
        raise ValueError("Model JSON exceeds the structured-output size limit.")
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    value = json.loads(cleaned, parse_constant=_reject_json_constant)
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object.")
    return value


_implementation._parse_json_object = _parse_json_object
_implementation.__doc__ = __doc__
sys.modules[__name__] = _implementation
