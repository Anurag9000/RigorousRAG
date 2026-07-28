"""Strict final boundary over the scientific-integrity compatibility layer."""

from __future__ import annotations

import itertools
import json
import re
import sys
from typing import Any, Dict, Iterable, List

from tools import integrity_boundary as _implementation
from tools.security import normalize_owner_id

_MAX_SCIENTIFIC_JSON_CHARS = 100_000
_original_extract_figure_region = _implementation._extract_figure_region
_original_check_visual_entailment = _implementation.check_visual_entailment
_original_compare_papers = _implementation.compare_papers
_original_generate_comparison_matrix = _implementation.generate_comparison_matrix
_original_extract_protocol = _implementation.extract_protocol
_original_run_scientific_debate = _implementation.run_scientific_debate
_original_detect_conflicts = _implementation.detect_conflicts
_original_extract_limitations = _implementation.extract_limitations
_SAFE_VISUAL_ERROR_PREFIXES = (
    "The retained PDF source bytes are missing or oversized.",
    "figure_id must contain",
    "The retained PDF could not be opened safely.",
    "Encrypted PDFs are not supported.",
    "The retained PDF exceeds",
    "The figure region exceeds",
    "The figure region could not be rendered safely.",
    "The encoded figure region exceeds",
    "The figure label was not found as selectable text.",
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant '{value}' is not allowed.")


def _text(
    value: Any,
    label: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if "\x00" in rendered:
        raise ValueError(f"{label} contains invalid control characters.")
    if len(rendered) > maximum:
        raise ValueError(f"{label} may contain at most {maximum:,} characters.")
    if not rendered and not allow_empty:
        raise ValueError(f"{label} is required.")
    return rendered


def _model(value: Any) -> str:
    return _text(value, "model", maximum=200)


def _owner(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("owner_id must be a string.")
    return normalize_owner_id(value)


def _values(
    values: Iterable[Any],
    label: str,
    *,
    maximum_items: int,
    maximum_chars: int,
    minimum_items: int,
) -> List[str]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an array, not a string.")
    try:
        raw_values = list(itertools.islice(iter(values), maximum_items + 1))
    except TypeError as exc:
        raise ValueError(f"{label} must be iterable.") from exc
    if len(raw_values) > maximum_items:
        raise ValueError(f"{label} supports at most {maximum_items} items.")
    bounded: List[str] = []
    for raw in raw_values:
        value = _text(raw, f"{label} item", maximum=maximum_chars)
        if value not in bounded:
            bounded.append(value)
    if len(bounded) < minimum_items:
        raise ValueError(f"{label} requires at least {minimum_items} unique item(s).")
    return bounded


def _parse_json_object(raw: str) -> Dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("Model output must be a JSON string.")
    cleaned = raw.strip()
    if len(cleaned) > _MAX_SCIENTIFIC_JSON_CHARS:
        raise ValueError("Model JSON exceeds the structured-output size limit.")
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned, parse_constant=_reject_json_constant)
    except RecursionError as exc:
        raise ValueError("Model JSON exceeds the supported nesting depth.") from exc
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object.")
    return value


def _extract_figure_region(pdf_bytes: bytes, figure_id: str):
    bounded_figure = _text(figure_id, "figure_id", maximum=200)
    try:
        return _original_extract_figure_region(pdf_bytes, bounded_figure)
    except ValueError as exc:
        message = str(exc)
        if message.startswith(_SAFE_VISUAL_ERROR_PREFIXES):
            raise ValueError(message) from exc
        raise ValueError("Visual evidence could not be extracted safely.") from exc
    except Exception as exc:
        raise ValueError("Visual evidence could not be extracted safely.") from exc


def check_visual_entailment(
    claim_text: str,
    figure_id: str,
    doc_id: str,
    *,
    owner_id: str = "default_user",
    client: Any = None,
    model: str = "gpt-4o",
) -> str:
    return _original_check_visual_entailment(
        _text(claim_text, "claim_text", maximum=10_000),
        _text(figure_id, "figure_id", maximum=200),
        _text(doc_id, "doc_id", maximum=200),
        owner_id=_owner(owner_id),
        client=client,
        model=_model(model),
    )


def compare_papers(
    doc_ids: Iterable[Any],
    query: str,
    *,
    owner_id: str = "default_user",
    client: Any = None,
    model: str = "gpt-4o",
) -> str:
    return _original_compare_papers(
        _values(
            doc_ids,
            "doc_ids",
            maximum_items=10,
            maximum_chars=200,
            minimum_items=2,
        ),
        _text(query, "query", maximum=10_000),
        owner_id=_owner(owner_id),
        client=client,
        model=_model(model),
    )


def generate_comparison_matrix(
    doc_ids: Iterable[Any],
    metrics: Iterable[Any],
    *,
    owner_id: str = "default_user",
    client: Any = None,
    model: str = "gpt-4o",
) -> str:
    return _original_generate_comparison_matrix(
        _values(
            doc_ids,
            "doc_ids",
            maximum_items=10,
            maximum_chars=200,
            minimum_items=1,
        ),
        _values(
            metrics,
            "metrics",
            maximum_items=12,
            maximum_chars=500,
            minimum_items=1,
        ),
        owner_id=_owner(owner_id),
        client=client,
        model=_model(model),
    )


def extract_protocol(
    text: str,
    doc_id: str = "",
    *,
    client: Any = None,
    model: str = "gpt-4o",
) -> str:
    return _original_extract_protocol(
        _text(text, "text", maximum=30_000, allow_empty=True),
        _text(doc_id, "doc_id", maximum=200, allow_empty=True),
        client=client,
        model=_model(model),
    )


def run_scientific_debate(
    claim: str,
    context: str,
    *,
    client: Any = None,
    model: str = "gpt-4o",
) -> str:
    return _original_run_scientific_debate(
        _text(claim, "claim", maximum=10_000),
        _text(context, "context", maximum=30_000, allow_empty=True),
        client=client,
        model=_model(model),
    )


def detect_conflicts(
    topic: str,
    context: str,
    *,
    client: Any = None,
    model: str = "gpt-4o",
) -> str:
    return _original_detect_conflicts(
        _text(topic, "topic", maximum=5_000),
        _text(context, "context", maximum=35_000, allow_empty=True),
        client=client,
        model=_model(model),
    )


def extract_limitations(
    doc_id: str,
    text: str = "",
    *,
    owner_id: str = "default_user",
    client: Any = None,
    model: str = "gpt-4o",
) -> str:
    return _original_extract_limitations(
        _text(doc_id, "doc_id", maximum=200),
        _text(text, "text", maximum=35_000, allow_empty=True),
        owner_id=_owner(owner_id),
        client=client,
        model=_model(model),
    )


_implementation._parse_json_object = _parse_json_object
_implementation._extract_figure_region = _extract_figure_region
_implementation.check_visual_entailment = check_visual_entailment
_implementation.compare_papers = compare_papers
_implementation.generate_comparison_matrix = generate_comparison_matrix
_implementation.extract_protocol = extract_protocol
_implementation.run_scientific_debate = run_scientific_debate
_implementation.detect_conflicts = detect_conflicts
_implementation.extract_limitations = extract_limitations
_implementation.__doc__ = __doc__
sys.modules[__name__] = _implementation
