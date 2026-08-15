"""Complementary adversarial diagnostics for OCR degradation and contradictory evidence."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

_MAX_TEXT = 4000
_MAX_RESULTS = 100_000
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _text(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if len(value) > _MAX_TEXT:
        raise ValueError(f"{label} exceeds {_MAX_TEXT} characters")
    return value


def _ids(values: Iterable[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an iterable of IDs")
    result = []
    for item in values:
        if len(result) >= _MAX_RESULTS:
            raise ValueError(f"{label} exceeds result limit")
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label} contains an invalid ID")
        result.append(item.strip())
    return tuple(result)


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class OcrStressReport:
    character_error_rate: float
    token_recall: float
    token_precision: float
    exact_match: bool


def ocr_stress_report(*, reference: str, observed: str) -> OcrStressReport:
    reference_text = _text(reference, "reference")
    observed_text = _text(observed, "observed")
    distance = _edit_distance(reference_text, observed_text)
    denominator = max(len(reference_text), 1)
    reference_tokens = _TOKEN.findall(reference_text.casefold())
    observed_tokens = _TOKEN.findall(observed_text.casefold())
    reference_counts: dict[str, int] = {}
    observed_counts: dict[str, int] = {}
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    for token in observed_tokens:
        observed_counts[token] = observed_counts.get(token, 0) + 1
    overlap = sum(min(count, observed_counts.get(token, 0)) for token, count in reference_counts.items())
    recall = overlap / len(reference_tokens) if reference_tokens else (1.0 if not observed_tokens else 0.0)
    precision = overlap / len(observed_tokens) if observed_tokens else (1.0 if not reference_tokens else 0.0)
    return OcrStressReport(
        character_error_rate=distance / denominator,
        token_recall=recall,
        token_precision=precision,
        exact_match=reference_text == observed_text,
    )


@dataclass(frozen=True)
class ContradictionExposureReport:
    k: int
    support_recall_at_k: float
    contradiction_exposure_rate: float
    first_contradiction_rank: int | None
    support_before_contradiction: bool


def contradiction_exposure_report(
    *,
    ranking: Sequence[str],
    support_ids: Iterable[str],
    contradiction_ids: Iterable[str],
    k: int = 10,
) -> ContradictionExposureReport:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= _MAX_RESULTS:
        raise ValueError("k must be a positive bounded integer")
    ranked = _ids(ranking[:k], "ranking")
    support = set(_ids(support_ids, "support_ids"))
    contradictions = set(_ids(contradiction_ids, "contradiction_ids"))
    if support & contradictions:
        raise ValueError("support and contradiction IDs must not overlap")
    support_hits = [index + 1 for index, item in enumerate(ranked) if item in support]
    contradiction_hits = [index + 1 for index, item in enumerate(ranked) if item in contradictions]
    support_recall = len(set(ranked) & support) / len(support) if support else 0.0
    contradiction_rate = len(set(ranked) & contradictions) / len(ranked) if ranked else 0.0
    first_contradiction = min(contradiction_hits) if contradiction_hits else None
    first_support = min(support_hits) if support_hits else None
    return ContradictionExposureReport(
        k=k,
        support_recall_at_k=support_recall,
        contradiction_exposure_rate=contradiction_rate,
        first_contradiction_rank=first_contradiction,
        support_before_contradiction=(
            first_support is not None
            and (first_contradiction is None or first_support < first_contradiction)
        ),
    )
