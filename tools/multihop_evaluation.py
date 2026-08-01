"""Dependency-free evaluation metrics for provenance-preserving multi-hop RAG."""

from __future__ import annotations

import itertools
import math
import operator
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_MAX_EXAMPLES = 100_000
_MAX_ANSWERS = 50
_MAX_SUPPORT_FACTS = 200
_MAX_EVIDENCE = 500
_MAX_TEXT_CHARS = 20_000
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = " ".join(value.split())
    if len(rendered) > _MAX_TEXT_CHARS or "\x00" in rendered:
        raise ValueError(f"{label} is invalid or too long.")
    if not rendered and not allow_empty:
        raise ValueError(f"{label} is required.")
    return rendered


def _strings(
    values: Any,
    *,
    label: str,
    maximum: int,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence of strings.")
    try:
        raw = list(values)
    except Exception as exc:
        raise ValueError(f"{label} must be safely iterable.") from exc
    if not raw or len(raw) > maximum:
        raise ValueError(f"{label} must contain 1-{maximum} values.")
    result: list[str] = []
    seen: set[str] = set()
    for value in raw:
        rendered = _text(value, label, allow_empty=allow_empty)
        key = rendered.casefold()
        if key not in seen:
            seen.add(key)
            result.append(rendered)
    return tuple(result)


def _safe_attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        try:
            return value.get(name, default)
        except Exception:
            return default
    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _metadata(value: Any) -> Mapping[str, Any]:
    metadata = _safe_attr(value, "metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def normalize_answer(value: str) -> str:
    """Apply Unicode-compatible, language-agnostic token normalization."""

    rendered = unicodedata.normalize("NFKC", _text(value, "answer", allow_empty=True))
    return " ".join(_TOKEN_RE.findall(rendered.casefold()))


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class SupportFact:
    document_id: str
    locator: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _text(self.document_id, "document_id"))
        if self.locator is not None:
            object.__setattr__(self, "locator", _text(self.locator, "locator"))


@dataclass(frozen=True)
class MultiHopEvaluationExample:
    example_id: str
    question: str
    answers: tuple[str, ...]
    support_facts: tuple[SupportFact, ...]
    required_hops: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _text(self.example_id, "example_id"))
        object.__setattr__(self, "question", _text(self.question, "question"))
        object.__setattr__(
            self,
            "answers",
            _strings(self.answers, label="answers", maximum=_MAX_ANSWERS, allow_empty=True),
        )
        if isinstance(self.support_facts, (str, bytes, bytearray)):
            raise ValueError("support_facts must be a sequence.")
        try:
            facts = tuple(self.support_facts)
        except Exception as exc:
            raise ValueError("support_facts must be safely iterable.") from exc
        if len(facts) > _MAX_SUPPORT_FACTS or any(
            not isinstance(item, SupportFact) for item in facts
        ):
            raise ValueError("support_facts are invalid or exceed the limit.")
        object.__setattr__(self, "support_facts", facts)
        object.__setattr__(
            self,
            "required_hops",
            _integer(self.required_hops, "required_hops", 1, 100),
        )


@dataclass(frozen=True)
class MultiHopEvaluationPrediction:
    answer: str
    evidence: tuple[Any, ...]
    abstained: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer", _text(self.answer, "answer", allow_empty=True))
        if not isinstance(self.abstained, bool):
            raise ValueError("abstained must be a boolean.")
        if isinstance(self.evidence, (str, bytes, bytearray)):
            raise ValueError("evidence must be a sequence.")
        try:
            rows = tuple(itertools.islice(iter(self.evidence), _MAX_EVIDENCE + 1))
        except Exception as exc:
            raise ValueError("evidence must be safely iterable.") from exc
        if len(rows) > _MAX_EVIDENCE:
            raise ValueError("evidence exceeds the limit.")
        object.__setattr__(self, "evidence", rows)


@dataclass(frozen=True)
class MultiHopExampleMetrics:
    example_id: str
    answer_exact_match: float
    answer_token_f1: float
    document_precision: float
    document_recall: float
    document_f1: float
    support_precision: float
    support_recall: float
    support_f1: float
    path_complete: bool
    hop_coverage: float
    citation_lineage_validity: float
    evidence_count: int
    abstained: bool
    answer_support_score: float


@dataclass(frozen=True)
class MultiHopAggregateMetrics:
    example_count: int
    answer_exact_match: float
    answer_token_f1: float
    document_precision: float
    document_recall: float
    document_f1: float
    support_precision: float
    support_recall: float
    support_f1: float
    path_complete_rate: float
    hop_coverage: float
    citation_lineage_validity: float
    abstention_rate: float
    answer_support_score: float


def _ratio(numerator: int, denominator: int, *, empty_value: float = 0.0) -> float:
    return numerator / denominator if denominator else empty_value


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _evidence_identity(value: Any) -> tuple[str | None, set[str], str | None, str | None]:
    document_id = _safe_attr(value, "doc_id", None)
    if not isinstance(document_id, str) or not document_id.strip():
        document_id = None
    else:
        document_id = document_id.strip()
    metadata = _metadata(value)
    locators: set[str] = set()
    page = _safe_attr(value, "page_number", None)
    if isinstance(page, int) and not isinstance(page, bool) and page > 0:
        locators.add(f"page:{page}")
    try:
        section = metadata.get("section_title")
        field = metadata.get("field_type")
    except Exception:
        section = None
        field = None
    if isinstance(section, str) and section.strip():
        locators.add(f"section:{' '.join(section.split())}")
    if isinstance(field, str) and field.strip():
        locators.add(f"field:{' '.join(field.split())}")
    source_id = _safe_attr(value, "source_id", None)
    if isinstance(source_id, str) and source_id.strip():
        source_id = source_id.strip()
        locators.add(f"source:{source_id}")
    else:
        source_id = None
    hop_id = _safe_attr(value, "hop_id", None)
    if not isinstance(hop_id, str) or not hop_id.strip():
        hop_id = None
    else:
        hop_id = hop_id.strip()
    return document_id, locators, hop_id, source_id


def _fact_matches(
    fact: SupportFact,
    document_id: str | None,
    locators: set[str],
) -> bool:
    if document_id != fact.document_id:
        return False
    if fact.locator is None:
        return True
    return fact.locator in locators


def evaluate_multihop_example(
    example: MultiHopEvaluationExample,
    prediction: MultiHopEvaluationPrediction,
) -> MultiHopExampleMetrics:
    if not isinstance(example, MultiHopEvaluationExample):
        raise ValueError("example must be a MultiHopEvaluationExample.")
    if not isinstance(prediction, MultiHopEvaluationPrediction):
        raise ValueError("prediction must be a MultiHopEvaluationPrediction.")

    normalized_prediction = normalize_answer(prediction.answer)
    exact = max(
        (float(normalized_prediction == normalize_answer(answer)) for answer in example.answers),
        default=0.0,
    )
    answer_f1 = max(
        (token_f1(prediction.answer, answer) for answer in example.answers),
        default=0.0,
    )

    identities = [_evidence_identity(item) for item in prediction.evidence]
    predicted_documents = {doc for doc, _locators, _hop, _source in identities if doc}
    gold_documents = {fact.document_id for fact in example.support_facts}
    document_overlap = len(predicted_documents & gold_documents)
    document_precision = _ratio(
        document_overlap,
        len(predicted_documents),
        empty_value=1.0 if not gold_documents else 0.0,
    )
    document_recall = _ratio(
        document_overlap,
        len(gold_documents),
        empty_value=1.0,
    )

    matched_gold = sum(
        any(
            _fact_matches(fact, document_id, locators)
            for document_id, locators, _hop, _source in identities
        )
        for fact in example.support_facts
    )
    predicted_units = {
        (document_id, tuple(sorted(locators)))
        for document_id, locators, _hop, _source in identities
        if document_id
    }
    matching_units = sum(
        any(
            _fact_matches(fact, document_id, set(locators))
            for fact in example.support_facts
        )
        for document_id, locators in predicted_units
    )
    support_precision = _ratio(
        matching_units,
        len(predicted_units),
        empty_value=1.0 if not example.support_facts else 0.0,
    )
    support_recall = _ratio(
        matched_gold,
        len(example.support_facts),
        empty_value=1.0,
    )
    hops = {hop for _doc, _locators, hop, _source in identities if hop}
    lineage_valid = sum(
        1 for _doc, _locators, hop, source in identities if hop and source
    )
    path_complete = matched_gold == len(example.support_facts)
    abstained = prediction.abstained or not normalized_prediction
    return MultiHopExampleMetrics(
        example_id=example.example_id,
        answer_exact_match=round(exact, 9),
        answer_token_f1=round(answer_f1, 9),
        document_precision=round(document_precision, 9),
        document_recall=round(document_recall, 9),
        document_f1=round(_f1(document_precision, document_recall), 9),
        support_precision=round(support_precision, 9),
        support_recall=round(support_recall, 9),
        support_f1=round(_f1(support_precision, support_recall), 9),
        path_complete=path_complete,
        hop_coverage=round(min(len(hops) / example.required_hops, 1.0), 9),
        citation_lineage_validity=round(
            _ratio(lineage_valid, len(identities), empty_value=1.0),
            9,
        ),
        evidence_count=len(identities),
        abstained=abstained,
        answer_support_score=round(answer_f1 * support_recall, 9),
    )


def aggregate_multihop_metrics(
    values: Iterable[MultiHopExampleMetrics],
) -> MultiHopAggregateMetrics:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("metrics must be an iterable.")
    try:
        rows = list(itertools.islice(iter(values), _MAX_EXAMPLES + 1))
    except Exception as exc:
        raise ValueError("metrics must be safely iterable.") from exc
    if len(rows) > _MAX_EXAMPLES:
        raise ValueError("metric example limit exceeded.")
    if any(not isinstance(row, MultiHopExampleMetrics) for row in rows):
        raise ValueError("every value must be MultiHopExampleMetrics.")
    if not rows:
        return MultiHopAggregateMetrics(
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def mean(name: str) -> float:
        values_for_name = [float(getattr(row, name)) for row in rows]
        result = sum(values_for_name) / len(values_for_name)
        return round(result if math.isfinite(result) else 0.0, 9)

    return MultiHopAggregateMetrics(
        example_count=len(rows),
        answer_exact_match=mean("answer_exact_match"),
        answer_token_f1=mean("answer_token_f1"),
        document_precision=mean("document_precision"),
        document_recall=mean("document_recall"),
        document_f1=mean("document_f1"),
        support_precision=mean("support_precision"),
        support_recall=mean("support_recall"),
        support_f1=mean("support_f1"),
        path_complete_rate=round(sum(row.path_complete for row in rows) / len(rows), 9),
        hop_coverage=mean("hop_coverage"),
        citation_lineage_validity=mean("citation_lineage_validity"),
        abstention_rate=round(sum(row.abstained for row in rows) / len(rows), 9),
        answer_support_score=mean("answer_support_score"),
    )


__all__ = [
    "MultiHopAggregateMetrics",
    "MultiHopEvaluationExample",
    "MultiHopEvaluationPrediction",
    "MultiHopExampleMetrics",
    "SupportFact",
    "aggregate_multihop_metrics",
    "evaluate_multihop_example",
    "normalize_answer",
    "token_f1",
]
