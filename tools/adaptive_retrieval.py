"""Deterministic adaptive retrieval policy and evidence-sufficiency signals."""

from __future__ import annotations

import itertools
import math
import operator
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._:/+\-][A-Za-z0-9]+)*")
_IDENTIFIER_RE = re.compile(
    r"(?:\b10\.\d{4,9}/\S+\b|\b[A-Z]{2,10}[-_:]?\d{2,}\b|\b\d{4}\.\d{4,5}\b)"
)
_COMPARATIVE = {"compare", "versus", "vs", "difference", "better", "worse", "contrast"}
_TEMPORAL = {"latest", "recent", "before", "after", "trend", "year", "timeline", "changed"}
_QUANTITATIVE = {"effect", "estimate", "rate", "risk", "ratio", "percent", "confidence", "p-value"}
_METHOD = {
    "method",
    "methods",
    "methodology",
    "methodological",
    "protocol",
    "protocols",
    "dataset",
    "datasets",
    "sample",
    "samples",
    "design",
    "designs",
    "procedure",
    "procedures",
    "algorithm",
    "algorithms",
}
_CITATION = {"cite", "citation", "source", "evidence", "paper", "reference"}
_EXPLANATORY = {"why", "how", "explain", "mechanism", "cause", "reason"}
_MAX_QUERY_CHARS = 20_000
_MAX_EVIDENCE = 100
_MODES = {"dense", "corpus-sparse", "corpus-hybrid"}
_RERANKERS = {"none", "heuristic", "cross-encoder"}


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


def _unit(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return max(0.0, min(parsed, 1.0))


def _attr(value: Any, name: str, default: Any = None) -> Any:
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
    metadata = _attr(value, "metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


@dataclass(frozen=True)
class QueryAnalysis:
    intent: str
    token_count: int
    complexity: float
    exact_identifier: bool
    comparative: bool
    temporal: bool
    quantitative: bool
    methodological: bool
    citation_seeking: bool
    explanatory: bool


@dataclass(frozen=True)
class EvidenceSignals:
    evidence_count: int
    unique_documents: int
    top_score: float
    mean_score: float
    strong_evidence_count: int
    provenance_fraction: float
    generation_fraction: float
    source_kind_count: int
    sufficiency: float
    decision: str


@dataclass(frozen=True)
class RetrievalAttempt:
    mode: str
    top_k: int
    candidate_pool: int
    use_multi_query: bool = False
    use_hyde: bool = False
    reranker: str = "none"
    reason: str = "initial"

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError("retrieval mode is invalid.")
        object.__setattr__(self, "top_k", _integer(self.top_k, "top_k", 1, 50))
        object.__setattr__(
            self,
            "candidate_pool",
            _integer(self.candidate_pool, "candidate_pool", self.top_k, 50),
        )
        if not isinstance(self.use_multi_query, bool) or not isinstance(self.use_hyde, bool):
            raise ValueError("query expansion controls must be booleans.")
        if self.reranker not in _RERANKERS:
            raise ValueError("reranker is invalid.")
        if not isinstance(self.reason, str) or not self.reason or len(self.reason) > 200:
            raise ValueError("attempt reason is invalid.")

    @property
    def estimated_cost(self) -> int:
        multiplier = 1.0
        if self.use_multi_query:
            multiplier += 1.5
        if self.use_hyde:
            multiplier += 0.5
        if self.reranker != "none":
            multiplier += 0.5
        return max(1, int(math.ceil(self.candidate_pool * multiplier)))


@dataclass(frozen=True)
class CorrectivePlan:
    analysis: QueryAnalysis
    signals: EvidenceSignals | None
    attempts: tuple[RetrievalAttempt, ...]
    estimated_cost: int
    abstain_after_exhaustion: bool = True

    def __post_init__(self) -> None:
        if not self.attempts:
            raise ValueError("corrective plans require at least one attempt.")
        if self.estimated_cost != sum(item.estimated_cost for item in self.attempts):
            raise ValueError("estimated_cost does not match the attempt sequence.")
        if not isinstance(self.abstain_after_exhaustion, bool):
            raise ValueError("abstain_after_exhaustion must be a boolean.")


def analyze_query(query: str) -> QueryAnalysis:
    if not isinstance(query, str):
        raise ValueError("query must be a string.")
    cleaned = query.strip()
    if (
        not cleaned
        or len(cleaned) > _MAX_QUERY_CHARS
        or any(
            (ord(character) < 32 and character not in "\t\r\n")
            or ord(character) == 127
            for character in cleaned
        )
    ):
        raise ValueError("query is empty, invalid, or too long.")
    tokens = tuple(token.lower() for token in _TOKEN_RE.findall(cleaned)[:1_000])
    words = set(tokens)
    exact = bool(_IDENTIFIER_RE.search(cleaned)) or ('"' in cleaned and cleaned.count('"') >= 2)
    comparative = bool(words & _COMPARATIVE)
    temporal = bool(words & _TEMPORAL)
    quantitative = bool(words & _QUANTITATIVE) or bool(re.search(r"\d+(?:\.\d+)?%", cleaned))
    methodological = bool(words & _METHOD)
    citation = bool(words & _CITATION)
    explanatory = bool(words & _EXPLANATORY)
    complexity_points = (
        min(len(tokens) / 40.0, 1.0)
        + 0.22 * comparative
        + 0.18 * temporal
        + 0.18 * quantitative
        + 0.15 * methodological
        + 0.12 * explanatory
    )
    complexity = min(complexity_points / 1.6, 1.0)
    if exact and len(tokens) <= 12:
        intent = "exact_lookup"
    elif comparative:
        intent = "comparison"
    elif temporal:
        intent = "temporal"
    elif methodological:
        intent = "method"
    elif quantitative:
        intent = "quantitative"
    elif citation:
        intent = "evidence_lookup"
    elif explanatory:
        intent = "explanation"
    else:
        intent = "general"
    return QueryAnalysis(
        intent=intent,
        token_count=len(tokens),
        complexity=round(complexity, 6),
        exact_identifier=exact,
        comparative=comparative,
        temporal=temporal,
        quantitative=quantitative,
        methodological=methodological,
        citation_seeking=citation,
        explanatory=explanatory,
    )


def evaluate_evidence(values: Iterable[Any]) -> EvidenceSignals:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("evidence must be an iterable of evidence records.")
    try:
        rows = list(itertools.islice(iter(values), _MAX_EVIDENCE + 1))
    except Exception as exc:
        raise ValueError("evidence is not safely iterable.") from exc
    if len(rows) > _MAX_EVIDENCE:
        rows = rows[:_MAX_EVIDENCE]
    scores: list[float] = []
    documents: set[str] = set()
    kinds: set[str] = set()
    provenance = 0
    generations = 0
    for row in rows:
        metadata = _metadata(row)
        raw_score = _attr(row, "score", None)
        if raw_score is None:
            try:
                raw_score = metadata.get("fused_score", metadata.get("relevance", 0.0))
            except Exception:
                raw_score = 0.0
        scores.append(_unit(raw_score))
        doc_id = _attr(row, "doc_id", None)
        if isinstance(doc_id, str) and doc_id.strip():
            documents.add(doc_id.strip()[:200])
        kind = _attr(row, "source_kind", None)
        if kind is None:
            try:
                kind = metadata.get("evidence_kind")
            except Exception:
                kind = None
        if isinstance(kind, str) and kind:
            kinds.add(kind[:100])
        page = _attr(row, "page_number", None)
        source_id = _attr(row, "source_id", None)
        if (
            isinstance(page, int)
            and not isinstance(page, bool)
            and page > 0
        ) or (isinstance(source_id, str) and source_id.strip()):
            provenance += 1
        sequence = _attr(row, "generation_sequence", None)
        if sequence is None:
            try:
                sequence = metadata.get("generation_sequence")
            except Exception:
                sequence = None
        if isinstance(sequence, int) and not isinstance(sequence, bool) and sequence > 0:
            generations += 1
    count = len(scores)
    if not count:
        return EvidenceSignals(0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, "empty")
    top = max(scores)
    mean = sum(scores) / count
    strong = sum(1 for score in scores if score >= 0.65)
    provenance_fraction = provenance / count
    generation_fraction = generations / count
    document_factor = min(len(documents) / 3.0, 1.0)
    count_factor = min(count / 5.0, 1.0)
    strength_factor = min(strong / 3.0, 1.0)
    sufficiency = (
        0.24 * top
        + 0.18 * mean
        + 0.16 * count_factor
        + 0.14 * document_factor
        + 0.12 * strength_factor
        + 0.08 * provenance_fraction
        + 0.08 * generation_fraction
    )
    if sufficiency >= 0.68 and strong >= 1:
        decision = "sufficient"
    elif sufficiency >= 0.38:
        decision = "weak"
    else:
        decision = "insufficient"
    return EvidenceSignals(
        evidence_count=count,
        unique_documents=len(documents),
        top_score=round(top, 6),
        mean_score=round(mean, 6),
        strong_evidence_count=strong,
        provenance_fraction=round(provenance_fraction, 6),
        generation_fraction=round(generation_fraction, 6),
        source_kind_count=len(kinds),
        sufficiency=round(sufficiency, 6),
        decision=decision,
    )


def initial_attempt(query: str, *, top_k: int = 5) -> RetrievalAttempt:
    analysis = analyze_query(query)
    requested = _integer(top_k, "top_k", 1, 50)
    pool = min(50, max(requested, 12 if analysis.complexity < 0.5 else 24))
    if analysis.intent == "exact_lookup":
        return RetrievalAttempt("corpus-sparse", requested, pool, reason="exact_identifier_or_quote")
    if analysis.intent in {"comparison", "temporal", "quantitative", "method"}:
        return RetrievalAttempt(
            "corpus-hybrid", requested, pool, use_multi_query=True,
            reranker="heuristic", reason=f"complex_{analysis.intent}",
        )
    if analysis.intent in {"evidence_lookup", "explanation"}:
        return RetrievalAttempt(
            "corpus-hybrid", requested, pool, reranker="heuristic",
            reason=f"{analysis.intent}_with_provenance",
        )
    return RetrievalAttempt("dense", requested, pool, reason="low_complexity_general")


def build_corrective_plan(
    query: str,
    *,
    signals: EvidenceSignals | None = None,
    top_k: int = 5,
    max_attempts: int = 4,
    max_estimated_cost: int = 300,
) -> CorrectivePlan:
    analysis = analyze_query(query)
    attempts_limit = _integer(max_attempts, "max_attempts", 1, 6)
    cost_limit = _integer(max_estimated_cost, "max_estimated_cost", 1, 5_000)
    initial = initial_attempt(query, top_k=top_k)
    if signals is not None and not isinstance(signals, EvidenceSignals):
        raise ValueError("signals must be EvidenceSignals or null.")
    if signals is not None and signals.decision == "sufficient":
        return CorrectivePlan(analysis, signals, (initial,), initial.estimated_cost)
    candidates = [initial]
    expanded_pool = min(50, max(initial.candidate_pool * 2, initial.top_k))
    candidates.append(
        RetrievalAttempt(
            "corpus-hybrid", initial.top_k, expanded_pool,
            use_multi_query=True, reranker="heuristic",
            reason="broaden_independent_corpus_retrieval",
        )
    )
    if analysis.explanatory and not analysis.exact_identifier:
        candidates.append(
            RetrievalAttempt(
                "corpus-hybrid", initial.top_k, expanded_pool,
                use_multi_query=True, use_hyde=True, reranker="heuristic",
                reason="semantic_hypothesis_expansion",
            )
        )
    else:
        candidates.append(
            RetrievalAttempt(
                "corpus-sparse" if analysis.exact_identifier else "corpus-hybrid",
                initial.top_k, 50, use_multi_query=not analysis.exact_identifier,
                reranker="cross-encoder", reason="high_precision_second_stage",
            )
        )
    candidates.append(
        RetrievalAttempt(
            "corpus-hybrid", min(10, max(initial.top_k, 5)), 50,
            use_multi_query=True, use_hyde=analysis.explanatory,
            reranker="cross-encoder", reason="final_bounded_retrieval_attempt",
        )
    )
    selected: list[RetrievalAttempt] = []
    seen: set[tuple[Any, ...]] = set()
    cost = 0
    for attempt in candidates:
        key = (
            attempt.mode, attempt.top_k, attempt.candidate_pool,
            attempt.use_multi_query, attempt.use_hyde, attempt.reranker,
        )
        if key in seen:
            continue
        projected = cost + attempt.estimated_cost
        if selected and projected > cost_limit:
            break
        seen.add(key)
        selected.append(attempt)
        cost = projected
        if len(selected) >= attempts_limit:
            break
    return CorrectivePlan(
        analysis=analysis,
        signals=signals,
        attempts=tuple(selected),
        estimated_cost=cost,
        abstain_after_exhaustion=True,
    )


__all__ = [
    "CorrectivePlan", "EvidenceSignals", "QueryAnalysis", "RetrievalAttempt",
    "analyze_query", "build_corrective_plan", "evaluate_evidence", "initial_attempt",
]
