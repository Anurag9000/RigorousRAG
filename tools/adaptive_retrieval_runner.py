"""Bounded execution of deterministic corrective retrieval plans."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from tools.adaptive_retrieval import (
    EvidenceSignals,
    RetrievalAttempt,
    build_corrective_plan,
    evaluate_evidence,
)
from tools.security import normalize_owner_id

_MAX_ACCUMULATED_EVIDENCE = 100


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


def _evidence_id(value: Any, index: int) -> str:
    for name in ("chunk_id", "source_id", "evidence_id"):
        candidate = _attr(value, name, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:500]
    doc_id = _attr(value, "doc_id", "unknown")
    return f"{str(doc_id)[:200]}:{index}"


@dataclass(frozen=True)
class AdaptiveAttemptTrace:
    attempt: RetrievalAttempt
    returned_evidence: int
    accumulated_evidence: int
    signals: EvidenceSignals
    error_type: str | None = None


@dataclass(frozen=True)
class AdaptiveRetrievalResult:
    evidence: tuple[Any, ...]
    traces: tuple[AdaptiveAttemptTrace, ...]
    final_signals: EvidenceSignals
    exhausted: bool
    abstain: bool
    estimated_cost: int


def _bounded_results(values: Any, maximum: int) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (str, bytes, bytearray)):
        raise RuntimeError("retrieval returned an invalid evidence collection.")
    try:
        return list(itertools.islice(iter(values), maximum + 1))[:maximum]
    except Exception as exc:
        raise RuntimeError("retrieval returned an invalid evidence collection.") from exc


def run_adaptive_retrieval(
    query: str,
    *,
    search: Callable[..., Sequence[Any] | Iterable[Any]],
    owner_id: str,
    doc_id: str | None = None,
    top_k: int = 5,
    max_attempts: int = 4,
    max_estimated_cost: int = 300,
    agent_client: Any = None,
    expansion_model: str = "gpt-4o-mini",
    diversity_lambda: float = 0.82,
) -> AdaptiveRetrievalResult:
    """Execute a corrective plan without recursively selecting adaptive mode."""

    if not callable(search):
        raise ValueError("search must be callable.")
    owner = normalize_owner_id(owner_id)
    plan = build_corrective_plan(
        query,
        top_k=top_k,
        max_attempts=max_attempts,
        max_estimated_cost=max_estimated_cost,
    )
    accumulated: dict[str, Any] = {}
    traces: list[AdaptiveAttemptTrace] = []
    final_signals = evaluate_evidence(())
    for attempt in plan.attempts:
        error_type: str | None = None
        try:
            returned = _bounded_results(
                search(
                    query,
                    owner_id=owner,
                    doc_id=doc_id,
                    use_hyde=attempt.use_hyde,
                    use_multi_query=attempt.use_multi_query,
                    agent_client=agent_client,
                    expansion_model=expansion_model,
                    n_results=attempt.top_k,
                    retrieval_mode=attempt.mode,
                    reranker=attempt.reranker,
                    candidate_pool=attempt.candidate_pool,
                    diversity_lambda=diversity_lambda,
                ),
                _MAX_ACCUMULATED_EVIDENCE,
            )
        except Exception as exc:
            returned = []
            error_type = type(exc).__name__[:200]
        for index, item in enumerate(returned):
            if len(accumulated) >= _MAX_ACCUMULATED_EVIDENCE:
                break
            accumulated[_evidence_id(item, index)] = item
        final_signals = evaluate_evidence(accumulated.values())
        traces.append(
            AdaptiveAttemptTrace(
                attempt=attempt,
                returned_evidence=len(returned),
                accumulated_evidence=len(accumulated),
                signals=final_signals,
                error_type=error_type,
            )
        )
        if final_signals.decision == "sufficient":
            break
    exhausted = bool(traces) and len(traces) == len(plan.attempts)
    abstain = final_signals.decision != "sufficient"
    return AdaptiveRetrievalResult(
        evidence=tuple(accumulated.values()),
        traces=tuple(traces),
        final_signals=final_signals,
        exhausted=exhausted,
        abstain=abstain,
        estimated_cost=sum(trace.attempt.estimated_cost for trace in traces),
    )


__all__ = [
    "AdaptiveAttemptTrace",
    "AdaptiveRetrievalResult",
    "run_adaptive_retrieval",
]
