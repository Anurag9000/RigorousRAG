"""Bounded provenance-preserving execution for query-decomposition plans."""

from __future__ import annotations

import itertools
import math
import operator
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Any

from tools.query_decomposition import DecompositionPlan, Subquestion

_MAX_EVIDENCE_PER_HOP = 50
_MAX_TOTAL_EVIDENCE = 200
_MAX_DEPENDENCY_EVIDENCE = 100


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


def _positive_float(value: Any, label: str, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or not 0.0 < parsed <= maximum:
        raise ValueError(f"{label} must be greater than zero and at most {maximum}.")
    return parsed


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


def _text(value: Any) -> str:
    for name in ("quote", "snippet", "text"):
        candidate = _attr(value, name, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:4_000]
    return ""


def _identifier(value: Any, fallback: str) -> str:
    for name in ("chunk_id", "source_id", "evidence_id"):
        candidate = _attr(value, name, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:500]
    return fallback[:500]


def _score(value: Any) -> float:
    raw = _attr(value, "score", None)
    metadata = _metadata(value)
    if raw is None:
        try:
            raw = metadata.get("fused_score", metadata.get("relevance", 0.0))
        except Exception:
            raw = 0.0
    if isinstance(raw, bool):
        return 0.0
    try:
        parsed = float(raw)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return max(0.0, min(parsed, 1.0))


@dataclass(frozen=True)
class HopEvidence:
    """Evidence with immutable hop and source lineage."""

    evidence_id: str
    hop_id: str
    source_id: str
    doc_id: str | None
    page_number: int | None
    text: str
    score: float
    raw: Any


@dataclass(frozen=True)
class HopTrace:
    hop_id: str
    dependencies: tuple[str, ...]
    status: str
    returned_evidence: int
    accepted_evidence: int
    error_type: str | None = None


@dataclass(frozen=True)
class EvidenceJoin:
    """A provenance-safe grouping; source identities are never collapsed."""

    join_key: str
    supporting_hops: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    maximum_score: float


@dataclass(frozen=True)
class MultiHopResult:
    plan_fingerprint: str
    evidence: tuple[HopEvidence, ...]
    traces: tuple[HopTrace, ...]
    joins: tuple[EvidenceJoin, ...]
    terminal_questions: tuple[str, ...]
    terminal_evidence_count: int
    exhausted: bool
    abstain: bool


def _bounded_results(values: Any, maximum: int) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (str, bytes, bytearray)):
        raise RuntimeError("retrieval returned an invalid evidence collection.")
    try:
        return list(itertools.islice(iter(values), maximum + 1))[:maximum]
    except Exception as exc:
        raise RuntimeError("retrieval returned an invalid evidence collection.") from exc


def _normalize_hop_evidence(hop_id: str, values: Sequence[Any]) -> list[HopEvidence]:
    result: list[HopEvidence] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        source_id = _identifier(raw, f"{hop_id}:{index}")
        evidence_id = f"{hop_id}:{source_id}"
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        doc_id = _attr(raw, "doc_id", None)
        if not isinstance(doc_id, str) or not doc_id.strip():
            doc_id = None
        else:
            doc_id = doc_id.strip()[:200]
        page = _attr(raw, "page_number", None)
        if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
            page = None
        result.append(
            HopEvidence(
                evidence_id=evidence_id[:800],
                hop_id=hop_id,
                source_id=source_id,
                doc_id=doc_id,
                page_number=page,
                text=_text(raw),
                score=round(_score(raw), 6),
                raw=raw,
            )
        )
    return result


def _dependency_rows(
    question: Subquestion,
    evidence_by_hop: Mapping[str, Sequence[HopEvidence]],
) -> tuple[HopEvidence, ...]:
    rows: list[HopEvidence] = []
    for dependency in question.depends_on:
        rows.extend(evidence_by_hop.get(dependency, ()))
        if len(rows) >= _MAX_DEPENDENCY_EVIDENCE:
            break
    return tuple(rows[:_MAX_DEPENDENCY_EVIDENCE])


def _run_one(
    search: Callable[[Subquestion, tuple[HopEvidence, ...]], Sequence[Any] | Iterable[Any]],
    question: Subquestion,
    dependencies: tuple[HopEvidence, ...],
    maximum: int,
) -> list[Any]:
    return _bounded_results(search(question, dependencies), maximum)


def _build_joins(evidence: Sequence[HopEvidence]) -> tuple[EvidenceJoin, ...]:
    groups: dict[str, list[HopEvidence]] = defaultdict(list)
    for item in evidence:
        key = f"doc:{item.doc_id}" if item.doc_id else f"source:{item.source_id}"
        groups[key].append(item)
    joins: list[EvidenceJoin] = []
    for key in sorted(groups):
        rows = groups[key]
        hops = tuple(sorted({row.hop_id for row in rows}))
        if len(hops) < 2:
            continue
        joins.append(
            EvidenceJoin(
                join_key=key,
                supporting_hops=hops,
                evidence_ids=tuple(row.evidence_id for row in rows),
                source_ids=tuple(sorted({row.source_id for row in rows})),
                document_ids=tuple(sorted({row.doc_id for row in rows if row.doc_id})),
                maximum_score=max(row.score for row in rows),
            )
        )
    return tuple(joins)


def run_multihop_retrieval(
    plan: DecompositionPlan,
    *,
    search: Callable[[Subquestion, tuple[HopEvidence, ...]], Sequence[Any] | Iterable[Any]],
    max_workers: int = 4,
    per_hop_limit: int = 10,
    hop_timeout_seconds: float = 30.0,
    require_dependency_evidence: bool = True,
) -> MultiHopResult:
    """Execute independent DAG nodes in parallel and dependent batches serially.

    The callback receives validated dependency evidence separately from the query.
    This keeps source lineage explicit and prevents dependency prose from becoming a
    synthetic citation. A dependent hop can be skipped when its required evidence
    is absent.
    """

    if not isinstance(plan, DecompositionPlan):
        raise ValueError("plan must be a DecompositionPlan.")
    if not callable(search):
        raise ValueError("search must be callable.")
    workers = _integer(max_workers, "max_workers", 1, 16)
    limit = _integer(per_hop_limit, "per_hop_limit", 1, _MAX_EVIDENCE_PER_HOP)
    timeout = _positive_float(hop_timeout_seconds, "hop_timeout_seconds", 600.0)
    if not isinstance(require_dependency_evidence, bool):
        raise ValueError("require_dependency_evidence must be a boolean.")

    by_id = plan.by_id()
    evidence_by_hop: dict[str, tuple[HopEvidence, ...]] = {}
    traces: list[HopTrace] = []
    total = 0

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rigorousrag-hop") as pool:
        for batch in plan.batches:
            futures: dict[str, tuple[Future[list[Any]], tuple[HopEvidence, ...]]] = {}
            for hop_id in batch:
                question = by_id[hop_id]
                dependencies = _dependency_rows(question, evidence_by_hop)
                if (
                    require_dependency_evidence
                    and question.depends_on
                    and any(not evidence_by_hop.get(item) for item in question.depends_on)
                ):
                    evidence_by_hop[hop_id] = ()
                    traces.append(
                        HopTrace(
                            hop_id,
                            question.depends_on,
                            "skipped_missing_dependency_evidence",
                            0,
                            0,
                        )
                    )
                    continue
                futures[hop_id] = (
                    pool.submit(_run_one, search, question, dependencies, limit),
                    dependencies,
                )

            for hop_id in batch:
                if hop_id not in futures:
                    continue
                question = by_id[hop_id]
                future, _dependencies = futures[hop_id]
                error_type: str | None = None
                status = "success"
                try:
                    raw = future.result(timeout=timeout)
                except TimeoutError:
                    raw = []
                    status = "timeout"
                    error_type = "TimeoutError"
                except Exception as exc:
                    raw = []
                    status = "error"
                    error_type = type(exc).__name__[:200]
                normalized = _normalize_hop_evidence(hop_id, raw)
                remaining = max(0, _MAX_TOTAL_EVIDENCE - total)
                accepted = tuple(normalized[:remaining])
                evidence_by_hop[hop_id] = accepted
                total += len(accepted)
                traces.append(
                    HopTrace(
                        hop_id,
                        question.depends_on,
                        status,
                        len(raw),
                        len(accepted),
                        error_type,
                    )
                )

    ordered = tuple(
        item
        for node in plan.subquestions
        for item in evidence_by_hop.get(node.question_id, ())
    )
    terminal_count = sum(len(evidence_by_hop.get(item, ())) for item in plan.terminal_questions)
    exhausted = all(
        trace.status
        in {"success", "error", "timeout", "skipped_missing_dependency_evidence"}
        for trace in traces
    )
    abstain = terminal_count == 0
    return MultiHopResult(
        plan_fingerprint=plan.fingerprint,
        evidence=ordered,
        traces=tuple(traces),
        joins=_build_joins(ordered),
        terminal_questions=plan.terminal_questions,
        terminal_evidence_count=terminal_count,
        exhausted=exhausted,
        abstain=abstain,
    )


__all__ = [
    "EvidenceJoin",
    "HopEvidence",
    "HopTrace",
    "MultiHopResult",
    "run_multihop_retrieval",
]
