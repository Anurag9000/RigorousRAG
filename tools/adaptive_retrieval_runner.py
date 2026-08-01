"""Bounded execution of deterministic corrective retrieval plans."""

from __future__ import annotations

import hashlib
import itertools
import math
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
_MAX_EVIDENCE_ID_CHARS = 500
_MAX_FINGERPRINT_TEXT_CHARS = 8_000


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


def _safe_identifier(value: Any, maximum: int) -> str:
    if not isinstance(value, str):
        return ""
    rendered = value.strip()
    if not rendered or any(ord(character) < 32 or ord(character) == 127 for character in rendered):
        return ""
    return rendered[:maximum]


def _safe_content(value: Any, maximum: int) -> str:
    if not isinstance(value, str):
        return ""
    rendered = value.strip()
    if not rendered or any(
        (ord(character) < 32 and character not in "\t\r\n")
        or ord(character) == 127
        for character in rendered
    ):
        return ""
    return rendered[:maximum]


def _evidence_id(value: Any, attempt_index: int, item_index: int) -> str:
    """Build a collision-resistant ID without stringifying hostile objects."""

    for name in ("chunk_id", "source_id", "evidence_id"):
        candidate = _safe_identifier(_attr(value, name, None), _MAX_EVIDENCE_ID_CHARS)
        if candidate:
            doc_id = _safe_identifier(_attr(value, "doc_id", None), 200)
            digest = hashlib.sha256(
                "\x1f".join((name, doc_id, candidate)).encode("utf-8")
            ).hexdigest()
            return f"explicit:{digest}"

    doc_id = _safe_identifier(_attr(value, "doc_id", None), 200)
    page = _attr(value, "page_number", None)
    page_text = (
        str(page)
        if isinstance(page, int) and not isinstance(page, bool) and 1 <= page <= 1_000_000
        else ""
    )
    content = ""
    for name in ("quote", "snippet", "text", "content"):
        content = _safe_content(_attr(value, name, None), _MAX_FINGERPRINT_TEXT_CHARS)
        if content:
            break
    section = _safe_identifier(_attr(value, "section", None), 500)
    if doc_id or page_text or content or section:
        digest = hashlib.sha256(
            "\x1f".join((doc_id, page_text, section, content)).encode("utf-8")
        ).hexdigest()
        return f"derived:{digest}"
    return f"anonymous:{attempt_index}:{item_index}"


def _evidence_score(value: Any) -> float:
    raw = _attr(value, "score", None)
    if raw is None:
        metadata = _attr(value, "metadata", {})
        if isinstance(metadata, Mapping):
            try:
                raw = metadata.get("fused_score", metadata.get("relevance", 0.0))
            except Exception:
                raw = 0.0
    if isinstance(raw, bool):
        return 0.0
    try:
        score = float(raw)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(score, 1.0))


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
    for attempt_index, attempt in enumerate(plan.attempts):
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
        for item_index, item in enumerate(returned):
            if len(accumulated) >= _MAX_ACCUMULATED_EVIDENCE:
                break
            evidence_id = _evidence_id(item, attempt_index, item_index)
            current = accumulated.get(evidence_id)
            if current is None or _evidence_score(item) > _evidence_score(current):
                accumulated[evidence_id] = item
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
    abstain = final_signals.decision != "sufficient"
    exhausted = abstain and bool(traces) and len(traces) == len(plan.attempts)
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
