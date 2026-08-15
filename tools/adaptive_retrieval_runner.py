"""Bounded execution of deterministic or learned corrective retrieval plans."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from tools.adaptive_policy_runtime import (
    RetrievalPolicyFeatures,
    RetrievalPolicyProvider,
    decide_policy,
)
from tools.adaptive_retrieval import (
    CorrectivePlan,
    EvidenceSignals,
    RetrievalAttempt,
    analyze_query,
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
    # Opaque evidence still needs stable de-duplication within this process. Object
    # identity is deliberately used only as an ephemeral dictionary key: it avoids
    # invoking hostile __str__/__repr__ implementations, collapses the same object
    # returned by corrective attempts, and never escapes into persisted lineage.
    return f"anonymous-object:{id(value):x}"


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
    policy_id: str = "deterministic-adaptive"
    policy_version: str = "1.0.0"
    policy_fallback_used: bool = False
    policy_feature_sha256: str = ""
    policy_decision_sha256: str = ""


def _bounded_results(values: Any, maximum: int) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (str, bytes, bytearray)):
        raise RuntimeError("retrieval returned an invalid evidence collection.")
    try:
        return list(itertools.islice(iter(values), maximum + 1))[:maximum]
    except Exception as exc:
        raise RuntimeError("retrieval returned an invalid evidence collection.") from exc


def _domain_scores(domain_registry: Any, query: str) -> Mapping[str, float]:
    if domain_registry is None or not hasattr(domain_registry, "route"):
        return {}
    output: dict[str, float] = {}
    try:
        routed = domain_registry.route(query, minimum_score=0.0)
    except Exception:
        return {}
    for adapter, features in tuple(routed)[:32]:
        try:
            domain_id = str(adapter.descriptor.domain_id)
            score = max((float(value) for value in features.scores.values()), default=0.0)
        except Exception:
            continue
        if math.isfinite(score):
            output[domain_id[:100]] = max(0.0, min(score, 1.0))
    return output


def _policy_features(query: str, *, domain_registry: Any = None) -> RetrievalPolicyFeatures:
    analysis = analyze_query(query)
    token_count = max(1, analysis.token_count)
    lexical_specificity = min(1.0, (0.45 if analysis.exact_identifier else 0.0) + min(token_count / 30.0, 0.55))
    entity_proxy = int(analysis.comparative) * 2 + int(analysis.exact_identifier)
    return RetrievalPolicyFeatures(
        query_length=max(1, len(query.strip())),
        lexical_specificity=lexical_specificity,
        entity_count=entity_proxy,
        temporal_signal=1.0 if analysis.temporal else 0.0,
        comparison_signal=1.0 if analysis.comparative else 0.0,
        numerical_signal=1.0 if analysis.quantitative else 0.0,
        domain_scores=_domain_scores(domain_registry, query),
    )


def _attempt_from_policy(query: str, action: Any, *, requested_top_k: int) -> RetrievalAttempt:
    route_to_mode = {
        "dense": "dense",
        "sparse": "corpus-sparse",
        "hybrid": "corpus-hybrid",
    }
    mode = route_to_mode.get(action.route)
    if mode is None:
        raise ValueError("learned policy selected a route unsupported by uploaded-document retrieval")
    top_k = max(1, min(50, int(action.top_k), requested_top_k if requested_top_k > 0 else 50))
    candidate_pool = max(top_k, min(50, max(int(action.top_k), top_k * max(1, int(action.depth)))))
    reranker = "heuristic" if bool(action.rerank) else "none"
    return RetrievalAttempt(
        mode=mode,
        top_k=top_k,
        candidate_pool=candidate_pool,
        use_multi_query=bool(action.expand_query),
        use_hyde=False,
        reranker=reranker,
        reason=f"policy:{str(action.reason_code)[:180]}",
    )


def _policy_plan(
    query: str,
    *,
    provider: RetrievalPolicyProvider | None,
    domain_registry: Any,
    top_k: int,
    max_attempts: int,
    max_estimated_cost: int,
) -> tuple[CorrectivePlan, Any]:
    features = _policy_features(query, domain_registry=domain_registry)
    decision = decide_policy(
        features,
        provider=provider,
        allowed_routes=("dense", "sparse", "hybrid"),
    )
    baseline = build_corrective_plan(
        query,
        top_k=top_k,
        max_attempts=max_attempts,
        max_estimated_cost=max_estimated_cost,
    )
    if decision.action.abstain:
        return baseline, decision
    try:
        first = _attempt_from_policy(query, decision.action, requested_top_k=top_k)
    except Exception:
        return baseline, decision

    attempts: list[RetrievalAttempt] = [first]
    seen = {
        (
            first.mode,
            first.top_k,
            first.candidate_pool,
            first.use_multi_query,
            first.use_hyde,
            first.reranker,
        )
    }
    cost = first.estimated_cost
    for attempt in baseline.attempts:
        key = (
            attempt.mode,
            attempt.top_k,
            attempt.candidate_pool,
            attempt.use_multi_query,
            attempt.use_hyde,
            attempt.reranker,
        )
        if key in seen or len(attempts) >= max_attempts:
            continue
        if cost + attempt.estimated_cost > max_estimated_cost:
            continue
        attempts.append(attempt)
        seen.add(key)
        cost += attempt.estimated_cost
    return CorrectivePlan(
        analysis=baseline.analysis,
        signals=None,
        attempts=tuple(attempts),
        estimated_cost=cost,
        abstain_after_exhaustion=True,
    ), decision


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
    policy_provider: RetrievalPolicyProvider | None = None,
    domain_registry: Any = None,
) -> AdaptiveRetrievalResult:
    """Execute an owner-scoped corrective plan with safe learned-policy fallback."""

    if not callable(search):
        raise ValueError("search must be callable.")
    owner = normalize_owner_id(owner_id)
    plan, policy_decision = _policy_plan(
        query,
        provider=policy_provider,
        domain_registry=domain_registry,
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
        policy_id=policy_decision.policy_id,
        policy_version=policy_decision.version,
        policy_fallback_used=policy_decision.fallback_used,
        policy_feature_sha256=policy_decision.feature_sha256,
        policy_decision_sha256=policy_decision.decision_sha256,
    )


__all__ = [
    "AdaptiveAttemptTrace",
    "AdaptiveRetrievalResult",
    "run_adaptive_retrieval",
]
