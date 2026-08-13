"""Composable governed answer-flow orchestration around existing RAG primitives.

The wrapper intentionally does not edit the legacy HTTP/search boundary. All retrieval,
cache, security, and telemetry dependencies are injected, making policy ordering easy
to test: security -> normalized routing -> cache/retrieve -> uncertainty -> review/block
-> cache write. Review/block outcomes are never cached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from evaluation.uncertainty import RagUncertaintySignal, combine_rag_uncertainty
from tools.domain_routing import DomainClassifier, DomainRoutingDecision, route_query_by_domain
from tools.query_normalization import EntityResolver, NormalizedQueryContext, TemporalParser, normalize_query_context
from tools.review_routing import ReviewDecision, ReviewPolicy, route_for_review
from tools.review_store import ReviewRecord, ReviewStore
from tools.security import normalize_owner_id

_MAX_ANSWER = 2_000_000
_MAX_EVIDENCE = 10_000


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 1.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be between 0 and 1.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return parsed


@dataclass(frozen=True)
class AnswerMaterial:
    answer: str
    retrieval_confidence: float
    generation_confidence: float
    evidence_conflict: float = 0.0
    proof_completeness: float = 1.0
    independent_sources: int = 1
    evidence_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str) or len(self.answer) > _MAX_ANSWER:
            raise ValueError("answer must be a bounded string.")
        for name in ("retrieval_confidence", "generation_confidence", "evidence_conflict", "proof_completeness"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        if isinstance(self.independent_sources, bool) or not isinstance(self.independent_sources, int) or not 0 <= self.independent_sources <= _MAX_EVIDENCE:
            raise ValueError("independent_sources is invalid.")
        if not isinstance(self.evidence_ids, tuple) or len(self.evidence_ids) > _MAX_EVIDENCE:
            raise ValueError("evidence_ids must be a bounded tuple.")
        if any(not isinstance(value, str) or not value.strip() or len(value) > 500 for value in self.evidence_ids):
            raise ValueError("evidence_ids contains an invalid value.")
        metadata = {} if self.metadata is None else self.metadata
        if not isinstance(metadata, Mapping) or len(metadata) > 100:
            raise ValueError("metadata must be a bounded mapping.")
        object.__setattr__(self, "metadata", dict(metadata))


@dataclass(frozen=True)
class AnswerFlowResult:
    owner_id: str
    request_id: str
    context: NormalizedQueryContext
    routing: DomainRoutingDecision
    material: AnswerMaterial | None
    uncertainty: RagUncertaintySignal | None
    decision: ReviewDecision
    review_record: ReviewRecord | None
    cache_hit: bool

    @property
    def answer(self) -> str | None:
        return None if self.material is None else self.material.answer


Retriever = Callable[[str, str, NormalizedQueryContext], AnswerMaterial]
CacheLookup = Callable[[str, str, NormalizedQueryContext], AnswerMaterial | None]
CacheStore = Callable[[str, str, NormalizedQueryContext, AnswerMaterial], None]
SecurityCheck = Callable[[str, str], bool]
EventHook = Callable[[str, Mapping[str, Any]], None]


def run_governed_answer_flow(
    query: str,
    *,
    owner_id: str,
    request_id: str,
    retrieve: Retriever,
    review_store: ReviewStore | None = None,
    review_policy: ReviewPolicy | None = None,
    classifier: DomainClassifier | None = None,
    classifier_version: str | None = None,
    entity_resolver: EntityResolver | None = None,
    temporal_parser: TemporalParser | None = None,
    cache_lookup: CacheLookup | None = None,
    cache_store: CacheStore | None = None,
    security_check: SecurityCheck | None = None,
    event_hook: EventHook | None = None,
    abstain_threshold: float = 0.5,
) -> AnswerFlowResult:
    """Run one governed answer request with fail-closed policy ordering."""

    if not isinstance(query, str) or not query.strip() or len(query) > 20_000:
        raise ValueError("query must be a bounded non-empty string.")
    owner = normalize_owner_id(owner_id)
    if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 500:
        raise ValueError("request_id is invalid.")
    request = request_id.strip()
    if not callable(retrieve):
        raise ValueError("retrieve must be callable.")
    if review_store is not None and not isinstance(review_store, ReviewStore):
        raise ValueError("review_store must be ReviewStore.")
    for hook, label in ((cache_lookup, "cache_lookup"), (cache_store, "cache_store"), (security_check, "security_check"), (event_hook, "event_hook")):
        if hook is not None and not callable(hook):
            raise ValueError(f"{label} must be callable.")

    def emit(event: str, payload: Mapping[str, Any]) -> None:
        if event_hook is not None:
            event_hook(event, dict(payload))

    if security_check is not None and not bool(security_check(owner, query)):
        decision = route_for_review(aggregate_uncertainty=0.0, security_violation=True, policy=review_policy)
        # Blocked input is processed only by deterministic in-process fallbacks; it is
        # never forwarded to caller-injected classifiers, entity resolvers, or parsers.
        context = normalize_query_context(query)
        routing = route_query_by_domain(query)
        emit("blocked", {"request_id": request, "reason": "security_violation"})
        return AnswerFlowResult(owner, request, context, routing, None, None, decision, None, False)

    context = normalize_query_context(query, entity_resolver=entity_resolver, temporal_parser=temporal_parser)
    routing = route_query_by_domain(query, classifier=classifier, classifier_version=classifier_version)
    emit("routed", {"request_id": request, "route": routing.route, "domain": routing.domain})

    material: AnswerMaterial | None = None
    cache_hit = False
    if cache_lookup is not None:
        cached = cache_lookup(query, routing.route, context)
        if cached is not None:
            if not isinstance(cached, AnswerMaterial):
                raise RuntimeError("cache_lookup returned an invalid value.")
            material = cached
            cache_hit = True
            emit("cache_hit", {"request_id": request, "route": routing.route})
    if material is None:
        material = retrieve(query, routing.route, context)
        if not isinstance(material, AnswerMaterial):
            raise RuntimeError("retrieve returned an invalid value.")
        emit("retrieved", {"request_id": request, "route": routing.route, "evidence_count": len(material.evidence_ids)})

    uncertainty = combine_rag_uncertainty(
        retrieval_confidence=material.retrieval_confidence,
        generation_confidence=material.generation_confidence,
        evidence_conflict=material.evidence_conflict,
        proof_completeness=material.proof_completeness,
        abstain_threshold=abstain_threshold,
    )
    decision = route_for_review(
        aggregate_uncertainty=uncertainty.aggregate_uncertainty,
        evidence_conflict=material.evidence_conflict,
        proof_completeness=material.proof_completeness,
        independent_sources=material.independent_sources,
        policy=review_policy,
    )
    review_record: ReviewRecord | None = None
    if decision.route == "human_review":
        if review_store is not None:
            review_record = review_store.enqueue(
                owner_id=owner,
                request_id=request,
                decision=decision,
                query=query,
                metadata={
                    "route": routing.route,
                    "domain": routing.domain,
                    "evidence_count": len(material.evidence_ids),
                    "cache_hit": cache_hit,
                },
            )
        emit("human_review", {"request_id": request, "priority": decision.priority, "reasons": decision.reasons})
    elif decision.route == "block":
        emit("blocked", {"request_id": request, "reasons": decision.reasons})
    else:
        if cache_store is not None and not cache_hit:
            cache_store(query, routing.route, context, material)
            emit("cache_store", {"request_id": request, "route": routing.route})
        emit("automatic", {"request_id": request, "confidence": uncertainty.confidence})
    return AnswerFlowResult(owner, request, context, routing, material, uncertainty, decision, review_record, cache_hit)


__all__ = [
    "AnswerFlowResult",
    "AnswerMaterial",
    "CacheLookup",
    "CacheStore",
    "EventHook",
    "Retriever",
    "SecurityCheck",
    "run_governed_answer_flow",
]
