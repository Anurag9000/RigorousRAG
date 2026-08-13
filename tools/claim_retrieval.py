"""Claim-conditioned retrieval for support and explicit counterfactual evidence search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from tools.hybrid_retrieval import RetrievalCandidate

_MAX_CLAIMS = 128
_MAX_RESULTS_PER_QUERY = 100


ClaimRetriever = Callable[[str, int], Sequence[RetrievalCandidate]]
CounterfactualBuilder = Callable[[str], str]


def _claim(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("claims must be strings.")
    text = value.strip()
    if not text or len(text) > 20_000:
        raise ValueError("claim is invalid.")
    return text


def _default_counterfactual(claim: str) -> str:
    return f"Find credible evidence that contradicts, falsifies, or limits this claim: {claim}"


@dataclass(frozen=True)
class ClaimEvidenceBundle:
    claim: str
    support_query: str
    counter_query: str | None
    support: tuple[RetrievalCandidate, ...]
    counter: tuple[RetrievalCandidate, ...]

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.source_id for item in (*self.support, *self.counter)}))


def _dedupe(values: Sequence[RetrievalCandidate], limit: int) -> tuple[RetrievalCandidate, ...]:
    result: list[RetrievalCandidate] = []
    seen: set[str] = set()
    for item in values[:_MAX_RESULTS_PER_QUERY]:
        if not isinstance(item, RetrievalCandidate):
            raise ValueError("retriever returned an invalid candidate.")
        if item.candidate_id in seen:
            continue
        seen.add(item.candidate_id)
        result.append(item)
        if len(result) >= limit:
            break
    return tuple(result)


def retrieve_claim_evidence(
    claims: Sequence[str],
    retriever: ClaimRetriever,
    *,
    per_claim: int = 5,
    include_counterfactual: bool = True,
    counterfactual_builder: CounterfactualBuilder | None = None,
) -> tuple[ClaimEvidenceBundle, ...]:
    """Retrieve evidence independently for each claim and its counter-hypothesis."""

    if isinstance(claims, (str, bytes, bytearray)) or not claims or len(claims) > _MAX_CLAIMS:
        raise ValueError("claims must be a bounded non-empty sequence.")
    if not callable(retriever):
        raise ValueError("retriever must be callable.")
    if isinstance(per_claim, bool) or not isinstance(per_claim, int) or not 1 <= per_claim <= _MAX_RESULTS_PER_QUERY:
        raise ValueError("per_claim is invalid.")
    if not isinstance(include_counterfactual, bool):
        raise ValueError("include_counterfactual must be boolean.")
    builder = counterfactual_builder or _default_counterfactual
    if not callable(builder):
        raise ValueError("counterfactual_builder must be callable.")
    bundles: list[ClaimEvidenceBundle] = []
    for raw_claim in claims:
        claim = _claim(raw_claim)
        support_query = claim
        support = _dedupe(retriever(support_query, per_claim), per_claim)
        counter_query: str | None = None
        counter: tuple[RetrievalCandidate, ...] = ()
        if include_counterfactual:
            built = builder(claim)
            if not isinstance(built, str) or not built.strip() or len(built) > 20_000:
                raise ValueError("counterfactual_builder returned an invalid query.")
            counter_query = built.strip()
            counter = _dedupe(retriever(counter_query, per_claim), per_claim)
        bundles.append(
            ClaimEvidenceBundle(
                claim=claim,
                support_query=support_query,
                counter_query=counter_query,
                support=support,
                counter=counter,
            )
        )
    return tuple(bundles)


def claim_source_diversity(bundles: Sequence[ClaimEvidenceBundle]) -> Mapping[str, int]:
    if isinstance(bundles, (str, bytes, bytearray)) or len(bundles) > _MAX_CLAIMS:
        raise ValueError("bundles must be a bounded sequence.")
    result: dict[str, int] = {}
    for bundle in bundles:
        if not isinstance(bundle, ClaimEvidenceBundle):
            raise ValueError("bundles contains an invalid value.")
        result[bundle.claim] = len(bundle.source_ids)
    return result


__all__ = ["ClaimEvidenceBundle", "claim_source_diversity", "retrieve_claim_evidence"]
