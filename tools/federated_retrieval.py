"""Content-blind federated retrieval contracts for private research collections.

Participating institutions execute retrieval locally and return bounded evidence handles,
content digests, scores/ranks, and explicitly disclosed metadata. The broker never needs
private source text to merge results. Fetching source bytes is a separate, explicit,
authorization-bearing operation implemented by an injected resolver.

This module defines contracts and deterministic merge logic only. It does not discover
institutions, open network connections, or claim cryptographic privacy between parties.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

_MAX_PROVIDERS = 1000
_MAX_RESULTS_PER_PROVIDER = 10_000
_MAX_TOTAL_RESULTS = 100_000


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if (not selected and not allow_empty) or len(selected) > maximum or "\x00" in selected:
        raise ValueError(f"{label} is invalid")
    return selected


def _sha(value: str, label: str, *, allow_empty: bool = False) -> str:
    selected = _text(value, label, 64, allow_empty=allow_empty).lower()
    if selected and (len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected)):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _bounded_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or len(value) > 100:
        raise ValueError("disclosed_metadata must be a bounded mapping")
    encoded = _canonical(dict(value))
    if len(encoded) > 64_000:
        raise ValueError("disclosed_metadata exceeds the byte limit")
    return dict(value)


@dataclass(frozen=True)
class FederatedCollection:
    provider_id: str
    collection_id: str
    collection_fingerprint: str
    disclosure_policy_fingerprint: str
    modalities: tuple[str, ...] = ("text",)
    jurisdiction: str = ""
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _text(self.provider_id, "provider_id", 256))
        object.__setattr__(self, "collection_id", _text(self.collection_id, "collection_id", 500))
        object.__setattr__(self, "collection_fingerprint", _sha(self.collection_fingerprint, "collection_fingerprint"))
        object.__setattr__(self, "disclosure_policy_fingerprint", _sha(self.disclosure_policy_fingerprint, "disclosure_policy_fingerprint"))
        modalities = tuple(sorted(set(_text(item, "modality", 64).lower() for item in self.modalities)))
        if not modalities or len(modalities) > 100:
            raise ValueError("modalities are invalid")
        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "jurisdiction", _text(self.jurisdiction, "jurisdiction", 256, allow_empty=True))
        details: dict[str, str] = {}
        for key, value in dict(self.metadata or {}).items():
            details[_text(key, "metadata key", 128)] = _text(value, "metadata value", 1000)
        if len(details) > 100:
            raise ValueError("too many collection metadata entries")
        object.__setattr__(self, "metadata", details)


@dataclass(frozen=True)
class FederatedSearchRequest:
    request_id: str
    owner_id: str
    project_id: str
    query_sha256: str
    purpose: str
    disclosure_policy_fingerprint: str
    max_results: int = 100
    modalities: tuple[str, ...] = ()
    query_payload: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id", 500))
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id", 256))
        object.__setattr__(self, "project_id", _text(self.project_id, "project_id", 500))
        object.__setattr__(self, "query_sha256", _sha(self.query_sha256, "query_sha256"))
        object.__setattr__(self, "purpose", _text(self.purpose, "purpose", 256).lower())
        object.__setattr__(self, "disclosure_policy_fingerprint", _sha(self.disclosure_policy_fingerprint, "disclosure_policy_fingerprint"))
        if isinstance(self.max_results, bool) or not isinstance(self.max_results, int) or not 1 <= self.max_results <= _MAX_RESULTS_PER_PROVIDER:
            raise ValueError("max_results is invalid")
        object.__setattr__(self, "modalities", tuple(sorted(set(_text(item, "modality", 64).lower() for item in self.modalities))))
        object.__setattr__(self, "query_payload", _text(self.query_payload, "query_payload", 100_000, allow_empty=True))

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["query_payload"] = hashlib.sha256(self.query_payload.encode("utf-8")).hexdigest()
        return hashlib.sha256(_canonical(payload)).hexdigest()


@dataclass(frozen=True)
class FederatedEvidenceHandle:
    provider_id: str
    collection_id: str
    evidence_id: str
    content_sha256: str
    rank: int
    score: float
    modality: str
    source_identity_sha256: str = ""
    disclosed_metadata: Mapping[str, Any] | None = None
    resolver_hint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _text(self.provider_id, "provider_id", 256))
        object.__setattr__(self, "collection_id", _text(self.collection_id, "collection_id", 500))
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, "evidence_id", 500))
        object.__setattr__(self, "content_sha256", _sha(self.content_sha256, "content_sha256"))
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or not 1 <= self.rank <= _MAX_RESULTS_PER_PROVIDER:
            raise ValueError("rank is invalid")
        object.__setattr__(self, "score", _finite(self.score, "score"))
        object.__setattr__(self, "modality", _text(self.modality, "modality", 64).lower())
        object.__setattr__(self, "source_identity_sha256", _sha(self.source_identity_sha256, "source_identity_sha256", allow_empty=True))
        object.__setattr__(self, "disclosed_metadata", _bounded_metadata(dict(self.disclosed_metadata or {})))
        object.__setattr__(self, "resolver_hint", _text(self.resolver_hint, "resolver_hint", 2000, allow_empty=True))

    @property
    def global_evidence_id(self) -> str:
        payload = {
            "provider_id": self.provider_id,
            "collection_id": self.collection_id,
            "evidence_id": self.evidence_id,
            "content_sha256": self.content_sha256,
        }
        return hashlib.sha256(_canonical(payload)).hexdigest()


@dataclass(frozen=True)
class FederatedSearchResponse:
    request_fingerprint: str
    collection: FederatedCollection
    handles: tuple[FederatedEvidenceHandle, ...]
    provider_trace_fingerprint: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_fingerprint", _sha(self.request_fingerprint, "request_fingerprint"))
        if not isinstance(self.collection, FederatedCollection):
            raise TypeError("collection must be FederatedCollection")
        if len(self.handles) > _MAX_RESULTS_PER_PROVIDER or any(not isinstance(item, FederatedEvidenceHandle) for item in self.handles):
            raise ValueError("handles are invalid")
        if any(item.provider_id != self.collection.provider_id or item.collection_id != self.collection.collection_id for item in self.handles):
            raise ValueError("handle provider/collection does not match response collection")
        ranks = [item.rank for item in self.handles]
        if len(set(ranks)) != len(ranks):
            raise ValueError("response contains duplicate ranks")
        evidence_ids = [item.evidence_id for item in self.handles]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("response contains duplicate evidence_id values")
        object.__setattr__(self, "handles", tuple(sorted(self.handles, key=lambda item: item.rank)))
        object.__setattr__(self, "provider_trace_fingerprint", _sha(self.provider_trace_fingerprint, "provider_trace_fingerprint"))
        object.__setattr__(self, "warnings", tuple(dict.fromkeys(_text(item, "warning", 2000) for item in self.warnings)))


class FederatedSearchProvider(Protocol):
    @property
    def collection(self) -> FederatedCollection: ...
    def search(self, request: FederatedSearchRequest) -> FederatedSearchResponse: ...


@dataclass(frozen=True)
class EvidenceAuthorization:
    principal_id: str
    project_id: str
    purpose: str
    token_fingerprint: str
    expires_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "principal_id", _text(self.principal_id, "principal_id", 256))
        object.__setattr__(self, "project_id", _text(self.project_id, "project_id", 500))
        object.__setattr__(self, "purpose", _text(self.purpose, "purpose", 256).lower())
        object.__setattr__(self, "token_fingerprint", _sha(self.token_fingerprint, "token_fingerprint"))
        object.__setattr__(self, "expires_at", _finite(self.expires_at, "expires_at"))


class AuthorizedEvidenceResolver(Protocol):
    def resolve(self, handle: FederatedEvidenceHandle, authorization: EvidenceAuthorization) -> bytes: ...


@dataclass(frozen=True)
class FederatedMergedHit:
    global_evidence_id: str
    handle: FederatedEvidenceHandle
    fused_score: float
    provider_rank: int
    provider_weight: float


@dataclass(frozen=True)
class FederatedMergeResult:
    request_fingerprint: str
    hits: tuple[FederatedMergedHit, ...]
    participating_collections: tuple[str, ...]
    failed_or_missing_collections: tuple[str, ...]
    warnings: tuple[str, ...]
    fingerprint: str


def merge_federated_responses(
    request: FederatedSearchRequest,
    responses: Sequence[FederatedSearchResponse],
    *,
    expected_collections: Sequence[FederatedCollection] = (),
    provider_weights: Mapping[str, float] | None = None,
    rrf_k: float = 60.0,
    limit: int = 100,
    max_per_provider: int = 50,
) -> FederatedMergeResult:
    if not isinstance(request, FederatedSearchRequest):
        raise TypeError("request must be FederatedSearchRequest")
    response_items = tuple(responses)
    if len(response_items) > _MAX_PROVIDERS or any(not isinstance(item, FederatedSearchResponse) for item in response_items):
        raise ValueError("responses are invalid")
    expected_items = tuple(expected_collections)
    if len(expected_items) > _MAX_PROVIDERS or any(not isinstance(item, FederatedCollection) for item in expected_items):
        raise ValueError("expected_collections contain an invalid value")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_TOTAL_RESULTS:
        raise ValueError("limit is invalid")
    if isinstance(max_per_provider, bool) or not isinstance(max_per_provider, int) or not 1 <= max_per_provider <= _MAX_RESULTS_PER_PROVIDER:
        raise ValueError("max_per_provider is invalid")
    selected_rrf_k = _finite(rrf_k, "rrf_k")
    if selected_rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    weights = dict(provider_weights or {})
    normalized_weights: dict[str, float] = {}
    for provider_id, value in weights.items():
        selected_provider = _text(provider_id, "provider_id", 256)
        weight = _finite(value, "provider_weight")
        if weight <= 0:
            raise ValueError("provider weights must be positive")
        normalized_weights[selected_provider] = weight

    expected_by_key = {(item.provider_id, item.collection_id): item for item in expected_items}
    if len(expected_by_key) != len(expected_items):
        raise ValueError("expected_collections contain duplicate provider/collection identities")
    seen_keys: set[tuple[str, str]] = set()
    merged: list[FederatedMergedHit] = []
    warnings: list[str] = []
    for response in response_items:
        if response.request_fingerprint != request.fingerprint:
            raise ValueError("response request fingerprint does not match request")
        collection = response.collection
        key = (collection.provider_id, collection.collection_id)
        if key in seen_keys:
            raise ValueError("duplicate response for provider collection")
        if expected_by_key and key not in expected_by_key:
            raise ValueError("response came from a provider collection that was not authorized for this request")
        seen_keys.add(key)
        if collection.disclosure_policy_fingerprint != request.disclosure_policy_fingerprint:
            raise ValueError("collection disclosure policy does not match request policy")
        expected = expected_by_key.get(key)
        if expected is not None and expected.collection_fingerprint != collection.collection_fingerprint:
            raise ValueError("response collection fingerprint differs from the authorized collection generation")
        weight = normalized_weights.get(collection.provider_id, 1.0)
        count = 0
        for handle in response.handles:
            if count >= max_per_provider:
                break
            if request.modalities and handle.modality not in request.modalities:
                continue
            fused = weight / (selected_rrf_k + handle.rank)
            merged.append(
                FederatedMergedHit(
                    global_evidence_id=handle.global_evidence_id,
                    handle=handle,
                    fused_score=fused,
                    provider_rank=handle.rank,
                    provider_weight=weight,
                )
            )
            count += 1
        warnings.extend(f"{collection.provider_id}:{collection.collection_id}:{item}" for item in response.warnings)

    merged.sort(key=lambda item: (-item.fused_score, item.handle.provider_id, item.handle.collection_id, item.provider_rank, item.global_evidence_id))
    effective_limit = min(limit, request.max_results)
    deduped: list[FederatedMergedHit] = []
    seen_global: set[str] = set()
    for item in merged:
        if item.global_evidence_id in seen_global:
            continue
        seen_global.add(item.global_evidence_id)
        deduped.append(item)
        if len(deduped) >= effective_limit:
            break

    missing_keys = sorted(set(expected_by_key) - seen_keys)
    participating = tuple(sorted(f"{provider}:{collection}" for provider, collection in seen_keys))
    missing = tuple(f"{provider}:{collection}" for provider, collection in missing_keys)
    if missing:
        warnings.append("one or more expected federated collections did not return a response")
    payload = {
        "request_fingerprint": request.fingerprint,
        "hits": [
            {
                "global_evidence_id": item.global_evidence_id,
                "content_sha256": item.handle.content_sha256,
                "provider_id": item.handle.provider_id,
                "collection_id": item.handle.collection_id,
                "rank": item.provider_rank,
                "fused_score": item.fused_score,
                "provider_weight": item.provider_weight,
            }
            for item in deduped
        ],
        "participating": participating,
        "missing": missing,
        "warnings": tuple(dict.fromkeys(warnings)),
    }
    return FederatedMergeResult(
        request_fingerprint=request.fingerprint,
        hits=tuple(deduped),
        participating_collections=participating,
        failed_or_missing_collections=missing,
        warnings=tuple(dict.fromkeys(warnings)),
        fingerprint=hashlib.sha256(_canonical(payload)).hexdigest(),
    )


def verify_resolved_content(handle: FederatedEvidenceHandle, content: bytes) -> bool:
    if not isinstance(handle, FederatedEvidenceHandle) or not isinstance(content, bytes):
        return False
    return hashlib.sha256(content).hexdigest() == handle.content_sha256


__all__ = [
    "AuthorizedEvidenceResolver",
    "EvidenceAuthorization",
    "FederatedCollection",
    "FederatedEvidenceHandle",
    "FederatedMergeResult",
    "FederatedMergedHit",
    "FederatedSearchProvider",
    "FederatedSearchRequest",
    "FederatedSearchResponse",
    "merge_federated_responses",
    "verify_resolved_content",
]
