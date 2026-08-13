"""Bounded calibrated retrieval across heterogeneous embedding profiles.

The single-profile corpus retriever remains responsible for generation validation and
owner isolation. This layer fans a query out only to explicitly configured physical
profile backends, calibrates each profile in its own score space, deduplicates by a
server-derived evidence identity, and fuses the surviving evidence without assuming
raw scores from different embedding models are comparable.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from tools.corpus_hybrid_retrieval import CorpusEvidence, retrieve_corpus_evidence
from tools.generation_store import GenerationStore
from tools.hybrid_retrieval import RetrievalCandidate, mmr_select
from tools.retrieval_architectures import ScoreCalibration, calibrate_score
from tools.security import normalize_owner_id
from tools.sparse_index import SparseIndex

_MAX_PROFILES = 8
_MAX_RESULTS = 50
_MAX_POOL_PER_PROFILE = 100
_MAX_QUERY = 20_000


def _fingerprint(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 fingerprint.")
    cleaned = value.strip().lower()
    if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
        raise ValueError(f"{label} must be a SHA-256 fingerprint.")
    return cleaned


def _positive(value: Any, label: str, maximum: float = 1000.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and positive.") from exc
    if not math.isfinite(parsed) or not 0.0 < parsed <= maximum:
        raise ValueError(f"{label} must be finite, positive and bounded.")
    return parsed


def _exact_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


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


def _evidence_key(value: CorpusEvidence) -> str:
    payload = {
        "contract": "rigorousrag-cross-profile-evidence-v1",
        "doc_id": value.doc_id,
        "generation_sequence": value.generation_sequence,
        "page_number": value.page_number,
        "section": value.section,
        "text_sha256": hashlib.sha256(value.text.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProfileCorpusBackend:
    profile_fingerprint: str
    rag: Any
    weight: float = 1.0
    calibration: ScoreCalibration = field(default_factory=ScoreCalibration)
    required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_fingerprint",
            _fingerprint(self.profile_fingerprint, "profile_fingerprint"),
        )
        if not callable(getattr(self.rag, "query", None)):
            raise ValueError("profile rag backend must expose query().")
        object.__setattr__(self, "weight", _positive(self.weight, "weight"))
        if not isinstance(self.calibration, ScoreCalibration):
            raise ValueError("calibration must be ScoreCalibration.")
        if not isinstance(self.required, bool):
            raise ValueError("required must be boolean.")


@dataclass(frozen=True)
class CrossProfileEvidence:
    evidence_key: str
    evidence: CorpusEvidence
    score: float
    profile_scores: Mapping[str, float]
    contributing_profiles: tuple[str, ...]
    sparse_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_key", _fingerprint(self.evidence_key, "evidence_key"))
        if not isinstance(self.evidence, CorpusEvidence):
            raise ValueError("evidence must be CorpusEvidence.")
        object.__setattr__(self, "score", _unit(self.score, "score"))
        if not isinstance(self.profile_scores, Mapping) or len(self.profile_scores) > _MAX_PROFILES:
            raise ValueError("profile_scores is invalid or exceeds the profile limit.")
        normalized: dict[str, float] = {}
        for key, value in self.profile_scores.items():
            fingerprint = _fingerprint(key, "profile score fingerprint")
            normalized[fingerprint] = _unit(value, "profile score")
        object.__setattr__(self, "profile_scores", normalized)
        profiles = tuple(_fingerprint(item, "contributing profile") for item in self.contributing_profiles)
        if profiles != tuple(sorted(set(profiles))):
            raise ValueError("contributing_profiles must be sorted and unique.")
        if set(profiles) != set(normalized):
            raise ValueError("contributing_profiles must match profile_scores.")
        object.__setattr__(self, "contributing_profiles", profiles)
        if self.sparse_score is not None:
            object.__setattr__(self, "sparse_score", _unit(self.sparse_score, "sparse_score"))


@dataclass
class _Aggregate:
    representative: CorpusEvidence
    weighted_sum: float = 0.0
    weight_sum: float = 0.0
    profile_scores: dict[str, float] = field(default_factory=dict)
    sparse_score: float | None = None
    strongest_component: float = -1.0

    def add_profile(self, profile: str, score: float, weight: float, evidence: CorpusEvidence) -> None:
        previous = self.profile_scores.get(profile)
        if previous is not None and previous >= score:
            return
        if previous is not None:
            self.weighted_sum -= previous * weight
            self.weight_sum -= weight
        self.profile_scores[profile] = score
        self.weighted_sum += score * weight
        self.weight_sum += weight
        if score > self.strongest_component:
            self.strongest_component = score
            self.representative = evidence

    def add_sparse(self, score: float, weight: float, evidence: CorpusEvidence) -> None:
        if self.sparse_score is not None and self.sparse_score >= score:
            return
        if self.sparse_score is not None:
            self.weighted_sum -= self.sparse_score * weight
            self.weight_sum -= weight
        self.sparse_score = score
        self.weighted_sum += score * weight
        self.weight_sum += weight
        if score > self.strongest_component:
            self.strongest_component = score
            self.representative = evidence

    @property
    def score(self) -> float:
        return 0.0 if self.weight_sum <= 0.0 else max(0.0, min(self.weighted_sum / self.weight_sum, 1.0))


def retrieve_cross_profile_evidence(
    query: str,
    *,
    owner_id: str,
    backends: Sequence[ProfileCorpusBackend],
    sparse: SparseIndex,
    generations: GenerationStore,
    doc_id: str | None = None,
    top_k: int = 5,
    per_profile_top_k: int = 20,
    include_sparse: bool = True,
    sparse_weight: float = 0.7,
    sparse_calibration: ScoreCalibration | None = None,
    diversity_lambda: float = 0.82,
) -> tuple[CrossProfileEvidence, ...]:
    """Fan out to explicit profiles, calibrate independently and fuse evidence.

    Optional profile backends that raise during retrieval are omitted. A backend marked
    ``required=True`` turns the same failure into a bounded RuntimeError. Sparse
    retrieval is executed once, not once per profile, so lexical evidence cannot gain
    accidental weight simply because more dense profiles were configured.
    """

    if not isinstance(query, str):
        raise ValueError("query must be a string.")
    bounded_query = query.strip()
    if not bounded_query or len(bounded_query) > _MAX_QUERY:
        raise ValueError("query is empty or exceeds the query limit.")
    owner = normalize_owner_id(owner_id)
    if isinstance(backends, (str, bytes, bytearray)):
        raise ValueError("backends must be a bounded sequence.")
    selected = tuple(backends)
    if not selected or len(selected) > _MAX_PROFILES or any(
        not isinstance(item, ProfileCorpusBackend) for item in selected
    ):
        raise ValueError("backends must contain between 1 and 8 profile backends.")
    fingerprints = [item.profile_fingerprint for item in selected]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("profile backends must have unique fingerprints.")
    requested = _exact_int(top_k, "top_k", 1, _MAX_RESULTS)
    per_profile = _exact_int(
        per_profile_top_k,
        "per_profile_top_k",
        1,
        _MAX_POOL_PER_PROFILE,
    )
    if not isinstance(include_sparse, bool):
        raise ValueError("include_sparse must be boolean.")
    sparse_w = _positive(sparse_weight, "sparse_weight")
    calibration = sparse_calibration or ScoreCalibration()
    if not isinstance(calibration, ScoreCalibration):
        raise ValueError("sparse_calibration must be ScoreCalibration.")
    diversity = _unit(diversity_lambda, "diversity_lambda")

    aggregates: dict[str, _Aggregate] = {}
    for backend in selected:
        try:
            rows = retrieve_corpus_evidence(
                bounded_query,
                owner_id=owner,
                rag=backend.rag,
                sparse=sparse,
                generations=generations,
                doc_id=doc_id,
                mode="dense",
                top_k=per_profile,
                dense_pool=min(_MAX_POOL_PER_PROFILE, max(per_profile * 3, per_profile)),
                sparse_pool=1,
                dense_weight=1.0,
                sparse_weight=0.0,
                diversity_lambda=1.0,
            )
        except Exception as exc:
            if backend.required:
                raise RuntimeError("required profile retrieval failed.") from exc
            continue
        for evidence in rows[:per_profile]:
            if evidence.profile_fingerprint != backend.profile_fingerprint:
                continue
            calibrated = calibrate_score(
                evidence.score,
                temperature=backend.calibration.temperature,
                bias=backend.calibration.bias,
            )
            key = _evidence_key(evidence)
            aggregate = aggregates.setdefault(key, _Aggregate(evidence))
            aggregate.add_profile(
                backend.profile_fingerprint,
                calibrated,
                backend.weight,
                evidence,
            )

    if include_sparse:
        try:
            sparse_rows = retrieve_corpus_evidence(
                bounded_query,
                owner_id=owner,
                rag=selected[0].rag,
                sparse=sparse,
                generations=generations,
                doc_id=doc_id,
                mode="sparse",
                top_k=min(per_profile, _MAX_RESULTS),
                dense_pool=1,
                sparse_pool=min(_MAX_POOL_PER_PROFILE, max(per_profile * 3, per_profile)),
                dense_weight=0.0,
                sparse_weight=1.0,
                diversity_lambda=1.0,
            )
        except Exception:
            sparse_rows = ()
        for evidence in sparse_rows[:per_profile]:
            calibrated = calibrate_score(
                evidence.score,
                temperature=calibration.temperature,
                bias=calibration.bias,
            )
            key = _evidence_key(evidence)
            aggregate = aggregates.setdefault(key, _Aggregate(evidence))
            aggregate.add_sparse(calibrated, sparse_w, evidence)

    candidates: list[tuple[RetrievalCandidate, float]] = []
    output: dict[str, CrossProfileEvidence] = {}
    for key, aggregate in aggregates.items():
        score = aggregate.score
        row = CrossProfileEvidence(
            evidence_key=key,
            evidence=aggregate.representative,
            score=score,
            profile_scores=dict(sorted(aggregate.profile_scores.items())),
            contributing_profiles=tuple(sorted(aggregate.profile_scores)),
            sparse_score=aggregate.sparse_score,
        )
        output[key] = row
        candidates.append(
            (
                RetrievalCandidate(
                    candidate_id=key,
                    text=row.evidence.text,
                    source_id=row.evidence.doc_id,
                    dense_score=score,
                ),
                score,
            )
        )
    if not candidates:
        return ()
    selected_rows = mmr_select(
        candidates,
        top_k=min(requested, len(candidates)),
        diversity_lambda=diversity,
        max_per_source=max(1, requested),
    )
    return tuple(output[item.candidate_id] for item, _score_value in selected_rows)


__all__ = [
    "CrossProfileEvidence",
    "ProfileCorpusBackend",
    "retrieve_cross_profile_evidence",
]
