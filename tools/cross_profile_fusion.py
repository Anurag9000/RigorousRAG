"""Governed fusion across heterogeneous retriever/model score spaces.

Raw BM25, sparse, dense, late-interaction and reranker scores are not numerically
comparable. This module makes that boundary explicit:

* fit a monotone held-out relevance calibrator independently for each retriever profile;
* bind calibrators to immutable profile and calibration-contract identities;
* fuse compatible calibrated probabilities with one contribution per profile; or
* fall back to rank-only weighted RRF when calibration is unavailable.

No raw scores from different profiles are ever averaged or otherwise combined directly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from evaluation.calibration import CalibrationExample, brier_score, expected_calibration_error
from tools.corpus_fusion import FusedCandidate, FusionPolicy, RetrievalCandidate, reciprocal_rank_fuse

_MAX_EXAMPLES = 10_000_000
_MAX_PROFILES = 10_000
_EPSILON = 1e-9


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid.")
    return selected


def _sha256(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite.")
    return selected


def _positive(value: Any, label: str) -> float:
    selected = _finite(value, label)
    if selected <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return selected


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ScoreDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class CrossProfileFusionMode(str, Enum):
    AUTO = "auto"
    CALIBRATED_LOGIT = "calibrated_logit"
    RRF_ONLY = "rrf_only"


@dataclass(frozen=True)
class CalibrationContract:
    """Immutable definition of what calibrated probability means."""

    dataset_manifest_sha256: str
    split_sha256: str
    relevance_contract_sha256: str
    candidate_universe_sha256: str
    domain_id: str
    cohort_id: str = "default"

    def __post_init__(self) -> None:
        for name in ("dataset_manifest_sha256", "split_sha256", "relevance_contract_sha256", "candidate_universe_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "domain_id", _identifier(self.domain_id, "domain_id"))
        object.__setattr__(self, "cohort_id", _identifier(self.cohort_id, "cohort_id"))

    @property
    def contract_sha256(self) -> str:
        return _canonical_digest({"schema": "rigorousrag-cross-profile-calibration-contract/v1", **asdict(self)})


@dataclass(frozen=True)
class RetrieverScoreProfile:
    profile_id: str
    family: str
    scoring_contract_sha256: str
    model_profile_sha256: str
    score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_BETTER

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _identifier(self.profile_id, "profile_id"))
        object.__setattr__(self, "family", _identifier(self.family, "family"))
        object.__setattr__(self, "scoring_contract_sha256", _sha256(self.scoring_contract_sha256, "scoring_contract_sha256"))
        object.__setattr__(self, "model_profile_sha256", _sha256(self.model_profile_sha256, "model_profile_sha256"))
        if not isinstance(self.score_direction, ScoreDirection):
            object.__setattr__(self, "score_direction", ScoreDirection(self.score_direction))

    @property
    def profile_sha256(self) -> str:
        return _canonical_digest({
            "schema": "rigorousrag-retriever-score-profile/v1",
            "profile_id": self.profile_id,
            "family": self.family,
            "scoring_contract_sha256": self.scoring_contract_sha256,
            "model_profile_sha256": self.model_profile_sha256,
            "score_direction": self.score_direction.value,
        })


@dataclass(frozen=True)
class ScoreCalibrationExample:
    raw_score: float
    relevant: bool
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_score", _finite(self.raw_score, "raw_score"))
        if not isinstance(self.relevant, bool):
            raise ValueError("relevant must be boolean.")
        object.__setattr__(self, "weight", _positive(self.weight, "weight"))


@dataclass(frozen=True)
class IsotonicBin:
    """A monotone probability block; ``upper_oriented_score=None`` means +infinity."""

    upper_oriented_score: float | None
    probability: float
    total_weight: float
    positive_weight: float

    def __post_init__(self) -> None:
        if self.upper_oriented_score is not None:
            object.__setattr__(self, "upper_oriented_score", _finite(self.upper_oriented_score, "upper_oriented_score"))
        probability = _finite(self.probability, "probability")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1].")
        object.__setattr__(self, "probability", probability)
        total = _positive(self.total_weight, "total_weight")
        positive = _finite(self.positive_weight, "positive_weight")
        if not 0.0 <= positive <= total:
            raise ValueError("positive_weight must be in [0, total_weight].")
        object.__setattr__(self, "total_weight", total)
        object.__setattr__(self, "positive_weight", positive)


@dataclass(frozen=True)
class IsotonicCalibrationArtifact:
    profile: RetrieverScoreProfile
    calibration_contract_sha256: str
    examples_sha256: str
    example_count: int
    bins: tuple[IsotonicBin, ...]
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile, RetrieverScoreProfile):
            raise ValueError("profile must be RetrieverScoreProfile.")
        object.__setattr__(self, "calibration_contract_sha256", _sha256(self.calibration_contract_sha256, "calibration_contract_sha256"))
        object.__setattr__(self, "examples_sha256", _sha256(self.examples_sha256, "examples_sha256"))
        if isinstance(self.example_count, bool) or not isinstance(self.example_count, int) or not 1 <= self.example_count <= _MAX_EXAMPLES:
            raise ValueError("example_count is invalid.")
        bins = tuple(self.bins)
        if not bins:
            raise ValueError("at least one isotonic bin is required.")
        previous_probability = -1.0
        previous_upper: float | None = None
        terminal = False
        for index, item in enumerate(bins):
            if not isinstance(item, IsotonicBin):
                raise ValueError("bins must contain IsotonicBin values.")
            if item.probability + 1e-15 < previous_probability:
                raise ValueError("isotonic probabilities must be non-decreasing.")
            previous_probability = item.probability
            if item.upper_oriented_score is None:
                if index != len(bins) - 1:
                    raise ValueError("only the terminal isotonic bin may be unbounded.")
                terminal = True
            else:
                if previous_upper is not None and item.upper_oriented_score <= previous_upper:
                    raise ValueError("isotonic boundaries must be strictly increasing.")
                previous_upper = item.upper_oriented_score
        if not terminal:
            raise ValueError("terminal isotonic bin must be unbounded.")
        object.__setattr__(self, "bins", bins)
        expected = _canonical_digest(self._digest_payload())
        provided = _sha256(self.artifact_sha256, "artifact_sha256")
        if provided != expected:
            raise ValueError("artifact_sha256 does not match calibration artifact content.")
        object.__setattr__(self, "artifact_sha256", provided)

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-isotonic-score-calibration/v1",
            "profile": {
                "profile_id": self.profile.profile_id,
                "family": self.profile.family,
                "scoring_contract_sha256": self.profile.scoring_contract_sha256,
                "model_profile_sha256": self.profile.model_profile_sha256,
                "score_direction": self.profile.score_direction.value,
                "profile_sha256": self.profile.profile_sha256,
            },
            "calibration_contract_sha256": self.calibration_contract_sha256,
            "examples_sha256": self.examples_sha256,
            "example_count": self.example_count,
            "bins": [asdict(item) for item in self.bins],
        }

    @classmethod
    def build(cls, *, profile: RetrieverScoreProfile, calibration_contract_sha256: str, examples_sha256: str, example_count: int, bins: Sequence[IsotonicBin]) -> "IsotonicCalibrationArtifact":
        draft = {
            "schema": "rigorousrag-isotonic-score-calibration/v1",
            "profile": {
                "profile_id": profile.profile_id,
                "family": profile.family,
                "scoring_contract_sha256": profile.scoring_contract_sha256,
                "model_profile_sha256": profile.model_profile_sha256,
                "score_direction": profile.score_direction.value,
                "profile_sha256": profile.profile_sha256,
            },
            "calibration_contract_sha256": _sha256(calibration_contract_sha256, "calibration_contract_sha256"),
            "examples_sha256": _sha256(examples_sha256, "examples_sha256"),
            "example_count": example_count,
            "bins": [asdict(item) for item in bins],
        }
        return cls(profile=profile, calibration_contract_sha256=draft["calibration_contract_sha256"], examples_sha256=draft["examples_sha256"], example_count=example_count, bins=tuple(bins), artifact_sha256=_canonical_digest(draft))

    def predict(self, raw_score: float) -> float:
        score = _finite(raw_score, "raw_score")
        oriented = score if self.profile.score_direction is ScoreDirection.HIGHER_IS_BETTER else -score
        for item in self.bins:
            if item.upper_oriented_score is None or oriented <= item.upper_oriented_score:
                return item.probability
        raise AssertionError("terminal isotonic bin must cover all scores.")


def _calibration_examples_digest(examples: Sequence[ScoreCalibrationExample]) -> str:
    rows = sorted((item.raw_score, int(item.relevant), item.weight) for item in examples)
    return _canonical_digest({"schema": "rigorousrag-score-calibration-examples/v1", "examples": rows})


def fit_isotonic_calibrator(*, profile: RetrieverScoreProfile, contract: CalibrationContract, examples: Iterable[ScoreCalibrationExample]) -> IsotonicCalibrationArtifact:
    """Fit dependency-free weighted PAV isotonic relevance calibration."""

    values = tuple(examples)
    if not values or len(values) > _MAX_EXAMPLES:
        raise ValueError("examples must be a non-empty bounded collection.")
    if any(not isinstance(item, ScoreCalibrationExample) for item in values):
        raise ValueError("examples must contain ScoreCalibrationExample values.")

    sign = 1.0 if profile.score_direction is ScoreDirection.HIGHER_IS_BETTER else -1.0
    grouped: list[dict[str, float]] = []
    for item in sorted(values, key=lambda row: (sign * row.raw_score, row.raw_score)):
        x = sign * item.raw_score
        if grouped and grouped[-1]["x"] == x:
            grouped[-1]["weight"] += item.weight
            grouped[-1]["positive"] += item.weight * float(item.relevant)
            continue
        grouped.append({"x": x, "min_x": x, "max_x": x, "weight": item.weight, "positive": item.weight * float(item.relevant)})

    blocks: list[dict[str, float]] = []
    for group in grouped:
        blocks.append(dict(group))
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left["positive"] / left["weight"] <= right["positive"] / right["weight"] + 1e-15:
                break
            blocks[-2:] = [{
                "x": left["x"],
                "min_x": left["min_x"],
                "max_x": right["max_x"],
                "weight": left["weight"] + right["weight"],
                "positive": left["positive"] + right["positive"],
            }]

    bins: list[IsotonicBin] = []
    for index, block in enumerate(blocks):
        upper = None if index == len(blocks) - 1 else (block["max_x"] + blocks[index + 1]["min_x"]) / 2.0
        bins.append(IsotonicBin(upper_oriented_score=upper, probability=block["positive"] / block["weight"], total_weight=block["weight"], positive_weight=block["positive"]))

    return IsotonicCalibrationArtifact.build(profile=profile, calibration_contract_sha256=contract.contract_sha256, examples_sha256=_calibration_examples_digest(values), example_count=len(values), bins=bins)


@dataclass(frozen=True)
class CalibrationDiagnostics:
    brier: float
    ece: float
    example_count: int

    def __post_init__(self) -> None:
        for name in ("brier", "ece"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
            object.__setattr__(self, name, value)
        if isinstance(self.example_count, bool) or not isinstance(self.example_count, int) or self.example_count < 1:
            raise ValueError("example_count must be positive.")


def evaluate_isotonic_calibrator(artifact: IsotonicCalibrationArtifact, examples: Iterable[ScoreCalibrationExample], *, bin_count: int = 10) -> CalibrationDiagnostics:
    values = tuple(examples)
    if not values:
        raise ValueError("at least one evaluation example is required.")
    calibrated = tuple(CalibrationExample(confidence=artifact.predict(item.raw_score), correct=item.relevant, weight=item.weight) for item in values)
    return CalibrationDiagnostics(brier=brier_score(calibrated), ece=expected_calibration_error(calibrated, bin_count=bin_count), example_count=len(calibrated))


@dataclass(frozen=True)
class ProfileRankedList:
    list_id: str
    profile: RetrieverScoreProfile
    candidates: tuple[RetrievalCandidate, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "list_id", _identifier(self.list_id, "list_id"))
        if not isinstance(self.profile, RetrieverScoreProfile):
            raise ValueError("profile must be RetrieverScoreProfile.")
        candidates = tuple(self.candidates)
        if len(candidates) > _MAX_EXAMPLES:
            raise ValueError("candidate list is too large.")
        seen_ids: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, RetrievalCandidate):
                raise ValueError("candidates must contain RetrievalCandidate values.")
            if candidate.retriever_id != self.profile.profile_id:
                raise ValueError("candidate retriever_id must equal the bound score profile id.")
            if candidate.candidate_id in seen_ids:
                raise ValueError("candidate_id must be unique within a profile list.")
            seen_ids.add(candidate.candidate_id)
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True)
class CalibratedContribution:
    profile_id: str
    list_id: str
    rank: int
    raw_score: float
    calibrated_probability: float
    weight: float


@dataclass(frozen=True)
class CrossProfileCandidate:
    candidate: RetrievalCandidate
    fused_probability: float | None
    fused_score: float
    best_rank: int
    calibrated_contributions: tuple[CalibratedContribution, ...] = ()
    rrf: FusedCandidate | None = None


@dataclass(frozen=True)
class CrossProfileFusionPolicy:
    mode: CrossProfileFusionMode = CrossProfileFusionMode.AUTO
    profile_weights: Mapping[str, float] = field(default_factory=dict)
    max_fused_candidates: int = 1000
    max_per_document: int = 3
    max_per_source: int | None = None
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CrossProfileFusionMode):
            object.__setattr__(self, "mode", CrossProfileFusionMode(self.mode))
        cleaned: dict[str, float] = {}
        if not isinstance(self.profile_weights, Mapping) or len(self.profile_weights) > _MAX_PROFILES:
            raise ValueError("profile_weights must be a bounded mapping.")
        for key, raw in self.profile_weights.items():
            value = _finite(raw, "profile weight")
            if value < 0.0:
                raise ValueError("profile weights must be non-negative.")
            cleaned[_identifier(key, "profile weight key")] = value
        object.__setattr__(self, "profile_weights", cleaned)
        for name in ("max_fused_candidates", "max_per_document", "rrf_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be positive.")
        if self.max_per_source is not None and (isinstance(self.max_per_source, bool) or not isinstance(self.max_per_source, int) or self.max_per_source < 1):
            raise ValueError("max_per_source must be positive when set.")

    def weight(self, profile_id: str) -> float:
        return self.profile_weights.get(profile_id, 1.0)


@dataclass(frozen=True)
class CrossProfileFusionResult:
    mode: CrossProfileFusionMode
    candidates: tuple[CrossProfileCandidate, ...]
    calibration_contract_sha256: str | None
    profile_artifact_sha256s: tuple[tuple[str, str], ...]
    result_sha256: str


def _logit(probability: float) -> float:
    clipped = min(max(probability, _EPSILON), 1.0 - _EPSILON)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _apply_caps(candidates: Sequence[CrossProfileCandidate], *, policy: CrossProfileFusionPolicy) -> tuple[CrossProfileCandidate, ...]:
    document_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    selected: list[CrossProfileCandidate] = []
    for item in candidates:
        document_id = item.candidate.document_id
        if document_counts.get(document_id, 0) >= policy.max_per_document:
            continue
        source_id = item.candidate.source_id
        if policy.max_per_source is not None and source_id is not None and source_counts.get(source_id, 0) >= policy.max_per_source:
            continue
        selected.append(item)
        document_counts[document_id] = document_counts.get(document_id, 0) + 1
        if source_id is not None:
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
        if len(selected) >= policy.max_fused_candidates:
            break
    return tuple(selected)


def _rrf_result(lists: Sequence[ProfileRankedList], policy: CrossProfileFusionPolicy) -> CrossProfileFusionResult:
    ranked_lists = {item.list_id: item.candidates for item in lists}
    rrf_policy = FusionPolicy(
        rrf_k=policy.rrf_k,
        max_per_input_list=max((len(item.candidates) for item in lists), default=1),
        max_fused_candidates=policy.max_fused_candidates,
        max_per_document=policy.max_per_document,
        max_per_source=policy.max_per_source,
        retriever_weights={item.profile.profile_id: policy.weight(item.profile.profile_id) for item in lists},
    )
    fused = reciprocal_rank_fuse(ranked_lists, policy=rrf_policy)
    candidates = tuple(CrossProfileCandidate(candidate=item.candidate, fused_probability=None, fused_score=item.fused_score, best_rank=item.best_rank, rrf=item) for item in fused)
    payload = {
        "schema": "rigorousrag-cross-profile-fusion-result/v1",
        "mode": CrossProfileFusionMode.RRF_ONLY.value,
        "calibration_contract_sha256": None,
        "profiles": sorted({item.profile.profile_sha256 for item in lists}),
        "candidates": [{"document_id": item.candidate.document_id, "chunk_id": item.candidate.chunk_id, "candidate_id": item.candidate.candidate_id, "fused_score": item.fused_score, "best_rank": item.best_rank} for item in candidates],
    }
    return CrossProfileFusionResult(CrossProfileFusionMode.RRF_ONLY, candidates, None, (), _canonical_digest(payload))


def _calibrated_result(lists: Sequence[ProfileRankedList], calibrators: Mapping[str, IsotonicCalibrationArtifact], policy: CrossProfileFusionPolicy) -> CrossProfileFusionResult:
    contracts = {calibrators[item.profile.profile_id].calibration_contract_sha256 for item in lists}
    if len(contracts) != 1:
        raise ValueError("calibrated fusion requires one shared calibration contract.")
    contract_sha256 = next(iter(contracts))
    states: dict[tuple[str, str], dict[str, Any]] = {}
    for profile_list in sorted(lists, key=lambda item: item.list_id):
        profile_id = profile_list.profile.profile_id
        calibrator = calibrators[profile_id]
        if calibrator.profile.profile_sha256 != profile_list.profile.profile_sha256:
            raise ValueError("calibrator profile identity does not match ranked-list profile.")
        weight = policy.weight(profile_id)
        if weight <= 0.0:
            continue
        for candidate in sorted(profile_list.candidates, key=lambda item: (item.rank, item.candidate_id)):
            if candidate.raw_score is None:
                raise ValueError("calibrated fusion requires a raw score for every participating candidate.")
            identity = (candidate.document_id, candidate.chunk_id)
            probability = calibrator.predict(candidate.raw_score)
            contribution = CalibratedContribution(profile_id, profile_list.list_id, candidate.rank, candidate.raw_score, probability, weight)
            state = states.setdefault(identity, {"candidate": candidate, "best_rank": candidate.rank, "by_profile": {}})
            current = state["candidate"]
            if (candidate.rank, candidate.candidate_id) < (state["best_rank"], current.candidate_id):
                state["candidate"] = candidate
            state["best_rank"] = min(state["best_rank"], candidate.rank)
            previous = state["by_profile"].get(profile_id)
            if previous is None or (contribution.calibrated_probability, -contribution.rank, contribution.list_id) > (previous.calibrated_probability, -previous.rank, previous.list_id):
                state["by_profile"][profile_id] = contribution

    fused: list[CrossProfileCandidate] = []
    for state in states.values():
        contributions = tuple(sorted(state["by_profile"].values(), key=lambda item: (item.profile_id, item.list_id, item.rank)))
        if not contributions:
            continue
        total_weight = sum(item.weight for item in contributions)
        probability = _sigmoid(sum(item.weight * _logit(item.calibrated_probability) for item in contributions) / total_weight)
        fused.append(CrossProfileCandidate(candidate=state["candidate"], fused_probability=probability, fused_score=probability, best_rank=state["best_rank"], calibrated_contributions=contributions))
    fused.sort(key=lambda item: (-item.fused_score, item.best_rank, item.candidate.document_id, item.candidate.chunk_id, item.candidate.candidate_id))
    selected = _apply_caps(fused, policy=policy)
    artifacts = tuple(sorted({(item.profile.profile_id, calibrators[item.profile.profile_id].artifact_sha256) for item in lists}))
    payload = {
        "schema": "rigorousrag-cross-profile-fusion-result/v1",
        "mode": CrossProfileFusionMode.CALIBRATED_LOGIT.value,
        "calibration_contract_sha256": contract_sha256,
        "profile_artifact_sha256s": artifacts,
        "candidates": [{"document_id": item.candidate.document_id, "chunk_id": item.candidate.chunk_id, "candidate_id": item.candidate.candidate_id, "fused_probability": item.fused_probability, "best_rank": item.best_rank, "contributions": [asdict(value) for value in item.calibrated_contributions]} for item in selected],
    }
    return CrossProfileFusionResult(CrossProfileFusionMode.CALIBRATED_LOGIT, selected, contract_sha256, artifacts, _canonical_digest(payload))


def fuse_cross_profile_rankings(ranked_lists: Iterable[ProfileRankedList], *, calibrators: Mapping[str, IsotonicCalibrationArtifact] | None = None, policy: CrossProfileFusionPolicy = CrossProfileFusionPolicy()) -> CrossProfileFusionResult:
    """Fuse heterogeneous score profiles without ever combining incomparable raw scores."""

    lists = tuple(ranked_lists)
    if not lists or len(lists) > _MAX_PROFILES:
        raise ValueError("ranked_lists must be a non-empty bounded collection.")
    if any(not isinstance(item, ProfileRankedList) for item in lists):
        raise ValueError("ranked_lists must contain ProfileRankedList values.")
    if len({item.list_id for item in lists}) != len(lists):
        raise ValueError("list_id values must be unique.")
    profiles: dict[str, RetrieverScoreProfile] = {}
    for item in lists:
        previous = profiles.get(item.profile.profile_id)
        if previous is not None and previous.profile_sha256 != item.profile.profile_sha256:
            raise ValueError("profile_id maps to inconsistent score-profile identities.")
        profiles[item.profile.profile_id] = item.profile

    supplied = {} if calibrators is None else dict(calibrators)
    for profile_id, artifact in supplied.items():
        _identifier(profile_id, "calibrator profile id")
        if not isinstance(artifact, IsotonicCalibrationArtifact):
            raise ValueError("calibrators must contain IsotonicCalibrationArtifact values.")
        if artifact.profile.profile_id != profile_id:
            raise ValueError("calibrator mapping key does not match artifact profile_id.")

    required_profile_ids = set(profiles)
    fully_calibrated = required_profile_ids.issubset(supplied)
    compatible = False
    if fully_calibrated:
        compatible = len({supplied[profile_id].calibration_contract_sha256 for profile_id in required_profile_ids}) == 1
        for profile_id, profile in profiles.items():
            if supplied[profile_id].profile.profile_sha256 != profile.profile_sha256:
                compatible = False
                break

    if policy.mode is CrossProfileFusionMode.RRF_ONLY:
        return _rrf_result(lists, policy)
    if policy.mode is CrossProfileFusionMode.CALIBRATED_LOGIT:
        if not fully_calibrated:
            raise ValueError("strict calibrated fusion requires every profile calibrator.")
        if not compatible:
            raise ValueError("strict calibrated fusion requires compatible calibrators.")
        return _calibrated_result(lists, supplied, policy)
    if fully_calibrated and compatible:
        return _calibrated_result(lists, supplied, policy)
    return _rrf_result(lists, policy)


__all__ = [
    "CalibrationContract", "CalibrationDiagnostics", "CalibratedContribution",
    "CrossProfileCandidate", "CrossProfileFusionMode", "CrossProfileFusionPolicy",
    "CrossProfileFusionResult", "IsotonicBin", "IsotonicCalibrationArtifact",
    "ProfileRankedList", "RetrieverScoreProfile", "ScoreCalibrationExample",
    "ScoreDirection", "evaluate_isotonic_calibrator", "fit_isotonic_calibrator",
    "fuse_cross_profile_rankings",
]
