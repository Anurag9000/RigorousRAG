"""Query-grouped ListNet training for calibrated heterogeneous-retriever fusion."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

_EPS = 1e-9


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return selected


def _git(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("source_revision must be hexadecimal")
    selected = value.strip().lower()
    if len(selected) not in (40, 64) or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError("source_revision must be a 40- or 64-character hexadecimal Git object id")
    return selected


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 500:
        raise ValueError(f"{label} is invalid")
    return value.strip()


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _positive(value: float, label: str, *, allow_zero: bool = False) -> float:
    selected = _finite(value, label)
    if selected < 0.0 or (not allow_zero and selected == 0.0):
        raise ValueError(f"{label} is invalid")
    return selected


def _probability(value: float, label: str) -> float:
    selected = _finite(value, label)
    if not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return selected


def _logit(value: float) -> float:
    p = min(max(value, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _softmax(values: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    exp_values = [math.exp(value - maximum) for value in scaled]
    total = sum(exp_values)
    return tuple(value / total for value in exp_values)


@dataclass(frozen=True)
class FusionRankingCandidate:
    candidate_id: str
    probabilities: Mapping[str, float]
    relevance_grade: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _identifier(self.candidate_id, "candidate_id"))
        if not isinstance(self.probabilities, Mapping) or not self.probabilities:
            raise ValueError("probabilities must be non-empty")
        object.__setattr__(self, "probabilities", {_identifier(key, "profile id"): _probability(value, "profile probability") for key, value in self.probabilities.items()})
        grade = _finite(self.relevance_grade, "relevance_grade")
        if grade < 0.0:
            raise ValueError("relevance_grade must be non-negative")
        object.__setattr__(self, "relevance_grade", grade)


@dataclass(frozen=True)
class FusionRankingQuery:
    query_sha256: str
    candidates: tuple[FusionRankingCandidate, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_sha256", _sha(self.query_sha256, "query_sha256"))
        values = tuple(self.candidates)
        if len(values) < 2 or len(values) > 100_000:
            raise ValueError("each ranking query requires 2-100000 candidates")
        if any(not isinstance(item, FusionRankingCandidate) for item in values):
            raise ValueError("candidates must contain FusionRankingCandidate values")
        if len({item.candidate_id for item in values}) != len(values):
            raise ValueError("candidate ids must be unique within a query")
        if len({item.relevance_grade for item in values}) < 2:
            raise ValueError("ranking query must contain at least two relevance grades")
        object.__setattr__(self, "candidates", values)
        object.__setattr__(self, "weight", _positive(self.weight, "query weight"))


@dataclass(frozen=True)
class ListwiseFusionTrainingConfig:
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 0.05
    l2: float = 1e-4
    gradient_clip_norm: float = 5.0
    target_temperature: float = 1.0
    prediction_temperature: float = 1.0
    patience: int = 10
    min_delta: float = 1e-5
    seed: int = 29

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in ("learning_rate", "gradient_clip_norm", "target_temperature", "prediction_temperature"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(self, "l2", _positive(self.l2, "l2", allow_zero=True))
        object.__setattr__(self, "min_delta", _positive(self.min_delta, "min_delta", allow_zero=True))

    @property
    def config_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-listwise-fusion-config/v1", **asdict(self)})


@dataclass(frozen=True)
class ListwiseFusionTrainingSpec:
    profile_ids: tuple[str, ...]
    calibration_contract_sha256: str
    calibration_artifact_sha256s: tuple[tuple[str, str], ...]
    train_split_sha256: str
    validation_split_sha256: str
    source_revision: str
    config: ListwiseFusionTrainingConfig

    def __post_init__(self) -> None:
        profiles = tuple(sorted(_identifier(value, "profile id") for value in self.profile_ids))
        if not profiles or len(set(profiles)) != len(profiles):
            raise ValueError("profile ids must be unique and non-empty")
        object.__setattr__(self, "profile_ids", profiles)
        object.__setattr__(self, "calibration_contract_sha256", _sha(self.calibration_contract_sha256, "calibration_contract_sha256"))
        object.__setattr__(self, "train_split_sha256", _sha(self.train_split_sha256, "train_split_sha256"))
        object.__setattr__(self, "validation_split_sha256", _sha(self.validation_split_sha256, "validation_split_sha256"))
        object.__setattr__(self, "source_revision", _git(self.source_revision))
        artifacts = tuple(sorted((_identifier(profile, "artifact profile id"), _sha(digest, "artifact sha256")) for profile, digest in self.calibration_artifact_sha256s))
        if tuple(profile for profile, _ in artifacts) != profiles:
            raise ValueError("calibration artifacts must exactly cover profiles")
        object.__setattr__(self, "calibration_artifact_sha256s", artifacts)
        if not isinstance(self.config, ListwiseFusionTrainingConfig):
            raise ValueError("config must be ListwiseFusionTrainingConfig")

    @property
    def spec_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-listwise-fusion-spec/v1",
            "profile_ids": self.profile_ids,
            "calibration_contract_sha256": self.calibration_contract_sha256,
            "calibration_artifact_sha256s": self.calibration_artifact_sha256s,
            "train_split_sha256": self.train_split_sha256,
            "validation_split_sha256": self.validation_split_sha256,
            "source_revision": self.source_revision,
            "config_sha256": self.config.config_sha256,
        })


def ranking_queries_sha256(queries: Sequence[FusionRankingQuery], profile_ids: Sequence[str]) -> str:
    profiles = tuple(profile_ids)
    payload = []
    for query in queries:
        if any(set(candidate.probabilities) != set(profiles) for candidate in query.candidates):
            raise ValueError("every ranking candidate must exactly cover training profiles")
        payload.append({
            "query_sha256": query.query_sha256,
            "weight": query.weight,
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "probabilities": tuple((profile, candidate.probabilities[profile]) for profile in profiles),
                    "relevance_grade": candidate.relevance_grade,
                }
                for candidate in query.candidates
            ],
        })
    return _digest({"schema": "rigorousrag-listwise-fusion-queries/v1", "profiles": profiles, "queries": payload})


@dataclass(frozen=True)
class ListwiseFusionTrainingState:
    spec_sha256: str
    train_queries_sha256: str
    validation_queries_sha256: str
    epoch: int
    batch_index: int
    theta: tuple[float, ...]
    best_theta: tuple[float, ...]
    best_validation_loss: float | None
    best_epoch: int | None
    stale_epochs: int
    completed: bool

    @property
    def state_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-listwise-fusion-state/v1", **asdict(self)})


@dataclass(frozen=True)
class LearnedListwiseFusionArtifact:
    spec_sha256: str
    profile_weights: tuple[tuple[str, float], ...]
    calibration_contract_sha256: str
    calibration_artifact_sha256s: tuple[tuple[str, str], ...]
    train_queries_sha256: str
    validation_queries_sha256: str
    best_epoch: int
    validation_listnet_loss: float
    artifact_sha256: str

    def __post_init__(self) -> None:
        for name in ("spec_sha256", "calibration_contract_sha256", "train_queries_sha256", "validation_queries_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        weights = tuple(sorted((profile, float(weight)) for profile, weight in self.profile_weights))
        if not weights or any(not math.isfinite(weight) or weight < 0.0 for _, weight in weights) or abs(sum(weight for _, weight in weights) - 1.0) > 1e-9:
            raise ValueError("profile weights must be finite, non-negative and sum to one")
        object.__setattr__(self, "profile_weights", weights)
        artifacts = tuple(sorted((profile, _sha(digest, "artifact sha256")) for profile, digest in self.calibration_artifact_sha256s))
        if tuple(profile for profile, _ in artifacts) != tuple(profile for profile, _ in weights):
            raise ValueError("calibration artifacts must cover learned profiles")
        object.__setattr__(self, "calibration_artifact_sha256s", artifacts)
        if isinstance(self.best_epoch, bool) or not isinstance(self.best_epoch, int) or self.best_epoch < 0:
            raise ValueError("best_epoch must be non-negative")
        loss = _positive(self.validation_listnet_loss, "validation_listnet_loss", allow_zero=True)
        object.__setattr__(self, "validation_listnet_loss", loss)
        expected = _digest(self._payload())
        provided = _sha(self.artifact_sha256, "artifact_sha256")
        if expected != provided:
            raise ValueError("artifact_sha256 does not match learned listwise artifact")
        object.__setattr__(self, "artifact_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-learned-listwise-fusion/v1",
            "spec_sha256": self.spec_sha256,
            "profile_weights": self.profile_weights,
            "calibration_contract_sha256": self.calibration_contract_sha256,
            "calibration_artifact_sha256s": self.calibration_artifact_sha256s,
            "train_queries_sha256": self.train_queries_sha256,
            "validation_queries_sha256": self.validation_queries_sha256,
            "best_epoch": self.best_epoch,
            "validation_listnet_loss": self.validation_listnet_loss,
        }

    @classmethod
    def build(cls, spec: ListwiseFusionTrainingSpec, state: ListwiseFusionTrainingState) -> "LearnedListwiseFusionArtifact":
        if state.best_validation_loss is None or state.best_epoch is None:
            raise ValueError("training state has no validated best checkpoint")
        weights = tuple(zip(spec.profile_ids, _softmax(state.best_theta)))
        payload = {
            "schema": "rigorousrag-learned-listwise-fusion/v1",
            "spec_sha256": spec.spec_sha256,
            "profile_weights": weights,
            "calibration_contract_sha256": spec.calibration_contract_sha256,
            "calibration_artifact_sha256s": spec.calibration_artifact_sha256s,
            "train_queries_sha256": state.train_queries_sha256,
            "validation_queries_sha256": state.validation_queries_sha256,
            "best_epoch": state.best_epoch,
            "validation_listnet_loss": state.best_validation_loss,
        }
        constructor = dict(payload)
        constructor.pop("schema")
        return cls(**constructor, artifact_sha256=_digest(payload))

    def score(self, probabilities: Mapping[str, float]) -> float:
        if set(probabilities) != {profile for profile, _ in self.profile_weights}:
            raise ValueError("probabilities must exactly cover learned profiles")
        return sum(weight * _logit(_probability(probabilities[profile], profile)) for profile, weight in self.profile_weights)


def _query_loss_and_gradient(theta: Sequence[float], query: FusionRankingQuery, profiles: Sequence[str], config: ListwiseFusionTrainingConfig) -> tuple[float, list[float]]:
    weights = _softmax(theta)
    features = [[_logit(candidate.probabilities[profile]) for profile in profiles] for candidate in query.candidates]
    scores = [sum(weights[index] * row[index] for index in range(len(weights))) for row in features]
    predicted = _softmax(scores, config.prediction_temperature)
    target = _softmax([candidate.relevance_grade for candidate in query.candidates], config.target_temperature)
    loss = -sum(target[index] * math.log(max(predicted[index], _EPS)) for index in range(len(target)))
    gradient = [0.0 for _ in theta]
    for candidate_index, row in enumerate(features):
        dloss_dscore = (predicted[candidate_index] - target[candidate_index]) / config.prediction_temperature
        score = scores[candidate_index]
        for profile_index in range(len(theta)):
            gradient[profile_index] += dloss_dscore * weights[profile_index] * (row[profile_index] - score)
    return loss, gradient


def _validation_loss(theta: Sequence[float], queries: Sequence[FusionRankingQuery], profiles: Sequence[str], config: ListwiseFusionTrainingConfig) -> float:
    total_weight = sum(query.weight for query in queries)
    return sum(query.weight * _query_loss_and_gradient(theta, query, profiles, config)[0] for query in queries) / total_weight


def _order(count: int, seed: int, epoch: int) -> list[int]:
    mixed = int(hashlib.sha256(f"{seed}:{epoch}".encode("utf-8")).hexdigest()[:16], 16)
    values = list(range(count))
    random.Random(mixed).shuffle(values)
    return values


def initialize_listwise_training(spec: ListwiseFusionTrainingSpec, train_queries: Sequence[FusionRankingQuery], validation_queries: Sequence[FusionRankingQuery]) -> ListwiseFusionTrainingState:
    if not train_queries or not validation_queries:
        raise ValueError("train and validation query sets must be non-empty")
    train_digest = ranking_queries_sha256(train_queries, spec.profile_ids)
    validation_digest = ranking_queries_sha256(validation_queries, spec.profile_ids)
    theta = tuple(0.0 for _ in spec.profile_ids)
    return ListwiseFusionTrainingState(spec.spec_sha256, train_digest, validation_digest, 0, 0, theta, theta, None, None, 0, False)


def advance_listwise_training(spec: ListwiseFusionTrainingSpec, state: ListwiseFusionTrainingState, train_queries: Sequence[FusionRankingQuery], validation_queries: Sequence[FusionRankingQuery], *, max_batches: int | None = None) -> ListwiseFusionTrainingState:
    if state.spec_sha256 != spec.spec_sha256:
        raise ValueError("state does not belong to spec")
    if ranking_queries_sha256(train_queries, spec.profile_ids) != state.train_queries_sha256 or ranking_queries_sha256(validation_queries, spec.profile_ids) != state.validation_queries_sha256:
        raise ValueError("query data differs from resumable state identity")
    if state.completed:
        return state
    if max_batches is not None and (isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1):
        raise ValueError("max_batches must be positive when set")
    config = spec.config
    theta = list(state.theta)
    best_theta = state.best_theta
    best_loss = state.best_validation_loss
    best_epoch = state.best_epoch
    stale = state.stale_epochs
    epoch = state.epoch
    batch_index = state.batch_index
    consumed = 0
    while epoch < config.epochs:
        order = _order(len(train_queries), config.seed, epoch)
        batch_count = (len(order) + config.batch_size - 1) // config.batch_size
        while batch_index < batch_count:
            indices = order[batch_index * config.batch_size : (batch_index + 1) * config.batch_size]
            gradient = [0.0 for _ in theta]
            normalizer = 0.0
            for index in indices:
                query = train_queries[index]
                _, query_gradient = _query_loss_and_gradient(theta, query, spec.profile_ids, config)
                for position in range(len(theta)):
                    gradient[position] += query.weight * query_gradient[position]
                normalizer += query.weight
            gradient = [gradient[position] / normalizer + config.l2 * theta[position] for position in range(len(theta))]
            norm = math.sqrt(sum(value * value for value in gradient))
            if norm > config.gradient_clip_norm:
                scale = config.gradient_clip_norm / norm
                gradient = [value * scale for value in gradient]
            theta = [theta[position] - config.learning_rate * gradient[position] for position in range(len(theta))]
            mean_theta = sum(theta) / len(theta)
            theta = [value - mean_theta for value in theta]
            batch_index += 1
            consumed += 1
            if max_batches is not None and consumed >= max_batches:
                return ListwiseFusionTrainingState(spec.spec_sha256, state.train_queries_sha256, state.validation_queries_sha256, epoch, batch_index, tuple(theta), best_theta, best_loss, best_epoch, stale, False)
        validation_loss = _validation_loss(theta, validation_queries, spec.profile_ids, config)
        if best_loss is None or validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_theta = tuple(theta)
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        epoch += 1
        batch_index = 0
        if stale >= config.patience:
            return ListwiseFusionTrainingState(spec.spec_sha256, state.train_queries_sha256, state.validation_queries_sha256, epoch, 0, tuple(theta), best_theta, best_loss, best_epoch, stale, True)
    return ListwiseFusionTrainingState(spec.spec_sha256, state.train_queries_sha256, state.validation_queries_sha256, epoch, 0, tuple(theta), best_theta, best_loss, best_epoch, stale, True)


def fit_listwise_fusion_weights(spec: ListwiseFusionTrainingSpec, train_queries: Sequence[FusionRankingQuery], validation_queries: Sequence[FusionRankingQuery]) -> LearnedListwiseFusionArtifact:
    state = initialize_listwise_training(spec, train_queries, validation_queries)
    state = advance_listwise_training(spec, state, train_queries, validation_queries)
    return LearnedListwiseFusionArtifact.build(spec, state)


__all__ = [
    "FusionRankingCandidate",
    "FusionRankingQuery",
    "LearnedListwiseFusionArtifact",
    "ListwiseFusionTrainingConfig",
    "ListwiseFusionTrainingSpec",
    "ListwiseFusionTrainingState",
    "advance_listwise_training",
    "fit_listwise_fusion_weights",
    "initialize_listwise_training",
    "ranking_queries_sha256",
]
