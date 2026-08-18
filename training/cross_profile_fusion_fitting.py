"""Deterministic fitting of cross-profile fusion weights over calibrated probabilities.

The learner never consumes heterogeneous raw retrieval scores.  Each feature is a
held-out calibrated relevance probability from a governed retriever profile.  Softmax
parameters produce non-negative weights that sum to one; the weighted profile logits are
then mapped back to a fused probability and optimized with weighted binary log-loss.

Training state is explicitly resumable at mini-batch boundaries.  Epoch permutations are
derived from the immutable seed+epoch pair, so resumption does not depend on hidden RNG
state.  Final artifacts bind the calibration contract, per-profile calibration artifacts,
training/validation data digests, configuration and source revision.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

_EPS = 1e-9


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


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
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return selected


def _git_revision(value: Any) -> str:
    selected = _identifier(value, "source_revision", 64).lower()
    if len(selected) not in (40, 64) or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError("source_revision must be a 40- or 64-character hexadecimal Git object id.")
    return selected


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be in [0, 1].")
    selected = float(value)
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be in [0, 1].")
    return selected


def _positive(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is invalid.")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0.0 or (not allow_zero and selected == 0.0):
        raise ValueError(f"{label} is invalid.")
    return selected


def _logit(probability: float) -> float:
    p = min(max(probability, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def _clip_probability(value: float) -> float:
    return min(max(value, _EPS), 1.0 - _EPS)


@dataclass(frozen=True)
class FusionWeightExample:
    probabilities: Mapping[str, float]
    relevant: bool
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.probabilities, Mapping) or not self.probabilities:
            raise ValueError("probabilities must be a non-empty mapping.")
        cleaned: dict[str, float] = {}
        for key, value in self.probabilities.items():
            cleaned[_identifier(key, "profile id")] = _probability(value, "profile probability")
        object.__setattr__(self, "probabilities", cleaned)
        if not isinstance(self.relevant, bool):
            raise ValueError("relevant must be boolean.")
        object.__setattr__(self, "weight", _positive(self.weight, "weight"))


@dataclass(frozen=True)
class FusionWeightTrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.05
    l2: float = 1e-4
    gradient_clip_norm: float = 5.0
    positive_class_weight: float = 1.0
    patience: int = 10
    min_delta: float = 1e-5
    seed: int = 17

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        object.__setattr__(self, "learning_rate", _positive(self.learning_rate, "learning_rate"))
        object.__setattr__(self, "l2", _positive(self.l2, "l2", allow_zero=True))
        object.__setattr__(self, "gradient_clip_norm", _positive(self.gradient_clip_norm, "gradient_clip_norm"))
        object.__setattr__(self, "positive_class_weight", _positive(self.positive_class_weight, "positive_class_weight"))
        object.__setattr__(self, "min_delta", _positive(self.min_delta, "min_delta", allow_zero=True))

    @property
    def config_sha256(self) -> str:
        return _canonical_digest({"schema": "rigorousrag-fusion-weight-training-config/v1", **asdict(self)})


@dataclass(frozen=True)
class FusionWeightTrainingSpec:
    profile_ids: tuple[str, ...]
    calibration_contract_sha256: str
    calibration_artifact_sha256s: tuple[tuple[str, str], ...]
    train_split_sha256: str
    validation_split_sha256: str
    source_revision: str
    config: FusionWeightTrainingConfig

    def __post_init__(self) -> None:
        profiles = tuple(sorted(_identifier(value, "profile id") for value in self.profile_ids))
        if not profiles or len(set(profiles)) != len(profiles):
            raise ValueError("profile_ids must be unique and non-empty.")
        object.__setattr__(self, "profile_ids", profiles)
        object.__setattr__(self, "calibration_contract_sha256", _sha256(self.calibration_contract_sha256, "calibration_contract_sha256"))
        object.__setattr__(self, "train_split_sha256", _sha256(self.train_split_sha256, "train_split_sha256"))
        object.__setattr__(self, "validation_split_sha256", _sha256(self.validation_split_sha256, "validation_split_sha256"))
        object.__setattr__(self, "source_revision", _git_revision(self.source_revision))
        if not isinstance(self.config, FusionWeightTrainingConfig):
            raise ValueError("config must be FusionWeightTrainingConfig.")
        artifacts = tuple(sorted((_identifier(profile, "artifact profile id"), _sha256(digest, "calibration artifact sha256")) for profile, digest in self.calibration_artifact_sha256s))
        if tuple(profile for profile, _ in artifacts) != profiles:
            raise ValueError("calibration artifacts must exactly cover profile_ids.")
        object.__setattr__(self, "calibration_artifact_sha256s", artifacts)

    @property
    def spec_sha256(self) -> str:
        return _canonical_digest({
            "schema": "rigorousrag-fusion-weight-training-spec/v1",
            "profile_ids": self.profile_ids,
            "calibration_contract_sha256": self.calibration_contract_sha256,
            "calibration_artifact_sha256s": self.calibration_artifact_sha256s,
            "train_split_sha256": self.train_split_sha256,
            "validation_split_sha256": self.validation_split_sha256,
            "source_revision": self.source_revision,
            "config_sha256": self.config.config_sha256,
        })


def examples_sha256(examples: Sequence[FusionWeightExample], profile_ids: Sequence[str]) -> str:
    profiles = tuple(profile_ids)
    rows = []
    for item in examples:
        if set(item.probabilities) != set(profiles):
            raise ValueError("every fusion-weight example must exactly cover the training profiles.")
        rows.append({
            "probabilities": tuple((profile, item.probabilities[profile]) for profile in profiles),
            "relevant": item.relevant,
            "weight": item.weight,
        })
    return _canonical_digest({"schema": "rigorousrag-fusion-weight-examples/v1", "profiles": profiles, "examples": rows})


@dataclass(frozen=True)
class FusionWeightTrainingState:
    spec_sha256: str
    train_examples_sha256: str
    validation_examples_sha256: str
    epoch: int
    batch_index: int
    theta: tuple[float, ...]
    best_theta: tuple[float, ...]
    best_validation_loss: float | None
    best_epoch: int | None
    stale_epochs: int
    completed: bool

    def __post_init__(self) -> None:
        for name in ("spec_sha256", "train_examples_sha256", "validation_examples_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in ("epoch", "batch_index", "stale_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        theta = tuple(float(value) for value in self.theta)
        best_theta = tuple(float(value) for value in self.best_theta)
        if not theta or len(theta) != len(best_theta) or any(not math.isfinite(value) for value in theta + best_theta):
            raise ValueError("theta vectors must be finite and equally sized.")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "best_theta", best_theta)
        if self.best_validation_loss is not None:
            value = float(self.best_validation_loss)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("best_validation_loss must be finite and non-negative.")
            object.__setattr__(self, "best_validation_loss", value)
        if self.best_epoch is not None and (isinstance(self.best_epoch, bool) or not isinstance(self.best_epoch, int) or self.best_epoch < 0):
            raise ValueError("best_epoch must be a non-negative integer when set.")
        if not isinstance(self.completed, bool):
            raise ValueError("completed must be boolean.")

    @property
    def state_sha256(self) -> str:
        return _canonical_digest({"schema": "rigorousrag-fusion-weight-training-state/v1", **asdict(self)})


@dataclass(frozen=True)
class LearnedFusionWeightArtifact:
    spec_sha256: str
    profile_weights: tuple[tuple[str, float], ...]
    calibration_contract_sha256: str
    calibration_artifact_sha256s: tuple[tuple[str, str], ...]
    train_examples_sha256: str
    validation_examples_sha256: str
    best_epoch: int
    validation_loss: float
    artifact_sha256: str

    def __post_init__(self) -> None:
        for name in ("spec_sha256", "calibration_contract_sha256", "train_examples_sha256", "validation_examples_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        weights = tuple(sorted(( _identifier(profile, "profile id"), float(weight)) for profile, weight in self.profile_weights))
        if not weights or any(not math.isfinite(weight) or weight < 0.0 for _, weight in weights):
            raise ValueError("profile weights must be finite and non-negative.")
        if abs(sum(weight for _, weight in weights) - 1.0) > 1e-9:
            raise ValueError("profile weights must sum to one.")
        object.__setattr__(self, "profile_weights", weights)
        artifacts = tuple(sorted((_identifier(profile, "artifact profile id"), _sha256(digest, "artifact sha256")) for profile, digest in self.calibration_artifact_sha256s))
        if tuple(profile for profile, _ in artifacts) != tuple(profile for profile, _ in weights):
            raise ValueError("calibration artifacts must cover the learned weight profiles.")
        object.__setattr__(self, "calibration_artifact_sha256s", artifacts)
        if isinstance(self.best_epoch, bool) or not isinstance(self.best_epoch, int) or self.best_epoch < 0:
            raise ValueError("best_epoch must be non-negative.")
        loss = float(self.validation_loss)
        if not math.isfinite(loss) or loss < 0.0:
            raise ValueError("validation_loss must be finite and non-negative.")
        object.__setattr__(self, "validation_loss", loss)
        expected = _canonical_digest(self._payload())
        provided = _sha256(self.artifact_sha256, "artifact_sha256")
        if expected != provided:
            raise ValueError("artifact_sha256 does not match learned fusion artifact content.")
        object.__setattr__(self, "artifact_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-learned-fusion-weights/v1",
            "spec_sha256": self.spec_sha256,
            "profile_weights": self.profile_weights,
            "calibration_contract_sha256": self.calibration_contract_sha256,
            "calibration_artifact_sha256s": self.calibration_artifact_sha256s,
            "train_examples_sha256": self.train_examples_sha256,
            "validation_examples_sha256": self.validation_examples_sha256,
            "best_epoch": self.best_epoch,
            "validation_loss": self.validation_loss,
        }

    @classmethod
    def build(cls, *, spec: FusionWeightTrainingSpec, state: FusionWeightTrainingState) -> "LearnedFusionWeightArtifact":
        if state.best_validation_loss is None or state.best_epoch is None:
            raise ValueError("training state has no validated best checkpoint.")
        weights = tuple(zip(spec.profile_ids, _softmax(state.best_theta)))
        payload = {
            "schema": "rigorousrag-learned-fusion-weights/v1",
            "spec_sha256": spec.spec_sha256,
            "profile_weights": weights,
            "calibration_contract_sha256": spec.calibration_contract_sha256,
            "calibration_artifact_sha256s": spec.calibration_artifact_sha256s,
            "train_examples_sha256": state.train_examples_sha256,
            "validation_examples_sha256": state.validation_examples_sha256,
            "best_epoch": state.best_epoch,
            "validation_loss": state.best_validation_loss,
        }
        return cls(**payload, artifact_sha256=_canonical_digest(payload))

    def probability(self, probabilities: Mapping[str, float]) -> float:
        if set(probabilities) != {profile for profile, _ in self.profile_weights}:
            raise ValueError("probability vector must exactly cover learned profiles.")
        fused_logit = sum(weight * _logit(_probability(probabilities[profile], profile)) for profile, weight in self.profile_weights)
        return _sigmoid(fused_logit)


def initialize_training_state(spec: FusionWeightTrainingSpec, train_examples: Sequence[FusionWeightExample], validation_examples: Sequence[FusionWeightExample]) -> FusionWeightTrainingState:
    if not train_examples or not validation_examples:
        raise ValueError("train and validation examples must be non-empty.")
    train_digest = examples_sha256(train_examples, spec.profile_ids)
    validation_digest = examples_sha256(validation_examples, spec.profile_ids)
    theta = tuple(0.0 for _ in spec.profile_ids)
    return FusionWeightTrainingState(spec.spec_sha256, train_digest, validation_digest, 0, 0, theta, theta, None, None, 0, False)


def _epoch_order(count: int, *, seed: int, epoch: int) -> list[int]:
    mixed = int(hashlib.sha256(f"{seed}:{epoch}".encode("utf-8")).hexdigest()[:16], 16)
    values = list(range(count))
    random.Random(mixed).shuffle(values)
    return values


def _example_loss(probability: float, relevant: bool) -> float:
    p = _clip_probability(probability)
    return -math.log(p) if relevant else -math.log(1.0 - p)


def _validation_loss(theta: Sequence[float], examples: Sequence[FusionWeightExample], profiles: Sequence[str], positive_class_weight: float) -> float:
    fusion_weights = _softmax(theta)
    total_weight = 0.0
    total_loss = 0.0
    for item in examples:
        effective = item.weight * (positive_class_weight if item.relevant else 1.0)
        z = sum(fusion_weights[index] * _logit(item.probabilities[profile]) for index, profile in enumerate(profiles))
        total_loss += effective * _example_loss(_sigmoid(z), item.relevant)
        total_weight += effective
    return total_loss / total_weight


def advance_training(spec: FusionWeightTrainingSpec, state: FusionWeightTrainingState, train_examples: Sequence[FusionWeightExample], validation_examples: Sequence[FusionWeightExample], *, max_batches: int | None = None) -> FusionWeightTrainingState:
    if state.spec_sha256 != spec.spec_sha256:
        raise ValueError("training state does not belong to this training spec.")
    if examples_sha256(train_examples, spec.profile_ids) != state.train_examples_sha256 or examples_sha256(validation_examples, spec.profile_ids) != state.validation_examples_sha256:
        raise ValueError("training or validation examples differ from resumable state identity.")
    if state.completed:
        return state
    if max_batches is not None and (isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1):
        raise ValueError("max_batches must be positive when set.")

    config = spec.config
    theta = list(state.theta)
    best_theta = state.best_theta
    best_loss = state.best_validation_loss
    best_epoch = state.best_epoch
    stale_epochs = state.stale_epochs
    epoch = state.epoch
    batch_index = state.batch_index
    consumed = 0

    while epoch < config.epochs:
        order = _epoch_order(len(train_examples), seed=config.seed, epoch=epoch)
        batch_count = (len(order) + config.batch_size - 1) // config.batch_size
        while batch_index < batch_count:
            start = batch_index * config.batch_size
            indices = order[start : start + config.batch_size]
            weights = _softmax(theta)
            gradient = [config.l2 * value for value in theta]
            normalizer = 0.0
            for row_index in indices:
                item = train_examples[row_index]
                effective = item.weight * (config.positive_class_weight if item.relevant else 1.0)
                features = [_logit(item.probabilities[profile]) for profile in spec.profile_ids]
                z = sum(weights[index] * features[index] for index in range(len(weights)))
                prediction = _sigmoid(z)
                error = prediction - float(item.relevant)
                for index in range(len(gradient)):
                    gradient[index] += effective * error * weights[index] * (features[index] - z)
                normalizer += effective
            if normalizer:
                gradient = [value / normalizer for value in gradient]
            norm = math.sqrt(sum(value * value for value in gradient))
            if norm > config.gradient_clip_norm:
                scale = config.gradient_clip_norm / norm
                gradient = [value * scale for value in gradient]
            theta = [theta[index] - config.learning_rate * gradient[index] for index in range(len(theta))]
            mean_theta = sum(theta) / len(theta)
            theta = [value - mean_theta for value in theta]
            batch_index += 1
            consumed += 1
            if max_batches is not None and consumed >= max_batches:
                return FusionWeightTrainingState(spec.spec_sha256, state.train_examples_sha256, state.validation_examples_sha256, epoch, batch_index, tuple(theta), best_theta, best_loss, best_epoch, stale_epochs, False)

        validation_loss = _validation_loss(theta, validation_examples, spec.profile_ids, config.positive_class_weight)
        if best_loss is None or validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_theta = tuple(theta)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        epoch += 1
        batch_index = 0
        if stale_epochs >= config.patience:
            return FusionWeightTrainingState(spec.spec_sha256, state.train_examples_sha256, state.validation_examples_sha256, epoch, 0, tuple(theta), best_theta, best_loss, best_epoch, stale_epochs, True)

    return FusionWeightTrainingState(spec.spec_sha256, state.train_examples_sha256, state.validation_examples_sha256, epoch, 0, tuple(theta), best_theta, best_loss, best_epoch, stale_epochs, True)


def fit_fusion_weights(spec: FusionWeightTrainingSpec, train_examples: Sequence[FusionWeightExample], validation_examples: Sequence[FusionWeightExample]) -> LearnedFusionWeightArtifact:
    state = initialize_training_state(spec, train_examples, validation_examples)
    state = advance_training(spec, state, train_examples, validation_examples)
    return LearnedFusionWeightArtifact.build(spec=spec, state=state)


__all__ = [
    "FusionWeightExample",
    "FusionWeightTrainingConfig",
    "FusionWeightTrainingSpec",
    "FusionWeightTrainingState",
    "LearnedFusionWeightArtifact",
    "advance_training",
    "examples_sha256",
    "fit_fusion_weights",
    "initialize_training_state",
]
