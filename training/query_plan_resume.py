"""Exact checkpoint/resume fitting for learned domain routing and plan ranking.

The lightweight fitting module emits deployable dependency-free artifacts. This module
adds exact resumability: optimizer-free SGD state, epoch/batch cursor, model parameters,
best parameters, early-stopping counters, deterministic permutation state and manifest
identity are all stored in canonical JSON. No fitting runs on import.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.learned_query_planning import FeatureVector, LinearDomainClassifier, LinearPlanRanker
from training.query_plan_fitting import DomainFitExample, FitEpoch, FitResult, FittingConfig, PlanPreferenceExample


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _sha256(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _finite(value: Any, label: str) -> float:
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _encode_random_state(value: Any) -> Any:
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_random_state(item) for item in value]}
    if isinstance(value, list):
        return [_encode_random_state(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode_random_state(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported random-state value {type(value).__name__}")


def _decode_random_state(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_decode_random_state(item) for item in value["__tuple__"])
    if isinstance(value, list):
        return [_decode_random_state(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_random_state(item) for key, item in value.items()}
    return value


def _softmax(logits: Sequence[float]) -> list[float]:
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


@dataclass(frozen=True)
class ResumeCursor:
    epoch: int
    next_batch_start: int
    permutation: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 0:
            raise ValueError("epoch must be non-negative")
        if isinstance(self.next_batch_start, bool) or not isinstance(self.next_batch_start, int) or self.next_batch_start < 0:
            raise ValueError("next_batch_start must be non-negative")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.permutation):
            raise ValueError("permutation entries must be non-negative integers")


@dataclass(frozen=True)
class DomainFittingState:
    version: int
    training_manifest_digest: str
    config_digest: str
    labels: tuple[str, ...]
    feature_schema: tuple[str, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    best_weights: tuple[tuple[float, ...], ...]
    best_bias: tuple[float, ...]
    best_loss: float | None
    best_epoch: int
    bad_epochs: int
    cursor: ResumeCursor
    random_state: Any
    history: tuple[FitEpoch, ...]

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("unsupported domain fitting state version")
        object.__setattr__(self, "training_manifest_digest", _sha256(self.training_manifest_digest, "training_manifest_digest"))
        object.__setattr__(self, "config_digest", _sha256(self.config_digest, "config_digest"))

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class PlanFittingState:
    version: int
    training_manifest_digest: str
    config_digest: str
    feature_schema: tuple[str, ...]
    weights: tuple[float, ...]
    best_weights: tuple[float, ...]
    best_loss: float | None
    best_epoch: int
    bad_epochs: int
    cursor: ResumeCursor
    random_state: Any
    history: tuple[FitEpoch, ...]

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("unsupported plan fitting state version")
        object.__setattr__(self, "training_manifest_digest", _sha256(self.training_manifest_digest, "training_manifest_digest"))
        object.__setattr__(self, "config_digest", _sha256(self.config_digest, "config_digest"))

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


class ResumeStateStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("resume-state root may not be a symlink")

    def save(self, name: str, payload: Mapping[str, Any]) -> str:
        selected_name = _identifier(name, "resume state name", 300)
        encoded = _canonical(payload) + b"\n"
        digest = hashlib.sha256(encoded).hexdigest()
        destination = self.root / f"{selected_name}-{digest}.json"
        descriptor, temporary_name = tempfile.mkstemp(prefix=".router-fit-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            if destination.exists():
                os.unlink(temporary_name)
            else:
                os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        pointer_payload = _canonical({"digest": digest, "filename": destination.name}) + b"\n"
        descriptor, pointer_tmp = tempfile.mkstemp(prefix=".router-fit-pointer-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(pointer_payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(pointer_tmp, self.root / f"{selected_name}-latest.json")
        finally:
            if os.path.exists(pointer_tmp):
                os.unlink(pointer_tmp)
        return digest

    def load_latest(self, name: str) -> Mapping[str, Any]:
        selected_name = _identifier(name, "resume state name", 300)
        pointer = json.loads((self.root / f"{selected_name}-latest.json").read_text(encoding="utf-8"))
        digest = _sha256(pointer["digest"], "resume state digest")
        filename = _identifier(pointer["filename"], "resume state filename", 500)
        if "/" in filename or "\\" in filename:
            raise ValueError("resume state filename must be a basename")
        path = (self.root / filename).resolve(strict=True)
        if path.parent != self.root or path.is_symlink():
            raise ValueError("resume state path escaped configured root")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise RuntimeError("resume-state payload digest mismatch")
        return json.loads(raw.decode("utf-8"))


def _domain_schema(examples: Sequence[DomainFitExample], labels: Sequence[str]) -> tuple[str, ...]:
    if not examples:
        raise ValueError("domain fitting requires examples")
    selected_labels = tuple(_identifier(value, "label", 500) for value in labels)
    if not selected_labels or len(set(selected_labels)) != len(selected_labels):
        raise ValueError("labels must be non-empty and unique")
    schema = examples[0].features.schema
    for example in examples:
        if not isinstance(example, DomainFitExample) or example.features.schema != schema:
            raise ValueError("domain examples must share one feature schema")
        if example.label not in selected_labels:
            raise ValueError("domain example references an unknown label")
    return schema


def _domain_loss(
    examples: Sequence[DomainFitExample],
    labels: Sequence[str],
    weights: Sequence[Sequence[float]],
    bias: Sequence[float],
    l2: float,
) -> float:
    label_index = {label: index for index, label in enumerate(labels)}
    total = 0.0
    for example in examples:
        logits = [bias[row] + sum(weight * feature for weight, feature in zip(weights[row], example.features.values)) for row in range(len(labels))]
        probability = _softmax(logits)
        total -= math.log(max(probability[label_index[example.label]], 1e-15))
    return total / len(examples) + 0.5 * l2 * sum(value * value for row in weights for value in row)


def _domain_state_from_payload(payload: Mapping[str, Any]) -> DomainFittingState:
    return DomainFittingState(
        version=int(payload["version"]),
        training_manifest_digest=payload["training_manifest_digest"],
        config_digest=payload["config_digest"],
        labels=tuple(payload["labels"]),
        feature_schema=tuple(payload["feature_schema"]),
        weights=tuple(tuple(float(value) for value in row) for row in payload["weights"]),
        bias=tuple(float(value) for value in payload["bias"]),
        best_weights=tuple(tuple(float(value) for value in row) for row in payload["best_weights"]),
        best_bias=tuple(float(value) for value in payload["best_bias"]),
        best_loss=None if payload.get("best_loss") is None else float(payload["best_loss"]),
        best_epoch=int(payload["best_epoch"]),
        bad_epochs=int(payload["bad_epochs"]),
        cursor=ResumeCursor(
            epoch=int(payload["cursor"]["epoch"]),
            next_batch_start=int(payload["cursor"]["next_batch_start"]),
            permutation=tuple(int(value) for value in payload["cursor"]["permutation"]),
        ),
        random_state=payload["random_state"],
        history=tuple(FitEpoch(**value) for value in payload.get("history", [])),
    )


def fit_domain_classifier_resumable(
    training: Sequence[DomainFitExample],
    *,
    labels: Sequence[str],
    fallback_label: str,
    training_manifest_digest: str,
    validation: Sequence[DomainFitExample] = (),
    minimum_confidence: float = 0.55,
    config: FittingConfig = FittingConfig(),
    state_store: ResumeStateStore | None = None,
    resume: bool = False,
    checkpoint_every_batches: int = 1,
) -> tuple[LinearDomainClassifier, FitResult]:
    schema = _domain_schema(training, labels)
    selected_labels = tuple(labels)
    if validation and _domain_schema(validation, labels) != schema:
        raise ValueError("validation schema differs from training schema")
    manifest_digest = _sha256(training_manifest_digest, "training_manifest_digest")
    config_digest = canonical_digest(asdict(config))
    label_index = {label: index for index, label in enumerate(selected_labels)}
    rng = random.Random(config.seed)
    weights = [[0.0 for _ in schema] for _ in selected_labels]
    bias = [0.0 for _ in selected_labels]
    best_weights = [row[:] for row in weights]
    best_bias = bias[:]
    best_loss: float | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[FitEpoch] = []
    epoch = 0
    permutation: list[int] = []
    next_batch_start = 0

    if resume:
        if state_store is None:
            raise ValueError("resume=True requires a state_store")
        state = _domain_state_from_payload(state_store.load_latest("domain-classifier"))
        if state.training_manifest_digest != manifest_digest or state.config_digest != config_digest:
            raise ValueError("domain resume state differs from configured manifest or fitting config")
        if state.labels != selected_labels or state.feature_schema != schema:
            raise ValueError("domain resume state label/feature schema differs")
        weights = [list(row) for row in state.weights]
        bias = list(state.bias)
        best_weights = [list(row) for row in state.best_weights]
        best_bias = list(state.best_bias)
        best_loss = state.best_loss
        best_epoch = state.best_epoch
        bad_epochs = state.bad_epochs
        history = list(state.history)
        epoch = state.cursor.epoch
        permutation = list(state.cursor.permutation)
        next_batch_start = state.cursor.next_batch_start
        rng.setstate(_decode_random_state(state.random_state))

    if checkpoint_every_batches <= 0:
        raise ValueError("checkpoint_every_batches must be positive")
    stopped = False
    batches_since_checkpoint = 0
    while epoch < config.epochs:
        if not permutation:
            permutation = list(range(len(training)))
            rng.shuffle(permutation)
            next_batch_start = 0
        while next_batch_start < len(permutation):
            batch_indices = permutation[next_batch_start : next_batch_start + config.batch_size]
            grad_weights = [[0.0 for _ in schema] for _ in selected_labels]
            grad_bias = [0.0 for _ in selected_labels]
            for index in batch_indices:
                example = training[index]
                logits = [bias[row] + sum(weight * feature for weight, feature in zip(weights[row], example.features.values)) for row in range(len(selected_labels))]
                probabilities = _softmax(logits)
                target = label_index[example.label]
                for row in range(len(selected_labels)):
                    error = probabilities[row] - (1.0 if row == target else 0.0)
                    grad_bias[row] += error
                    for column, feature in enumerate(example.features.values):
                        grad_weights[row][column] += error * feature
            scale = 1.0 / len(batch_indices)
            for row in range(len(selected_labels)):
                bias[row] -= config.learning_rate * scale * grad_bias[row]
                for column in range(len(schema)):
                    weights[row][column] -= config.learning_rate * (scale * grad_weights[row][column] + config.l2 * weights[row][column])
            next_batch_start += len(batch_indices)
            batches_since_checkpoint += 1
            if state_store is not None and batches_since_checkpoint >= checkpoint_every_batches:
                state_store.save(
                    "domain-classifier",
                    asdict(
                        DomainFittingState(
                            1,
                            manifest_digest,
                            config_digest,
                            selected_labels,
                            schema,
                            tuple(tuple(value for value in row) for row in weights),
                            tuple(bias),
                            tuple(tuple(value for value in row) for row in best_weights),
                            tuple(best_bias),
                            best_loss,
                            best_epoch,
                            bad_epochs,
                            ResumeCursor(epoch, next_batch_start, tuple(permutation)),
                            _encode_random_state(rng.getstate()),
                            tuple(history),
                        )
                    ),
                )
                batches_since_checkpoint = 0

        training_loss = _domain_loss(training, selected_labels, weights, bias, config.l2)
        validation_loss = _domain_loss(validation, selected_labels, weights, bias, config.l2) if validation else None
        monitored = validation_loss if validation_loss is not None else training_loss
        history.append(FitEpoch(epoch + 1, training_loss, validation_loss))
        if best_loss is None or monitored < best_loss - config.min_delta:
            best_loss = monitored
            best_epoch = epoch + 1
            best_weights = [row[:] for row in weights]
            best_bias = bias[:]
            bad_epochs = 0
        else:
            bad_epochs += 1
        epoch += 1
        permutation = []
        next_batch_start = 0
        if config.patience is not None and bad_epochs >= config.patience:
            stopped = True
            break

    artifact = LinearDomainClassifier(
        labels=selected_labels,
        feature_schema=schema,
        weights=tuple(tuple(value for value in row) for row in best_weights),
        bias=tuple(best_bias),
        fallback_label=fallback_label,
        minimum_confidence=minimum_confidence,
        training_manifest_digest=manifest_digest,
    )
    return artifact, FitResult(artifact.artifact_digest, len(history), best_epoch, best_loss, tuple(history), stopped)


def _plan_schema(examples: Sequence[PlanPreferenceExample]) -> tuple[str, ...]:
    if not examples:
        raise ValueError("plan fitting requires preference examples")
    schema = examples[0].preferred_features.schema
    for example in examples:
        if not isinstance(example, PlanPreferenceExample):
            raise ValueError("invalid plan preference type")
        if example.preferred_features.schema != schema or example.rejected_features.schema != schema:
            raise ValueError("plan preferences must share one feature schema")
    return schema


def _plan_loss(examples: Sequence[PlanPreferenceExample], weights: Sequence[float], l2: float) -> float:
    total = 0.0
    for example in examples:
        difference = [left - right for left, right in zip(example.preferred_features.values, example.rejected_features.values)]
        margin = sum(weight * value for weight, value in zip(weights, difference))
        total += example.weight * (max(-margin, 0.0) + math.log1p(math.exp(-abs(margin))))
    return total / len(examples) + 0.5 * l2 * sum(value * value for value in weights)


def _plan_state_from_payload(payload: Mapping[str, Any]) -> PlanFittingState:
    return PlanFittingState(
        version=int(payload["version"]),
        training_manifest_digest=payload["training_manifest_digest"],
        config_digest=payload["config_digest"],
        feature_schema=tuple(payload["feature_schema"]),
        weights=tuple(float(value) for value in payload["weights"]),
        best_weights=tuple(float(value) for value in payload["best_weights"]),
        best_loss=None if payload.get("best_loss") is None else float(payload["best_loss"]),
        best_epoch=int(payload["best_epoch"]),
        bad_epochs=int(payload["bad_epochs"]),
        cursor=ResumeCursor(
            epoch=int(payload["cursor"]["epoch"]),
            next_batch_start=int(payload["cursor"]["next_batch_start"]),
            permutation=tuple(int(value) for value in payload["cursor"]["permutation"]),
        ),
        random_state=payload["random_state"],
        history=tuple(FitEpoch(**value) for value in payload.get("history", [])),
    )


def fit_plan_ranker_resumable(
    training: Sequence[PlanPreferenceExample],
    *,
    training_manifest_digest: str,
    validation: Sequence[PlanPreferenceExample] = (),
    config: FittingConfig = FittingConfig(),
    latency_penalty: float = 0.0,
    cost_penalty: float = 0.0,
    risk_penalty: float = 0.0,
    state_store: ResumeStateStore | None = None,
    resume: bool = False,
    checkpoint_every_batches: int = 1,
) -> tuple[LinearPlanRanker, FitResult]:
    schema = _plan_schema(training)
    if validation and _plan_schema(validation) != schema:
        raise ValueError("validation plan schema differs")
    manifest_digest = _sha256(training_manifest_digest, "training_manifest_digest")
    config_digest = canonical_digest(asdict(config))
    rng = random.Random(config.seed)
    weights = [0.0 for _ in schema]
    best_weights = weights[:]
    best_loss: float | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[FitEpoch] = []
    epoch = 0
    permutation: list[int] = []
    next_batch_start = 0

    if resume:
        if state_store is None:
            raise ValueError("resume=True requires a state_store")
        state = _plan_state_from_payload(state_store.load_latest("plan-ranker"))
        if state.training_manifest_digest != manifest_digest or state.config_digest != config_digest:
            raise ValueError("plan resume state differs from configured manifest or fitting config")
        if state.feature_schema != schema:
            raise ValueError("plan resume feature schema differs")
        weights = list(state.weights)
        best_weights = list(state.best_weights)
        best_loss = state.best_loss
        best_epoch = state.best_epoch
        bad_epochs = state.bad_epochs
        history = list(state.history)
        epoch = state.cursor.epoch
        permutation = list(state.cursor.permutation)
        next_batch_start = state.cursor.next_batch_start
        rng.setstate(_decode_random_state(state.random_state))

    if checkpoint_every_batches <= 0:
        raise ValueError("checkpoint_every_batches must be positive")
    stopped = False
    batches_since_checkpoint = 0
    while epoch < config.epochs:
        if not permutation:
            permutation = list(range(len(training)))
            rng.shuffle(permutation)
            next_batch_start = 0
        while next_batch_start < len(permutation):
            batch = [training[index] for index in permutation[next_batch_start : next_batch_start + config.batch_size]]
            gradient = [0.0 for _ in schema]
            for example in batch:
                difference = [left - right for left, right in zip(example.preferred_features.values, example.rejected_features.values)]
                margin = sum(weight * value for weight, value in zip(weights, difference))
                sigmoid_negative = math.exp(-margin) / (1.0 + math.exp(-margin)) if margin >= 0.0 else 1.0 / (1.0 + math.exp(margin))
                for column, value in enumerate(difference):
                    gradient[column] += -example.weight * sigmoid_negative * value
            scale = 1.0 / len(batch)
            for column in range(len(schema)):
                weights[column] -= config.learning_rate * (scale * gradient[column] + config.l2 * weights[column])
            next_batch_start += len(batch)
            batches_since_checkpoint += 1
            if state_store is not None and batches_since_checkpoint >= checkpoint_every_batches:
                state_store.save(
                    "plan-ranker",
                    asdict(
                        PlanFittingState(
                            1,
                            manifest_digest,
                            config_digest,
                            schema,
                            tuple(weights),
                            tuple(best_weights),
                            best_loss,
                            best_epoch,
                            bad_epochs,
                            ResumeCursor(epoch, next_batch_start, tuple(permutation)),
                            _encode_random_state(rng.getstate()),
                            tuple(history),
                        )
                    ),
                )
                batches_since_checkpoint = 0

        training_loss = _plan_loss(training, weights, config.l2)
        validation_loss = _plan_loss(validation, weights, config.l2) if validation else None
        monitored = validation_loss if validation_loss is not None else training_loss
        history.append(FitEpoch(epoch + 1, training_loss, validation_loss))
        if best_loss is None or monitored < best_loss - config.min_delta:
            best_loss = monitored
            best_epoch = epoch + 1
            best_weights = weights[:]
            bad_epochs = 0
        else:
            bad_epochs += 1
        epoch += 1
        permutation = []
        next_batch_start = 0
        if config.patience is not None and bad_epochs >= config.patience:
            stopped = True
            break

    artifact = LinearPlanRanker(
        feature_schema=schema,
        weights=tuple(best_weights),
        bias=0.0,
        latency_penalty=_finite(latency_penalty, "latency_penalty"),
        cost_penalty=_finite(cost_penalty, "cost_penalty"),
        risk_penalty=_finite(risk_penalty, "risk_penalty"),
        training_manifest_digest=manifest_digest,
    )
    return artifact, FitResult(artifact.artifact_digest, len(history), best_epoch, best_loss, tuple(history), stopped)


__all__ = [
    "DomainFittingState",
    "PlanFittingState",
    "ResumeCursor",
    "ResumeStateStore",
    "canonical_digest",
    "fit_domain_classifier_resumable",
    "fit_plan_ranker_resumable",
]
