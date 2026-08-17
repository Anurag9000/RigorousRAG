"""Executable fitting routines for domain classification and query-plan ranking.

The production artifacts in :mod:`tools.learned_query_planning` are intentionally small
and dependency-free.  This module adds their actual training algorithms without running
them automatically: minibatch multinomial logistic regression for domain routing and
pairwise logistic ranking for query-plan selection, both with deterministic shuffling,
L2 regularisation, validation/early stopping, and atomic JSON checkpoints.
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


def _finite(value: Any, label: str) -> float:
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


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


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _softmax(logits: Sequence[float]) -> list[float]:
    maximum = max(logits)
    values = [math.exp(value - maximum) for value in logits]
    total = sum(values)
    return [value / total for value in values]


@dataclass(frozen=True)
class FittingConfig:
    learning_rate: float = 0.05
    epochs: int = 100
    batch_size: int = 64
    l2: float = 1e-4
    seed: int = 0
    patience: int | None = 10
    min_delta: float = 1e-5
    checkpoint_every_epochs: int = 1

    def __post_init__(self) -> None:
        if _finite(self.learning_rate, "learning_rate") <= 0.0:
            raise ValueError("learning_rate must be positive")
        for name in ("epochs", "batch_size", "checkpoint_every_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if _finite(self.l2, "l2") < 0.0 or _finite(self.min_delta, "min_delta") < 0.0:
            raise ValueError("regularisation/min_delta must be non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed <= 2**63 - 1:
            raise ValueError("seed must be a non-negative 63-bit integer")
        if self.patience is not None and (isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience <= 0):
            raise ValueError("patience must be positive or None")


@dataclass(frozen=True)
class DomainFitExample:
    features: FeatureVector
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.features, FeatureVector):
            raise ValueError("features must be FeatureVector")
        object.__setattr__(self, "label", _identifier(self.label, "label", 500))


@dataclass(frozen=True)
class PlanPreferenceExample:
    preferred_features: FeatureVector
    rejected_features: FeatureVector
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.preferred_features, FeatureVector) or not isinstance(self.rejected_features, FeatureVector):
            raise ValueError("plan preference features must be FeatureVector")
        if self.preferred_features.schema != self.rejected_features.schema:
            raise ValueError("preferred and rejected features must share a schema")
        selected = _finite(self.weight, "weight")
        if selected <= 0.0:
            raise ValueError("weight must be positive")
        object.__setattr__(self, "weight", selected)


@dataclass(frozen=True)
class FitEpoch:
    epoch: int
    training_loss: float
    validation_loss: float | None


@dataclass(frozen=True)
class FitResult:
    artifact_digest: str
    epochs_completed: int
    best_epoch: int
    best_validation_loss: float | None
    history: tuple[FitEpoch, ...]
    stopped_early: bool


class AtomicFitCheckpointStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("fit checkpoint root may not be a symlink")

    def save(self, name: str, payload: Mapping[str, Any]) -> str:
        selected_name = _identifier(name, "checkpoint name", 500)
        encoded = _canonical(payload) + b"\n"
        digest = hashlib.sha256(encoded).hexdigest()
        destination = self.root / f"{selected_name}-{digest}.json"
        descriptor, temporary_name = tempfile.mkstemp(prefix=".fit-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            if destination.exists():
                os.unlink(temporary_name)
            else:
                os.replace(temporary_name, destination)
            pointer = self.root / f"{selected_name}-latest.json"
            descriptor, pointer_tmp = tempfile.mkstemp(prefix=".fit-pointer-", suffix=".tmp", dir=self.root)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(_canonical({"digest": digest, "filename": destination.name}) + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(pointer_tmp, pointer)
            finally:
                if os.path.exists(pointer_tmp):
                    os.unlink(pointer_tmp)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return digest


def _validate_domain_examples(examples: Sequence[DomainFitExample], labels: Sequence[str]) -> tuple[str, ...]:
    if not examples:
        raise ValueError("domain fitting requires examples")
    selected_labels = tuple(_identifier(value, "domain label", 500) for value in labels)
    if not selected_labels or len(set(selected_labels)) != len(selected_labels):
        raise ValueError("domain labels must be non-empty and unique")
    schema = examples[0].features.schema
    for example in examples:
        if example.features.schema != schema:
            raise ValueError("all domain examples must share the same feature schema")
        if example.label not in selected_labels:
            raise ValueError(f"domain example references unknown label {example.label}")
    return schema


def _domain_loss(
    examples: Sequence[DomainFitExample],
    labels: Sequence[str],
    weights: Sequence[Sequence[float]],
    bias: Sequence[float],
    l2: float,
) -> float:
    label_index = {label: index for index, label in enumerate(labels)}
    loss = 0.0
    for example in examples:
        logits = [
            bias[row] + sum(weight * feature for weight, feature in zip(weights[row], example.features.values))
            for row in range(len(labels))
        ]
        probabilities = _softmax(logits)
        loss -= math.log(max(probabilities[label_index[example.label]], 1e-15))
    loss /= len(examples)
    loss += 0.5 * l2 * sum(value * value for row in weights for value in row)
    return loss


def fit_domain_classifier(
    training: Sequence[DomainFitExample],
    *,
    labels: Sequence[str],
    fallback_label: str,
    minimum_confidence: float = 0.55,
    validation: Sequence[DomainFitExample] = (),
    config: FittingConfig = FittingConfig(),
    training_manifest_digest: str,
    checkpoint_store: AtomicFitCheckpointStore | None = None,
) -> tuple[LinearDomainClassifier, FitResult]:
    schema = _validate_domain_examples(training, labels)
    selected_labels = tuple(labels)
    if validation:
        validation_schema = _validate_domain_examples(validation, labels)
        if validation_schema != schema:
            raise ValueError("validation feature schema differs from training schema")
    manifest_digest = _sha256(training_manifest_digest, "training_manifest_digest")
    label_index = {label: index for index, label in enumerate(selected_labels)}
    weights = [[0.0 for _ in schema] for _ in selected_labels]
    bias = [0.0 for _ in selected_labels]
    rng = random.Random(config.seed)
    best_weights = [row[:] for row in weights]
    best_bias = bias[:]
    best_loss: float | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[FitEpoch] = []
    stopped = False

    for epoch in range(config.epochs):
        order = list(range(len(training)))
        rng.shuffle(order)
        for start in range(0, len(order), config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            grad_weights = [[0.0 for _ in schema] for _ in selected_labels]
            grad_bias = [0.0 for _ in selected_labels]
            for index in batch_indices:
                example = training[index]
                logits = [
                    bias[row] + sum(weight * feature for weight, feature in zip(weights[row], example.features.values))
                    for row in range(len(selected_labels))
                ]
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
                    gradient = scale * grad_weights[row][column] + config.l2 * weights[row][column]
                    weights[row][column] -= config.learning_rate * gradient

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
        if checkpoint_store is not None and (epoch + 1) % config.checkpoint_every_epochs == 0:
            checkpoint_store.save(
                "domain-classifier",
                {
                    "epoch": epoch + 1,
                    "labels": list(selected_labels),
                    "feature_schema": list(schema),
                    "weights": weights,
                    "bias": bias,
                    "best_epoch": best_epoch,
                    "best_loss": best_loss,
                    "rng_state": repr(rng.getstate()),
                    "training_manifest_digest": manifest_digest,
                    "config": asdict(config),
                },
            )
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
    return artifact, FitResult(
        artifact_digest=artifact.artifact_digest,
        epochs_completed=len(history),
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        history=tuple(history),
        stopped_early=stopped,
    )


def _validate_preferences(examples: Sequence[PlanPreferenceExample]) -> tuple[str, ...]:
    if not examples:
        raise ValueError("plan-ranker fitting requires preference examples")
    schema = examples[0].preferred_features.schema
    for example in examples:
        if example.preferred_features.schema != schema or example.rejected_features.schema != schema:
            raise ValueError("all plan preferences must share one feature schema")
    return schema


def _preference_loss(examples: Sequence[PlanPreferenceExample], weights: Sequence[float], l2: float) -> float:
    total = 0.0
    for example in examples:
        difference = [left - right for left, right in zip(example.preferred_features.values, example.rejected_features.values)]
        margin = sum(weight * value for weight, value in zip(weights, difference))
        total += example.weight * (max(-margin, 0.0) + math.log1p(math.exp(-abs(margin))))
    return total / len(examples) + 0.5 * l2 * sum(value * value for value in weights)


def fit_plan_ranker(
    training: Sequence[PlanPreferenceExample],
    *,
    validation: Sequence[PlanPreferenceExample] = (),
    config: FittingConfig = FittingConfig(),
    latency_penalty: float = 0.0,
    cost_penalty: float = 0.0,
    risk_penalty: float = 0.0,
    training_manifest_digest: str,
    checkpoint_store: AtomicFitCheckpointStore | None = None,
) -> tuple[LinearPlanRanker, FitResult]:
    schema = _validate_preferences(training)
    if validation and _validate_preferences(validation) != schema:
        raise ValueError("validation plan feature schema differs")
    manifest_digest = _sha256(training_manifest_digest, "training_manifest_digest")
    weights = [0.0 for _ in schema]
    best_weights = weights[:]
    best_loss: float | None = None
    best_epoch = 0
    bad_epochs = 0
    stopped = False
    history: list[FitEpoch] = []
    rng = random.Random(config.seed)

    for epoch in range(config.epochs):
        order = list(range(len(training)))
        rng.shuffle(order)
        for start in range(0, len(order), config.batch_size):
            batch = [training[index] for index in order[start : start + config.batch_size]]
            gradient = [0.0 for _ in schema]
            for example in batch:
                difference = [left - right for left, right in zip(example.preferred_features.values, example.rejected_features.values)]
                margin = sum(weight * value for weight, value in zip(weights, difference))
                # d softplus(-margin) / d margin = -sigmoid(-margin), stable branch.
                if margin >= 0.0:
                    sigmoid_negative = math.exp(-margin) / (1.0 + math.exp(-margin))
                else:
                    sigmoid_negative = 1.0 / (1.0 + math.exp(margin))
                for column, value in enumerate(difference):
                    gradient[column] += -example.weight * sigmoid_negative * value
            scale = 1.0 / len(batch)
            for column in range(len(schema)):
                weights[column] -= config.learning_rate * (scale * gradient[column] + config.l2 * weights[column])

        training_loss = _preference_loss(training, weights, config.l2)
        validation_loss = _preference_loss(validation, weights, config.l2) if validation else None
        monitored = validation_loss if validation_loss is not None else training_loss
        history.append(FitEpoch(epoch + 1, training_loss, validation_loss))
        if best_loss is None or monitored < best_loss - config.min_delta:
            best_loss = monitored
            best_epoch = epoch + 1
            best_weights = weights[:]
            bad_epochs = 0
        else:
            bad_epochs += 1
        if checkpoint_store is not None and (epoch + 1) % config.checkpoint_every_epochs == 0:
            checkpoint_store.save(
                "plan-ranker",
                {
                    "epoch": epoch + 1,
                    "feature_schema": list(schema),
                    "weights": weights,
                    "best_epoch": best_epoch,
                    "best_loss": best_loss,
                    "training_manifest_digest": manifest_digest,
                    "config": asdict(config),
                },
            )
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
    return artifact, FitResult(
        artifact_digest=artifact.artifact_digest,
        epochs_completed=len(history),
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        history=tuple(history),
        stopped_early=stopped,
    )


__all__ = [
    "AtomicFitCheckpointStore",
    "DomainFitExample",
    "FitEpoch",
    "FitResult",
    "FittingConfig",
    "PlanPreferenceExample",
    "canonical_digest",
    "fit_domain_classifier",
    "fit_plan_ranker",
]
