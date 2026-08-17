"""Privacy-aware aggregate-learning contracts with explicit security boundaries.

A production secure-aggregation implementation is injected through ``SecureAggregationProvider``
and operates on opaque participant payloads. The included ``LocalReferenceAggregator`` is
intentionally labeled cleartext and is useful only for deterministic local/offline workflows;
it does not provide MPC, homomorphic encryption, or protection from the aggregator.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

_MAX_DIMENSIONS = 100_000
_MAX_PARTICIPANTS = 1_000_000


def _text(value: Any, label: str, maximum: int = 500, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if (not selected and not allow_empty) or len(selected) > maximum or "\x00" in selected:
        raise ValueError(f"{label} is invalid")
    return selected


def _finite(value: Any, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise ValueError(f"{label} is invalid")
    return parsed


def _sha(value: str, label: str) -> str:
    selected = _text(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    if len(values) == 0 or len(values) > _MAX_DIMENSIONS:
        raise ValueError(f"{label} dimension is invalid")
    return tuple(_finite(value, f"{label}[{index}]") for index, value in enumerate(values))


def _l2(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


@dataclass(frozen=True)
class AggregateRoundSpec:
    round_id: str
    feature_schema_sha256: str
    purpose: str
    minimum_cohort_size: int
    clipping_l2_norm: float
    weighting: str = "sample_count"
    differential_privacy_epsilon: float | None = None
    differential_privacy_delta: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_id", _text(self.round_id, "round_id", 500))
        object.__setattr__(self, "feature_schema_sha256", _sha(self.feature_schema_sha256, "feature_schema_sha256"))
        object.__setattr__(self, "purpose", _text(self.purpose, "purpose", 500))
        if isinstance(self.minimum_cohort_size, bool) or not isinstance(self.minimum_cohort_size, int) or not 2 <= self.minimum_cohort_size <= _MAX_PARTICIPANTS:
            raise ValueError("minimum_cohort_size must be at least two")
        object.__setattr__(self, "clipping_l2_norm", _finite(self.clipping_l2_norm, "clipping_l2_norm", 1e-12))
        weighting = _text(self.weighting, "weighting", 64).lower()
        if weighting not in {"sample_count", "uniform"}:
            raise ValueError("weighting must be sample_count or uniform")
        object.__setattr__(self, "weighting", weighting)
        epsilon = self.differential_privacy_epsilon
        delta = self.differential_privacy_delta
        if (epsilon is None) != (delta is None):
            raise ValueError("epsilon and delta must either both be set or both be null")
        if epsilon is not None:
            object.__setattr__(self, "differential_privacy_epsilon", _finite(epsilon, "epsilon", 1e-12))
            selected_delta = _finite(delta, "delta", 0.0)
            if not 0.0 <= selected_delta < 1.0:
                raise ValueError("delta must be in [0,1)")
            object.__setattr__(self, "differential_privacy_delta", selected_delta)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class ParticipantCommitment:
    participant_pseudonym_sha256: str
    payload_sha256: str
    sample_count: int
    feature_schema_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "participant_pseudonym_sha256", _sha(self.participant_pseudonym_sha256, "participant_pseudonym_sha256"))
        object.__setattr__(self, "payload_sha256", _sha(self.payload_sha256, "payload_sha256"))
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or not 1 <= self.sample_count <= 10**12:
            raise ValueError("sample_count is invalid")
        object.__setattr__(self, "feature_schema_sha256", _sha(self.feature_schema_sha256, "feature_schema_sha256"))


@dataclass(frozen=True)
class BoundedContribution:
    commitment: ParticipantCommitment
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.commitment, ParticipantCommitment):
            raise TypeError("commitment must be ParticipantCommitment")
        object.__setattr__(self, "values", _vector(self.values, "values"))
        actual = hashlib.sha256(_canonical(self.values)).hexdigest()
        if actual != self.commitment.payload_sha256:
            raise ValueError("contribution values do not match payload commitment")


@dataclass(frozen=True)
class ProviderAggregateResult:
    values: tuple[float, ...]
    included_commitments: tuple[str, ...]
    aggregate_proof_fingerprint: str
    provider_id: str
    privacy_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _vector(self.values, "values"))
        object.__setattr__(self, "included_commitments", tuple(sorted(set(_sha(item, "included_commitment") for item in self.included_commitments))))
        object.__setattr__(self, "aggregate_proof_fingerprint", _sha(self.aggregate_proof_fingerprint, "aggregate_proof_fingerprint"))
        object.__setattr__(self, "provider_id", _text(self.provider_id, "provider_id", 500))
        if not isinstance(self.privacy_metadata, Mapping):
            raise ValueError("privacy_metadata must be a mapping")
        if len(_canonical(dict(self.privacy_metadata))) > 64_000:
            raise ValueError("privacy_metadata exceeds the byte limit")
        object.__setattr__(self, "privacy_metadata", dict(self.privacy_metadata))


class SecureAggregationProvider(Protocol):
    """Production provider contract; opaque payloads are never decoded by this module."""

    @property
    def provider_id(self) -> str: ...

    def aggregate(
        self,
        spec: AggregateRoundSpec,
        commitments: Sequence[ParticipantCommitment],
        opaque_payloads: Sequence[bytes],
    ) -> ProviderAggregateResult: ...


class DifferentialPrivacyNoiseProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    def add_noise(
        self,
        values: Sequence[float],
        *,
        l2_sensitivity: float,
        epsilon: float,
        delta: float,
        round_fingerprint: str,
    ) -> Sequence[float]: ...


@dataclass(frozen=True)
class AggregateLearningResult:
    round_fingerprint: str
    provider_id: str
    privacy_mode: str
    values: tuple[float, ...]
    participant_count: int
    total_sample_count: int
    included_commitments: tuple[str, ...]
    differential_privacy_applied: bool
    differential_privacy_provider_id: str
    proof_fingerprint: str
    warnings: tuple[str, ...]
    fingerprint: str


class LocalReferenceAggregator:
    """Cleartext reference aggregator. It offers clipping/cohort/DP hooks, not MPC privacy."""

    provider_id = "local-cleartext-reference"

    def aggregate(
        self,
        spec: AggregateRoundSpec,
        contributions: Sequence[BoundedContribution],
        *,
        noise_provider: DifferentialPrivacyNoiseProvider | None = None,
    ) -> AggregateLearningResult:
        if not isinstance(spec, AggregateRoundSpec):
            raise TypeError("spec must be AggregateRoundSpec")
        if len(contributions) > _MAX_PARTICIPANTS or any(not isinstance(item, BoundedContribution) for item in contributions):
            raise ValueError("contributions are invalid")
        if len(contributions) < spec.minimum_cohort_size:
            raise ValueError("minimum cohort size is not satisfied")
        pseudonyms = [item.commitment.participant_pseudonym_sha256 for item in contributions]
        if len(set(pseudonyms)) != len(pseudonyms):
            raise ValueError("duplicate participant pseudonym in aggregate round")
        if any(item.commitment.feature_schema_sha256 != spec.feature_schema_sha256 for item in contributions):
            raise ValueError("contribution feature schema does not match round")
        dimension = len(contributions[0].values)
        if any(len(item.values) != dimension for item in contributions):
            raise ValueError("contribution dimensions do not match")

        accum = [0.0] * dimension
        total_weight = 0.0
        total_samples = 0
        clipped_count = 0
        for item in contributions:
            norm = _l2(item.values)
            scale = 1.0 if norm <= spec.clipping_l2_norm else spec.clipping_l2_norm / norm
            if scale < 1.0:
                clipped_count += 1
            weight = float(item.commitment.sample_count if spec.weighting == "sample_count" else 1)
            total_samples += item.commitment.sample_count
            total_weight += weight
            for index, value in enumerate(item.values):
                accum[index] += value * scale * weight
        if total_weight <= 0:
            raise RuntimeError("aggregate weight is zero")
        values = tuple(item / total_weight for item in accum)
        dp_applied = False
        dp_provider_id = ""
        warnings = ["local reference aggregation exposes cleartext contributions to the aggregator; it is not secure MPC"]
        if spec.differential_privacy_epsilon is not None:
            if noise_provider is None:
                raise RuntimeError("differential privacy parameters require an injected noise provider")
            provider_id = _text(noise_provider.provider_id, "noise provider_id", 500)
            noised = noise_provider.add_noise(
                values,
                l2_sensitivity=spec.clipping_l2_norm / total_weight,
                epsilon=spec.differential_privacy_epsilon,
                delta=spec.differential_privacy_delta or 0.0,
                round_fingerprint=spec.fingerprint,
            )
            values = _vector(tuple(noised), "noised values")
            if len(values) != dimension:
                raise RuntimeError("noise provider changed aggregate dimensionality")
            dp_applied = True
            dp_provider_id = provider_id
        commitments = tuple(sorted(item.commitment.payload_sha256 for item in contributions))
        proof_payload = {
            "spec": spec.fingerprint,
            "commitments": commitments,
            "provider_id": self.provider_id,
            "clipped_count": clipped_count,
            "total_samples": total_samples,
            "dp_provider_id": dp_provider_id,
        }
        proof = hashlib.sha256(_canonical(proof_payload)).hexdigest()
        result_payload = {
            "round_fingerprint": spec.fingerprint,
            "provider_id": self.provider_id,
            "privacy_mode": "reference_cleartext_not_secure_aggregation",
            "values": values,
            "participant_count": len(contributions),
            "total_sample_count": total_samples,
            "commitments": commitments,
            "dp_applied": dp_applied,
            "dp_provider_id": dp_provider_id,
            "proof": proof,
        }
        return AggregateLearningResult(
            round_fingerprint=spec.fingerprint,
            provider_id=self.provider_id,
            privacy_mode="reference_cleartext_not_secure_aggregation",
            values=values,
            participant_count=len(contributions),
            total_sample_count=total_samples,
            included_commitments=commitments,
            differential_privacy_applied=dp_applied,
            differential_privacy_provider_id=dp_provider_id,
            proof_fingerprint=proof,
            warnings=tuple(warnings),
            fingerprint=hashlib.sha256(_canonical(result_payload)).hexdigest(),
        )


def validate_provider_result(
    spec: AggregateRoundSpec,
    commitments: Sequence[ParticipantCommitment],
    result: ProviderAggregateResult,
) -> AggregateLearningResult:
    if not isinstance(spec, AggregateRoundSpec) or not isinstance(result, ProviderAggregateResult):
        raise TypeError("spec/result types are invalid")
    if len(commitments) < spec.minimum_cohort_size:
        raise ValueError("minimum cohort size is not satisfied")
    known = {item.payload_sha256: item for item in commitments}
    if any(not isinstance(item, ParticipantCommitment) for item in commitments):
        raise ValueError("commitments contain an invalid value")
    if any(item.feature_schema_sha256 != spec.feature_schema_sha256 for item in commitments):
        raise ValueError("commitment feature schema does not match round")
    if not set(result.included_commitments).issubset(known):
        raise ValueError("provider result references an unknown commitment")
    if len(result.included_commitments) < spec.minimum_cohort_size:
        raise ValueError("provider result does not satisfy minimum cohort size")
    total_samples = sum(known[item].sample_count for item in result.included_commitments)
    dp_claimed = bool(result.privacy_metadata.get("differential_privacy_applied", False))
    dp_provider_id = str(result.privacy_metadata.get("differential_privacy_provider_id", "")) if dp_claimed else ""
    payload = {
        "round_fingerprint": spec.fingerprint,
        "provider_id": result.provider_id,
        "privacy_mode": "injected_secure_aggregation_provider",
        "values": result.values,
        "participant_count": len(result.included_commitments),
        "total_sample_count": total_samples,
        "commitments": result.included_commitments,
        "dp_claimed": dp_claimed,
        "dp_provider_id": dp_provider_id,
        "proof": result.aggregate_proof_fingerprint,
        "privacy_metadata": result.privacy_metadata,
    }
    return AggregateLearningResult(
        round_fingerprint=spec.fingerprint,
        provider_id=result.provider_id,
        privacy_mode="injected_secure_aggregation_provider",
        values=result.values,
        participant_count=len(result.included_commitments),
        total_sample_count=total_samples,
        included_commitments=result.included_commitments,
        differential_privacy_applied=dp_claimed,
        differential_privacy_provider_id=dp_provider_id,
        proof_fingerprint=result.aggregate_proof_fingerprint,
        warnings=("cryptographic/privacy guarantees are those asserted by the injected provider and must be independently reviewed",),
        fingerprint=hashlib.sha256(_canonical(payload)).hexdigest(),
    )


__all__ = [
    "AggregateLearningResult",
    "AggregateRoundSpec",
    "BoundedContribution",
    "DifferentialPrivacyNoiseProvider",
    "LocalReferenceAggregator",
    "ParticipantCommitment",
    "ProviderAggregateResult",
    "SecureAggregationProvider",
    "validate_provider_result",
]
